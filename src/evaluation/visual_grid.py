"""
Visual Evaluation and Comparison Grid Module

This module generates side-by-side comparison grids for evaluating ControlNet
image generation quality. It supports displaying condition maps, generated images,
and reference images in organized grids with labels and titles.

Additionally, it provides automated visual quality assessment metrics including
SSIM (Structural Similarity Index) and LPIPS (Learned Perceptual Image Patch Similarity).

Key Features:
- Comparison grid generation with condition map, generated image, and reference
- Configurable grid layout with labels and titles
- Multiple samples in a single grid
- Automated visual quality metrics (SSIM, LPIPS)
- Save grids to disk or return as PIL Images
- Memory-efficient processing for Colab T4 GPU constraints

Requirements Validated: 5.3, 5.4
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    """A single evaluation sample containing condition map, generated image, and reference.

    Attributes:
        condition_map: The spatial conditioning input (depth, pose, or edge map)
        generated_image: The image produced by the ControlNet model
        reference_image: The ground truth reference image
        prompt: The text prompt used for generation
        condition_type: Type of conditioning ('depth', 'pose', or 'edge')
        metadata: Additional metadata for the sample
    """

    condition_map: Image.Image
    generated_image: Image.Image
    reference_image: Image.Image
    prompt: str = ""
    condition_type: str = "unknown"
    metadata: Dict = field(default_factory=dict)


@dataclass
class VisualQualityMetrics:
    """Results from automated visual quality assessment.

    Attributes:
        ssim_score: Structural Similarity Index (0 to 1, higher is better)
        lpips_score: Learned Perceptual Image Patch Similarity (lower is better)
        mse: Mean Squared Error between generated and reference
        psnr: Peak Signal-to-Noise Ratio in dB (higher is better)
    """

    ssim_score: float
    lpips_score: float
    mse: float
    psnr: float

    def __str__(self) -> str:
        return (
            f"SSIM: {self.ssim_score:.4f} | "
            f"LPIPS: {self.lpips_score:.4f} | "
            f"PSNR: {self.psnr:.2f} dB"
        )


class VisualQualityAssessor:
    """Automated visual quality assessment using SSIM and LPIPS metrics.

    This class computes perceptual quality metrics between generated images
    and reference images to provide quantitative evaluation scores.
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the visual quality assessor.

        Args:
            device: Device to run LPIPS model on. Defaults to CUDA if available.
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._lpips_model = None

    @property
    def lpips_model(self):
        """Lazy-load LPIPS model to avoid unnecessary memory usage."""
        if self._lpips_model is None:
            try:
                import lpips

                self._lpips_model = lpips.LPIPS(net="alex").to(self.device)
                self._lpips_model.eval()
                logger.info("LPIPS model loaded successfully")
            except ImportError:
                logger.warning(
                    "lpips package not available. LPIPS scores will be set to -1.0"
                )
                self._lpips_model = None
            except Exception as e:
                logger.warning(f"Failed to load LPIPS model: {e}")
                self._lpips_model = None
        return self._lpips_model

    def compute_ssim(
        self,
        image1: Image.Image,
        image2: Image.Image,
    ) -> float:
        """
        Compute Structural Similarity Index between two images.

        SSIM measures the perceived quality difference between two images
        based on luminance, contrast, and structure comparisons.

        Args:
            image1: First image (generated)
            image2: Second image (reference)

        Returns:
            SSIM score between 0 and 1 (higher is better)
        """
        from skimage.metrics import structural_similarity

        # Convert to same size if needed
        if image1.size != image2.size:
            image2 = image2.resize(image1.size, Image.LANCZOS)

        # Convert to numpy arrays
        arr1 = np.array(image1.convert("RGB")).astype(np.float64)
        arr2 = np.array(image2.convert("RGB")).astype(np.float64)

        # Compute SSIM with multichannel support
        ssim_value = structural_similarity(
            arr1, arr2, channel_axis=2, data_range=255.0
        )

        return float(ssim_value)

    def compute_lpips(
        self,
        image1: Image.Image,
        image2: Image.Image,
    ) -> float:
        """
        Compute LPIPS (Learned Perceptual Image Patch Similarity) between two images.

        LPIPS uses a pre-trained neural network to measure perceptual similarity.
        Lower scores indicate more similar images.

        Args:
            image1: First image (generated)
            image2: Second image (reference)

        Returns:
            LPIPS score (lower is better). Returns -1.0 if LPIPS is unavailable.
        """
        model = self.lpips_model
        if model is None:
            return -1.0

        # Convert to same size if needed
        if image1.size != image2.size:
            image2 = image2.resize(image1.size, Image.LANCZOS)

        # Convert PIL images to tensors in [-1, 1] range (LPIPS expected format)
        def pil_to_tensor(img: Image.Image) -> torch.Tensor:
            arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
            # Normalize to [-1, 1]
            arr = arr * 2.0 - 1.0
            # Convert to (1, C, H, W)
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            return tensor.to(self.device)

        tensor1 = pil_to_tensor(image1)
        tensor2 = pil_to_tensor(image2)

        with torch.no_grad():
            lpips_value = model(tensor1, tensor2)

        return float(lpips_value.item())

    def compute_mse(self, image1: Image.Image, image2: Image.Image) -> float:
        """
        Compute Mean Squared Error between two images.

        Args:
            image1: First image (generated)
            image2: Second image (reference)

        Returns:
            MSE value (lower is better)
        """
        if image1.size != image2.size:
            image2 = image2.resize(image1.size, Image.LANCZOS)

        arr1 = np.array(image1.convert("RGB")).astype(np.float64)
        arr2 = np.array(image2.convert("RGB")).astype(np.float64)

        mse_value = np.mean((arr1 - arr2) ** 2)
        return float(mse_value)

    def compute_psnr(self, image1: Image.Image, image2: Image.Image) -> float:
        """
        Compute Peak Signal-to-Noise Ratio between two images.

        Args:
            image1: First image (generated)
            image2: Second image (reference)

        Returns:
            PSNR value in dB (higher is better). Returns inf if images are identical.
        """
        mse_value = self.compute_mse(image1, image2)
        if mse_value == 0:
            return float("inf")

        max_pixel = 255.0
        psnr_value = 10.0 * np.log10((max_pixel ** 2) / mse_value)
        return float(psnr_value)

    def assess(
        self,
        generated_image: Image.Image,
        reference_image: Image.Image,
    ) -> VisualQualityMetrics:
        """
        Run full visual quality assessment on a generated/reference image pair.

        Args:
            generated_image: The image produced by the model
            reference_image: The ground truth reference image

        Returns:
            VisualQualityMetrics with SSIM, LPIPS, MSE, and PSNR scores
        """
        ssim = self.compute_ssim(generated_image, reference_image)
        lpips_score = self.compute_lpips(generated_image, reference_image)
        mse = self.compute_mse(generated_image, reference_image)
        psnr = self.compute_psnr(generated_image, reference_image)

        return VisualQualityMetrics(
            ssim_score=ssim,
            lpips_score=lpips_score,
            mse=mse,
            psnr=psnr,
        )

    def batch_assess(
        self,
        samples: List[EvaluationSample],
    ) -> List[VisualQualityMetrics]:
        """
        Run visual quality assessment on a batch of evaluation samples.

        Args:
            samples: List of EvaluationSample objects

        Returns:
            List of VisualQualityMetrics, one per sample
        """
        results = []
        for sample in samples:
            metrics = self.assess(sample.generated_image, sample.reference_image)
            results.append(metrics)
        return results


class VisualGridGenerator:
    """Generates comparison grids for visual evaluation of ControlNet outputs.

    Creates organized grids showing condition maps, generated images, and reference
    images side-by-side with labels and titles for easy comparison.
    """

    def __init__(
        self,
        cell_size: Tuple[int, int] = (256, 256),
        padding: int = 10,
        label_height: int = 30,
        title_height: int = 50,
        background_color: Tuple[int, int, int] = (255, 255, 255),
        text_color: Tuple[int, int, int] = (0, 0, 0),
        font_size: int = 16,
        title_font_size: int = 22,
    ):
        """
        Initialize the visual grid generator.

        Args:
            cell_size: Size (width, height) for each image cell in the grid
            padding: Padding between cells in pixels
            label_height: Height reserved for column labels
            title_height: Height reserved for the grid title
            background_color: RGB background color for the grid
            text_color: RGB text color for labels
            font_size: Font size for column labels
            title_font_size: Font size for the grid title
        """
        self.cell_size = cell_size
        self.padding = padding
        self.label_height = label_height
        self.title_height = title_height
        self.background_color = background_color
        self.text_color = text_color
        self.font_size = font_size
        self.title_font_size = title_font_size

    def _get_font(self, size: int) -> ImageFont.ImageFont:
        """Get a font at the specified size, falling back to default if needed."""
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except (OSError, IOError):
            try:
                return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
            except (OSError, IOError):
                return ImageFont.load_default()

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize an image to the configured cell size."""
        return image.resize(self.cell_size, Image.LANCZOS)

    def _draw_centered_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        x_center: int,
        y: int,
        font: ImageFont.ImageFont,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw text centered at the given x position."""
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        x = x_center - text_width // 2
        draw.text((x, y), text, fill=color, font=font)

    def generate_comparison_grid(
        self,
        samples: List[EvaluationSample],
        title: Optional[str] = None,
        show_metrics: bool = False,
        metrics: Optional[List[VisualQualityMetrics]] = None,
        column_labels: Optional[List[str]] = None,
    ) -> Image.Image:
        """
        Generate a comparison grid showing condition map, generated image, and reference.

        Creates a grid with one row per sample and three columns:
        - Column 1: Condition map (spatial conditioning input)
        - Column 2: Generated image (ControlNet output)
        - Column 3: Reference image (ground truth)

        Args:
            samples: List of EvaluationSample objects to display
            title: Optional title for the grid
            show_metrics: Whether to display quality metrics below each row
            metrics: Pre-computed metrics (if None and show_metrics=True, will compute)
            column_labels: Custom column labels (default: Condition Map, Generated, Reference)

        Returns:
            PIL Image containing the comparison grid
        """
        if not samples:
            raise ValueError("No samples provided for grid generation")

        if column_labels is None:
            column_labels = ["Condition Map", "Generated", "Reference"]

        num_rows = len(samples)
        num_cols = 3

        # Calculate grid dimensions
        cell_w, cell_h = self.cell_size
        metrics_height = 20 if show_metrics else 0

        grid_width = (
            self.padding
            + num_cols * (cell_w + self.padding)
        )
        grid_height = (
            (self.title_height if title else 0)
            + self.label_height
            + num_rows * (cell_h + self.padding + metrics_height)
            + self.padding
        )

        # Create grid canvas
        grid = Image.new("RGB", (grid_width, grid_height), self.background_color)
        draw = ImageDraw.Draw(grid)

        label_font = self._get_font(self.font_size)
        title_font = self._get_font(self.title_font_size)
        metrics_font = self._get_font(max(10, self.font_size - 4))

        y_offset = 0

        # Draw title
        if title:
            self._draw_centered_text(
                draw, title, grid_width // 2, self.padding, title_font, self.text_color
            )
            y_offset += self.title_height

        # Draw column labels
        for col_idx, label in enumerate(column_labels):
            x_center = self.padding + col_idx * (cell_w + self.padding) + cell_w // 2
            self._draw_centered_text(
                draw, label, x_center, y_offset + 5, label_font, self.text_color
            )
        y_offset += self.label_height

        # Draw each sample row
        for row_idx, sample in enumerate(samples):
            row_y = y_offset + row_idx * (cell_h + self.padding + metrics_height)

            # Prepare images for this row
            images = [
                self._resize_image(sample.condition_map.convert("RGB")),
                self._resize_image(sample.generated_image.convert("RGB")),
                self._resize_image(sample.reference_image.convert("RGB")),
            ]

            # Paste images into grid
            for col_idx, img in enumerate(images):
                x = self.padding + col_idx * (cell_w + self.padding)
                grid.paste(img, (x, row_y))

            # Draw metrics below the row if requested
            if show_metrics and metrics and row_idx < len(metrics):
                metric = metrics[row_idx]
                metric_text = str(metric)
                metric_y = row_y + cell_h + 2
                self._draw_centered_text(
                    draw,
                    metric_text,
                    grid_width // 2,
                    metric_y,
                    metrics_font,
                    (80, 80, 80),
                )

        return grid

    def generate_single_comparison(
        self,
        sample: EvaluationSample,
        title: Optional[str] = None,
        show_metrics: bool = False,
        assessor: Optional[VisualQualityAssessor] = None,
    ) -> Image.Image:
        """
        Generate a comparison grid for a single sample.

        Convenience method for generating a grid with one sample.

        Args:
            sample: Single EvaluationSample to display
            title: Optional title for the grid
            show_metrics: Whether to display quality metrics
            assessor: VisualQualityAssessor instance for computing metrics

        Returns:
            PIL Image containing the single-row comparison grid
        """
        metrics = None
        if show_metrics:
            if assessor is None:
                assessor = VisualQualityAssessor()
            metric = assessor.assess(sample.generated_image, sample.reference_image)
            metrics = [metric]

        if title is None and sample.prompt:
            title = f'"{sample.prompt}" ({sample.condition_type})'

        return self.generate_comparison_grid(
            samples=[sample],
            title=title,
            show_metrics=show_metrics,
            metrics=metrics,
        )

    def save_grid(
        self,
        grid: Image.Image,
        output_path: Union[str, Path],
        format: Optional[str] = None,
        quality: int = 95,
    ) -> Path:
        """
        Save a generated grid to disk.

        Args:
            grid: The PIL Image grid to save
            output_path: Path where the grid should be saved
            format: Image format (inferred from extension if None)
            quality: JPEG quality (1-100, only used for JPEG format)

        Returns:
            Path to the saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_kwargs = {}
        if format:
            save_kwargs["format"] = format
        if output_path.suffix.lower() in (".jpg", ".jpeg"):
            save_kwargs["quality"] = quality

        grid.save(output_path, **save_kwargs)
        logger.info(f"Grid saved to: {output_path}")
        return output_path


def generate_visual_comparison(
    samples: List[EvaluationSample],
    title: Optional[str] = None,
    show_metrics: bool = True,
    output_path: Optional[Union[str, Path]] = None,
    cell_size: Tuple[int, int] = (256, 256),
    device: Optional[torch.device] = None,
) -> Image.Image:
    """
    High-level function to generate a visual comparison grid with quality metrics.

    This is the main entry point for generating evaluation grids. It creates a
    side-by-side comparison of condition maps, generated images, and references,
    optionally annotated with automated quality metrics.

    Args:
        samples: List of EvaluationSample objects to evaluate
        title: Optional title for the comparison grid
        show_metrics: Whether to compute and display quality metrics
        output_path: Optional path to save the grid (also returns the image)
        cell_size: Size of each image cell in the grid
        device: Device for LPIPS computation

    Returns:
        PIL Image containing the comparison grid

    Example:
        >>> sample = EvaluationSample(
        ...     condition_map=depth_map_image,
        ...     generated_image=generated_img,
        ...     reference_image=reference_img,
        ...     prompt="a cat sitting on a couch",
        ...     condition_type="depth"
        ... )
        >>> grid = generate_visual_comparison([sample], title="Depth ControlNet Evaluation")
    """
    generator = VisualGridGenerator(cell_size=cell_size)

    metrics = None
    if show_metrics:
        assessor = VisualQualityAssessor(device=device)
        metrics = assessor.batch_assess(samples)

    grid = generator.generate_comparison_grid(
        samples=samples,
        title=title,
        show_metrics=show_metrics,
        metrics=metrics,
    )

    if output_path:
        generator.save_grid(grid, output_path)

    return grid


def generate_evaluation_report(
    samples: List[EvaluationSample],
    output_dir: Union[str, Path],
    title: str = "ControlNet Evaluation Report",
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Generate a complete evaluation report with visual grids and numerical scores.

    Produces both visual comparison grids and aggregated numerical metrics,
    saving results to the specified output directory.

    Args:
        samples: List of EvaluationSample objects to evaluate
        output_dir: Directory to save report outputs
        title: Title for the evaluation report
        device: Device for metric computation

    Returns:
        Dictionary containing:
            - 'grid_path': Path to the saved comparison grid
            - 'metrics': List of per-sample VisualQualityMetrics
            - 'summary': Dictionary with aggregated metric statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute metrics
    assessor = VisualQualityAssessor(device=device)
    metrics = assessor.batch_assess(samples)

    # Generate comparison grid
    generator = VisualGridGenerator()
    grid = generator.generate_comparison_grid(
        samples=samples,
        title=title,
        show_metrics=True,
        metrics=metrics,
    )

    # Save grid
    grid_path = generator.save_grid(grid, output_dir / "comparison_grid.png")

    # Compute summary statistics
    ssim_scores = [m.ssim_score for m in metrics]
    lpips_scores = [m.lpips_score for m in metrics if m.lpips_score >= 0]
    psnr_scores = [m.psnr for m in metrics if m.psnr != float("inf")]

    summary = {
        "num_samples": len(samples),
        "ssim_mean": float(np.mean(ssim_scores)) if ssim_scores else 0.0,
        "ssim_std": float(np.std(ssim_scores)) if ssim_scores else 0.0,
        "lpips_mean": float(np.mean(lpips_scores)) if lpips_scores else -1.0,
        "lpips_std": float(np.std(lpips_scores)) if lpips_scores else 0.0,
        "psnr_mean": float(np.mean(psnr_scores)) if psnr_scores else 0.0,
        "psnr_std": float(np.std(psnr_scores)) if psnr_scores else 0.0,
    }

    # Log summary
    logger.info(f"Evaluation Report Summary ({len(samples)} samples):")
    logger.info(f"  SSIM: {summary['ssim_mean']:.4f} ± {summary['ssim_std']:.4f}")
    if summary["lpips_mean"] >= 0:
        logger.info(f"  LPIPS: {summary['lpips_mean']:.4f} ± {summary['lpips_std']:.4f}")
    logger.info(f"  PSNR: {summary['psnr_mean']:.2f} ± {summary['psnr_std']:.2f} dB")

    return {
        "grid_path": grid_path,
        "metrics": metrics,
        "summary": summary,
    }
