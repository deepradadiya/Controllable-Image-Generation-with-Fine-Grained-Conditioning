"""
Depth Map Extraction for ControlNet Training Pipeline.

A depth map is a grayscale image where brighter pixels indicate closer proximity
to the camera and darker pixels represent objects further away. This module uses
DPT (Dense Prediction Transformer), a Vision Transformer architecture trained by
Intel to estimate per-pixel depth from a single RGB image. The model produces
continuous depth predictions that are normalized to the 0-255 range for use as
conditioning maps in ControlNet training.
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DepthMapExtractor:
    """Extracts depth maps using Intel DPT-Large with batch processing.

    Processes source images through the DPT-Large model to generate
    grayscale depth maps. Supports checkpoint/resume by skipping
    already-processed images and processes in batches for GPU efficiency.
    """

    def __init__(
        self,
        input_dir: str = "data/raw/images",
        output_dir: str = "data/depth",
        batch_size: int = 8,
        device: str = "cuda",
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.device = device

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load model and processor
        self._load_model()

    def _load_model(self) -> None:
        """Load Intel DPT-Large model and processor from HuggingFace."""
        import torch
        from transformers import DPTForDepthEstimation, DPTImageProcessor

        self.processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        self.model.to(self.device)
        self.model.eval()
        self.torch = torch

    def extract_all(self) -> None:
        """Process all images, skipping already-processed ones.

        Reads source images from input_dir, generates depth maps using
        DPT-Large inference, and saves results to output_dir. Images
        with existing output files are skipped for checkpoint/resume.
        """
        # Get all source images
        image_paths = sorted(self.input_dir.glob("*.png"))

        if not image_paths:
            logger.warning(f"No PNG images found in {self.input_dir}")
            return

        # Filter to only unprocessed images (checkpoint/resume)
        to_process = []
        for path in image_paths:
            output_path = self.output_dir / path.name
            if not output_path.exists():
                to_process.append(path)

        logger.info(
            f"Found {len(image_paths)} images, "
            f"{len(image_paths) - len(to_process)} already processed, "
            f"{len(to_process)} to process"
        )

        if not to_process:
            print("All images already processed. Nothing to do.")
            self._display_samples(image_paths)
            return

        # Process in batches
        processed_paths = []
        failed_count = 0

        for i in tqdm(
            range(0, len(to_process), self.batch_size),
            desc="Extracting depth maps",
            unit="batch",
        ):
            batch_paths = to_process[i : i + self.batch_size]

            try:
                depth_maps = self._process_batch(batch_paths)

                for path, depth_map in zip(batch_paths, depth_maps):
                    if depth_map is not None:
                        output_path = self.output_dir / path.name
                        depth_image = Image.fromarray(depth_map, mode="L")
                        depth_image.save(output_path)
                        processed_paths.append(path)
                    else:
                        failed_count += 1
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Try individual images in the failed batch
                for path in batch_paths:
                    try:
                        depth_maps = self._process_batch([path])
                        if depth_maps[0] is not None:
                            output_path = self.output_dir / path.name
                            depth_image = Image.fromarray(depth_maps[0], mode="L")
                            depth_image.save(output_path)
                            processed_paths.append(path)
                        else:
                            failed_count += 1
                    except Exception as img_e:
                        logger.error(
                            f"Failed to process {path.name}: {img_e}"
                        )
                        failed_count += 1

        print(f"\nDepth extraction complete:")
        print(f"  Processed: {len(processed_paths)}")
        print(f"  Failed: {failed_count}")
        print(f"  Skipped (already done): {len(image_paths) - len(to_process)}")

        # Display sample results
        all_processed = [
            p for p in image_paths if (self.output_dir / p.name).exists()
        ]
        self._display_samples(all_processed)

    def _process_batch(self, image_paths: List[Path]) -> List[np.ndarray]:
        """Process a batch of images through DPT.

        Args:
            image_paths: List of paths to source images.

        Returns:
            List of 512x512 grayscale depth maps (uint8, 0-255).
            Returns None in the list for images that failed processing.
        """
        results = []
        images = []
        valid_indices = []

        for idx, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize((512, 512))
                images.append(img)
                valid_indices.append(idx)
            except Exception as e:
                logger.error(f"Failed to load {path.name}: {e}")
                results.append(None)

        if not images:
            return [None] * len(image_paths)

        # Run DPT inference
        with self.torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to 512x512 and normalize each depth map
        predicted_depth = self.torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(512, 512),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1)

        # Build results list maintaining order
        final_results = [None] * len(image_paths)

        for i, orig_idx in enumerate(valid_indices):
            depth_np = predicted_depth[i].cpu().numpy()
            normalized = self._normalize_depth(depth_np)
            final_results[orig_idx] = normalized

        return final_results

    def _normalize_depth(self, raw_depth: np.ndarray) -> np.ndarray:
        """Per-image min-max normalization to [0, 255] uint8.

        Args:
            raw_depth: 2D float array of raw depth predictions.

        Returns:
            2D uint8 array with values in [0, 255].
        """
        depth_min = raw_depth.min()
        depth_max = raw_depth.max()

        if depth_max - depth_min > 0:
            normalized = (raw_depth - depth_min) / (depth_max - depth_min)
        else:
            normalized = np.zeros_like(raw_depth)

        return (normalized * 255).astype(np.uint8)

    def _display_samples(self, processed_paths: List[Path]) -> None:
        """Display 3 depth maps at equal intervals with originals.

        Selects 3 images at equal intervals from the processed set
        and displays each depth map alongside its original image
        using matplotlib.

        Args:
            processed_paths: List of source image paths that have
                corresponding depth maps in the output directory.
        """
        import matplotlib.pyplot as plt

        if len(processed_paths) < 3:
            logger.warning("Not enough processed images to display 3 samples")
            return

        # Select 3 at equal intervals
        n = len(processed_paths)
        indices = [0, n // 2, n - 1]
        sample_paths = [processed_paths[i] for i in indices]

        fig, axes = plt.subplots(3, 2, figsize=(10, 15))

        for row, path in enumerate(sample_paths):
            # Original image
            original = Image.open(path).convert("RGB")
            axes[row, 0].imshow(original)
            axes[row, 0].set_title(f"Original: {path.name}")
            axes[row, 0].axis("off")

            # Depth map
            depth_path = self.output_dir / path.name
            depth = Image.open(depth_path)
            axes[row, 1].imshow(depth, cmap="gray")
            axes[row, 1].set_title(f"Depth: {path.name}")
            axes[row, 1].axis("off")

        plt.tight_layout()
        plt.savefig(self.output_dir / "depth_samples.png", dpi=100)
        plt.show()
        print("Sample depth maps displayed and saved to depth_samples.png")


if __name__ == "__main__":
    extractor = DepthMapExtractor()
    extractor.extract_all()
