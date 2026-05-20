"""
Visual Grid Generator for ControlNet Evaluation.

Generates 4-column comparison grids for qualitative evaluation of trained
ControlNet adapters. Each grid shows:
  Column 1: Original image
  Column 2: Condition map (depth/pose/edge)
  Column 3: Generated image with ControlNet conditioning
  Column 4: Generated image without ControlNet (vanilla SD1.5 baseline)

Grids include column headers and row labels (text prompts) for easy
identification and comparison.

Requirements Validated: 3.1, 3.2, 3.5
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from model.pipeline import ControlNetPipeline

logger = logging.getLogger(__name__)


class EvaluationGridGenerator:
    """Generates 4-column comparison grids for qualitative evaluation.

    Columns: Original Image | Condition Map | With ControlNet | Without ControlNet

    The generator uses the provided ControlNetPipeline to produce conditioned
    images, and generates unconditioned (baseline) images by passing a zero
    condition map to simulate vanilla SD1.5 output.
    """

    # Column header labels
    COLUMN_HEADERS = ["Original", "Condition Map", "With ControlNet", "Without ControlNet"]
    NUM_COLUMNS = 4

    def __init__(
        self,
        pipeline: ControlNetPipeline,
        cell_size: Tuple[int, int] = (256, 256),
        output_dir: str = "evaluation/results",
    ):
        """Initialize the EvaluationGridGenerator.

        Args:
            pipeline: Loaded ControlNetPipeline for image generation.
            cell_size: Size (width, height) for each image cell in the grid.
                Each cell will be at least this size. Defaults to (256, 256).
            output_dir: Directory where generated grid images will be saved.
                Defaults to 'evaluation/results'.
        """
        self.pipeline = pipeline
        self.cell_size = cell_size
        self.output_dir = Path(output_dir)

        # Layout configuration
        self.padding = 10
        self.header_height = 40
        self.row_label_width = 150
        self.background_color = (255, 255, 255)
        self.text_color = (0, 0, 0)
        self.header_font_size = 16
        self.label_font_size = 12

    def _get_font(self, size: int) -> ImageFont.ImageFont:
        """Get a font at the specified size, with fallback to default.

        Args:
            size: Desired font size in points.

        Returns:
            An ImageFont instance at the requested size, or the default font
            if no TrueType font is available.
        """
        font_paths = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
        for font_path in font_paths:
            try:
                return ImageFont.truetype(font_path, size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize an image to the configured cell size.

        Args:
            image: PIL Image to resize.

        Returns:
            Resized PIL Image matching self.cell_size dimensions.
        """
        return image.convert("RGB").resize(self.cell_size, Image.LANCZOS)

    def _truncate_text(self, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
        """Truncate text to fit within a given pixel width.

        Adds an ellipsis ('...') if the text is too long to fit.

        Args:
            text: The text string to potentially truncate.
            font: The font used for rendering (to measure text width).
            max_width: Maximum allowed width in pixels.

        Returns:
            The original text if it fits, or a truncated version with '...' appended.
        """
        # Use a temporary image for text measurement
        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)

        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]

        if text_width <= max_width:
            return text

        # Binary search for the right truncation point
        for end in range(len(text), 0, -1):
            truncated = text[:end] + "..."
            bbox = draw.textbbox((0, 0), truncated, font=font)
            if bbox[2] - bbox[0] <= max_width:
                return truncated

        return "..."

    def _draw_centered_text(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        x_center: int,
        y: int,
        font: ImageFont.ImageFont,
    ) -> None:
        """Draw text centered horizontally at the given x position.

        Args:
            draw: ImageDraw instance to draw on.
            text: Text string to render.
            x_center: The x-coordinate to center the text around.
            y: The y-coordinate for the top of the text.
            font: Font to use for rendering.
        """
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        x = x_center - text_width // 2
        draw.text((x, y), text, fill=self.text_color, font=font)

    def _generate_with_controlnet(
        self,
        prompt: str,
        condition_map: Image.Image,
        condition_type: str,
        seed: int,
    ) -> Image.Image:
        """Generate an image using the ControlNet pipeline.

        Args:
            prompt: Text prompt for generation.
            condition_map: Spatial condition image (depth/pose/edge).
            condition_type: Type of conditioning ('depth', 'pose', or 'edge').
            seed: Random seed for reproducibility.

        Returns:
            Generated PIL Image.
        """
        generated = self.pipeline(
            text_prompt=prompt,
            condition_image=condition_map,
            condition_type=condition_type,
            seed=seed,
        )
        return generated

    def _generate_without_controlnet(
        self,
        prompt: str,
        condition_type: str,
        seed: int,
    ) -> Image.Image:
        """Generate an image without ControlNet conditioning (baseline).

        Uses a zero (black) condition map so the ControlNet contribution is
        effectively zero, simulating vanilla SD1.5 generation.

        Args:
            prompt: Text prompt for generation.
            condition_type: Condition type to pass to the pipeline (required
                by the interface but the zero map nullifies its effect).
            seed: Random seed for reproducibility.

        Returns:
            Generated PIL Image (baseline without spatial conditioning).
        """
        # Create a zero (black) condition map to nullify ControlNet contribution
        zero_condition = Image.new("RGB", (512, 512), (0, 0, 0))
        generated = self.pipeline(
            text_prompt=prompt,
            condition_image=zero_condition,
            condition_type=condition_type,
            seed=seed,
        )
        return generated

    def generate_grid(
        self,
        condition_type: str,
        original_images: List[Image.Image],
        condition_maps: List[Image.Image],
        prompts: List[str],
        num_rows: int = 20,
        seed: int = 42,
    ) -> Image.Image:
        """Generate a 4-column comparison grid for one condition type.

        Creates a grid image with the following columns:
          1. Original image
          2. Condition map (depth/pose/edge)
          3. Generated with ControlNet
          4. Generated without ControlNet (vanilla SD1.5)

        Each row corresponds to one prompt/image pair. Column headers are
        rendered at the top, and row labels (truncated prompts) are shown
        on the left side.

        Args:
            condition_type: One of 'depth', 'pose', or 'edge'.
            original_images: List of original source images.
            condition_maps: List of condition map images corresponding to originals.
            prompts: List of text prompts for generation.
            num_rows: Number of rows to include in the grid (default 20).
            seed: Base random seed for reproducible generation.

        Returns:
            PIL Image containing the complete comparison grid.
        """
        # Limit to available data
        num_rows = min(num_rows, len(original_images), len(condition_maps), len(prompts))

        cell_w, cell_h = self.cell_size

        # Calculate grid dimensions
        # Width: row_label_width + padding + (4 cells with padding between them) + padding
        grid_width = (
            self.row_label_width
            + self.padding
            + self.NUM_COLUMNS * cell_w
            + (self.NUM_COLUMNS + 1) * self.padding
        )
        # Height: header + padding + (num_rows cells with padding between them) + padding
        grid_height = (
            self.header_height
            + self.padding
            + num_rows * cell_h
            + (num_rows + 1) * self.padding
        )

        # Create the grid canvas
        grid = Image.new("RGB", (grid_width, grid_height), self.background_color)
        draw = ImageDraw.Draw(grid)

        header_font = self._get_font(self.header_font_size)
        label_font = self._get_font(self.label_font_size)

        # Draw column headers
        content_start_x = self.row_label_width + self.padding
        for col_idx, header in enumerate(self.COLUMN_HEADERS):
            x_center = content_start_x + self.padding + col_idx * (cell_w + self.padding) + cell_w // 2
            y = (self.header_height - self.header_font_size) // 2
            self._draw_centered_text(draw, header, x_center, y, header_font)

        # Generate and place each row
        for row_idx in range(num_rows):
            row_y = self.header_height + self.padding + row_idx * (cell_h + self.padding)

            # Draw row label (truncated prompt)
            truncated_prompt = self._truncate_text(
                prompts[row_idx], label_font, self.row_label_width - self.padding
            )
            # Vertically center the label in the row
            label_y = row_y + (cell_h - self.label_font_size) // 2
            draw.text(
                (self.padding, label_y),
                truncated_prompt,
                fill=self.text_color,
                font=label_font,
            )

            # Column 1: Original image
            original_resized = self._resize_image(original_images[row_idx])
            col1_x = content_start_x + self.padding
            grid.paste(original_resized, (col1_x, row_y))

            # Column 2: Condition map
            condition_resized = self._resize_image(condition_maps[row_idx])
            col2_x = content_start_x + self.padding + (cell_w + self.padding)
            grid.paste(condition_resized, (col2_x, row_y))

            # Column 3: With ControlNet
            row_seed = seed + row_idx
            logger.info(
                f"Generating with ControlNet for row {row_idx + 1}/{num_rows}: "
                f"'{prompts[row_idx][:50]}...'"
            )
            with_controlnet = self._generate_with_controlnet(
                prompt=prompts[row_idx],
                condition_map=condition_maps[row_idx],
                condition_type=condition_type,
                seed=row_seed,
            )
            with_controlnet_resized = self._resize_image(with_controlnet)
            col3_x = content_start_x + self.padding + 2 * (cell_w + self.padding)
            grid.paste(with_controlnet_resized, (col3_x, row_y))

            # Column 4: Without ControlNet
            logger.info(
                f"Generating without ControlNet for row {row_idx + 1}/{num_rows}: "
                f"'{prompts[row_idx][:50]}...'"
            )
            without_controlnet = self._generate_without_controlnet(
                prompt=prompts[row_idx],
                condition_type=condition_type,
                seed=row_seed,
            )
            without_controlnet_resized = self._resize_image(without_controlnet)
            col4_x = content_start_x + self.padding + 3 * (cell_w + self.padding)
            grid.paste(without_controlnet_resized, (col4_x, row_y))

        logger.info(
            f"Generated {condition_type} grid with {num_rows} rows "
            f"({grid_width}x{grid_height} pixels)"
        )
        return grid

    def generate_combined_grid(
        self,
        original_images: List[Image.Image],
        condition_maps: Dict[str, List[Image.Image]],
        prompts: List[str],
        num_rows: int = 5,
        seed: int = 42,
    ) -> Image.Image:
        """Generate a combined grid showing all 3 condition types on the same inputs.

        The combined grid groups rows by condition type. For each condition type,
        it generates a section with a condition-type header followed by rows
        showing: Original | Condition Map | With ControlNet | Without ControlNet.

        Args:
            original_images: List of original source images.
            condition_maps: Dict mapping condition type ('depth', 'pose', 'edge')
                to a list of condition map images.
            prompts: List of text prompts for generation.
            num_rows: Number of rows per condition type (default 5).
            seed: Base random seed for reproducible generation.

        Returns:
            PIL Image containing the combined comparison grid with all
            available condition types.
        """
        # Determine which condition types are available
        available_types = [ct for ct in ["depth", "pose", "edge"] if ct in condition_maps]

        if not available_types:
            logger.warning("No condition types available for combined grid generation")
            # Return a minimal placeholder image
            placeholder = Image.new("RGB", (512, 64), self.background_color)
            draw = ImageDraw.Draw(placeholder)
            font = self._get_font(self.header_font_size)
            draw.text((10, 20), "No condition types available", fill=self.text_color, font=font)
            return placeholder

        # Limit rows to available data
        num_rows = min(num_rows, len(original_images), len(prompts))
        for ct in available_types:
            num_rows = min(num_rows, len(condition_maps[ct]))

        cell_w, cell_h = self.cell_size
        section_header_height = 30  # Height for condition type section headers

        # Calculate grid dimensions
        # Width: same as single grid
        grid_width = (
            self.row_label_width
            + self.padding
            + self.NUM_COLUMNS * cell_w
            + (self.NUM_COLUMNS + 1) * self.padding
        )

        # Height: column header + for each condition type: section header + rows
        num_sections = len(available_types)
        total_rows = num_sections * num_rows
        grid_height = (
            self.header_height
            + self.padding
            + num_sections * section_header_height
            + total_rows * cell_h
            + (total_rows + num_sections) * self.padding
        )

        # Create the grid canvas
        grid = Image.new("RGB", (grid_width, grid_height), self.background_color)
        draw = ImageDraw.Draw(grid)

        header_font = self._get_font(self.header_font_size)
        label_font = self._get_font(self.label_font_size)

        # Draw column headers
        content_start_x = self.row_label_width + self.padding
        for col_idx, header in enumerate(self.COLUMN_HEADERS):
            x_center = content_start_x + self.padding + col_idx * (cell_w + self.padding) + cell_w // 2
            y = (self.header_height - self.header_font_size) // 2
            self._draw_centered_text(draw, header, x_center, y, header_font)

        # Track current y position
        current_y = self.header_height + self.padding

        for condition_type in available_types:
            # Draw section header for this condition type
            section_label = f"— {condition_type.upper()} —"
            section_x_center = grid_width // 2
            section_label_y = current_y + (section_header_height - self.header_font_size) // 2
            self._draw_centered_text(draw, section_label, section_x_center, section_label_y, header_font)
            current_y += section_header_height + self.padding

            # Generate rows for this condition type
            for row_idx in range(num_rows):
                row_y = current_y

                # Draw row label (truncated prompt)
                truncated_prompt = self._truncate_text(
                    prompts[row_idx], label_font, self.row_label_width - self.padding
                )
                label_y = row_y + (cell_h - self.label_font_size) // 2
                draw.text(
                    (self.padding, label_y),
                    truncated_prompt,
                    fill=self.text_color,
                    font=label_font,
                )

                # Column 1: Original image
                original_resized = self._resize_image(original_images[row_idx])
                col1_x = content_start_x + self.padding
                grid.paste(original_resized, (col1_x, row_y))

                # Column 2: Condition map
                condition_resized = self._resize_image(condition_maps[condition_type][row_idx])
                col2_x = content_start_x + self.padding + (cell_w + self.padding)
                grid.paste(condition_resized, (col2_x, row_y))

                # Column 3: With ControlNet
                row_seed = seed + row_idx
                logger.info(
                    f"Combined grid [{condition_type}] generating with ControlNet "
                    f"row {row_idx + 1}/{num_rows}: '{prompts[row_idx][:50]}...'"
                )
                with_controlnet = self._generate_with_controlnet(
                    prompt=prompts[row_idx],
                    condition_map=condition_maps[condition_type][row_idx],
                    condition_type=condition_type,
                    seed=row_seed,
                )
                with_controlnet_resized = self._resize_image(with_controlnet)
                col3_x = content_start_x + self.padding + 2 * (cell_w + self.padding)
                grid.paste(with_controlnet_resized, (col3_x, row_y))

                # Column 4: Without ControlNet
                logger.info(
                    f"Combined grid [{condition_type}] generating without ControlNet "
                    f"row {row_idx + 1}/{num_rows}: '{prompts[row_idx][:50]}...'"
                )
                without_controlnet = self._generate_without_controlnet(
                    prompt=prompts[row_idx],
                    condition_type=condition_type,
                    seed=row_seed,
                )
                without_controlnet_resized = self._resize_image(without_controlnet)
                col4_x = content_start_x + self.padding + 3 * (cell_w + self.padding)
                grid.paste(without_controlnet_resized, (col4_x, row_y))

                current_y += cell_h + self.padding

        logger.info(
            f"Generated combined grid with {len(available_types)} condition types, "
            f"{num_rows} rows each ({grid_width}x{grid_height} pixels)"
        )
        return grid

    def save_all_grids(
        self,
        original_images: List[Image.Image],
        condition_maps: Dict[str, List[Image.Image]],
        prompts: List[str],
    ) -> List[str]:
        """Generate and save all grids (per-condition + combined) as lossless PNGs.

        For each condition type in condition_maps, generates a per-condition grid
        using generate_grid. Then generates a combined grid showing all condition
        types together. All grids are saved as lossless PNG files.

        If a condition type's pipeline fails during generation, that condition
        type is skipped with a warning and the remaining types are still processed.

        Args:
            original_images: List of original source images.
            condition_maps: Dict mapping condition type ('depth', 'pose', 'edge')
                to a list of condition map images.
            prompts: List of text prompts for generation.

        Returns:
            List of file paths (as strings) for all successfully saved grid images.
        """
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: List[str] = []
        successful_types: List[str] = []

        # Generate per-condition grids
        for condition_type in ["depth", "pose", "edge"]:
            if condition_type not in condition_maps:
                logger.warning(
                    f"Skipping {condition_type} grid: no condition maps provided"
                )
                continue

            try:
                logger.info(f"Generating per-condition grid for '{condition_type}'...")
                grid = self.generate_grid(
                    condition_type=condition_type,
                    original_images=original_images,
                    condition_maps=condition_maps[condition_type],
                    prompts=prompts,
                    num_rows=20,
                    seed=42,
                )
                output_path = self.output_dir / f"visual_grid_{condition_type}.png"
                grid.save(str(output_path), format="PNG", compress_level=0)
                saved_paths.append(str(output_path))
                successful_types.append(condition_type)
                logger.info(f"Saved {condition_type} grid to {output_path}")
            except Exception as e:
                logger.warning(
                    f"Failed to generate grid for condition type '{condition_type}': {e}. "
                    f"Skipping this condition type."
                )
                continue

        # Generate combined grid (only with successful condition types)
        if successful_types:
            try:
                # Filter condition_maps to only include successful types
                available_maps = {
                    ct: condition_maps[ct] for ct in successful_types
                }
                logger.info(
                    f"Generating combined grid with condition types: {successful_types}"
                )
                combined_grid = self.generate_combined_grid(
                    original_images=original_images,
                    condition_maps=available_maps,
                    prompts=prompts,
                    num_rows=5,
                    seed=42,
                )
                combined_path = self.output_dir / "visual_grid_combined.png"
                combined_grid.save(str(combined_path), format="PNG", compress_level=0)
                saved_paths.append(str(combined_path))
                logger.info(f"Saved combined grid to {combined_path}")
            except Exception as e:
                logger.warning(f"Failed to generate combined grid: {e}")
        else:
            logger.warning(
                "No condition types succeeded — skipping combined grid generation"
            )

        logger.info(f"Saved {len(saved_paths)} grid files total")
        return saved_paths
