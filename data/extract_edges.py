"""
Edge Map Extraction for ControlNet Training Pipeline.

An edge map is a binary image that highlights object outlines and boundaries within
a scene. Each pixel is either 0 (non-edge) or 255 (edge), producing a clean
representation of structural contours. The Canny edge detection algorithm identifies
these edges by finding rapid changes in pixel intensity using gradient magnitude
computation, non-maximum suppression, and hysteresis thresholding with a low and
high threshold pair to determine which gradient responses qualify as true edges.
"""

import logging
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EdgeMapExtractor:
    """Extracts edge maps using OpenCV Canny on CPU.

    Processes source images through Canny edge detection and saves
    binary edge maps preserving original image dimensions.
    Supports checkpoint/resume by skipping already-processed images.
    """

    def __init__(
        self,
        input_dir: str = "data/raw/images",
        output_dir: str = "data/edges",
        low_threshold: int = 100,
        high_threshold: int = 200,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        # low_threshold=100 means edges with gradient magnitude below 100 are discarded
        self.low_threshold = low_threshold
        # high_threshold=200 means edges with gradient magnitude above 200 are always kept
        self.high_threshold = high_threshold

    def extract_all(self) -> None:
        """Process all images on CPU, skipping already-processed ones."""
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Gather source images
        image_paths = sorted(self.input_dir.glob("*.png"))
        if not image_paths:
            logger.warning(f"No PNG images found in {self.input_dir}")
            return

        processed_paths: List[Path] = []

        for image_path in tqdm(image_paths, desc="Extracting edges", unit="img"):
            output_path = self.output_dir / image_path.name

            # Skip if output already exists (checkpoint/resume)
            if output_path.exists():
                continue

            # Process the image
            edge_map = self._process_image(image_path)
            if edge_map is None:
                continue

            # Save edge map as single-channel grayscale PNG
            cv2.imwrite(str(output_path), edge_map)
            processed_paths.append(image_path)

        # Display samples if any were processed
        if processed_paths:
            self._display_samples(processed_paths)

        print(f"\nEdge extraction complete. Processed {len(processed_paths)} new images.")

    def _process_image(self, image_path: Path) -> np.ndarray | None:
        """Apply Canny edge detection to a single image.

        Args:
            image_path: Path to the source image.

        Returns:
            Binary edge map with same dimensions as source (values 0 or 255),
            or None if the image could not be read.
        """
        # Read the image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Could not read image: {image_path.name}, skipping.")
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection with configured thresholds
        # low_threshold=100: edges with gradient magnitude below 100 are discarded
        # high_threshold=200: edges with gradient magnitude above 200 are always kept
        edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)

        return edges

    def _display_samples(self, processed_paths: List[Path]) -> None:
        """Display first 3 edge maps side by side with originals using matplotlib.

        Args:
            processed_paths: List of source image paths that were processed.
        """
        # Select first 3 processed images
        sample_paths = processed_paths[:3]
        num_samples = len(sample_paths)

        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i, image_path in enumerate(sample_paths):
            # Load original image (convert BGR to RGB for display)
            original = cv2.imread(str(image_path))
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

            # Load edge map
            edge_path = self.output_dir / image_path.name
            edge_map = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)

            # Display original
            axes[i, 0].imshow(original_rgb)
            axes[i, 0].set_title(f"Original: {image_path.name}")
            axes[i, 0].axis("off")

            # Display edge map
            axes[i, 1].imshow(edge_map, cmap="gray")
            axes[i, 1].set_title(f"Edge Map: {image_path.name}")
            axes[i, 1].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    extractor = EdgeMapExtractor()
    extractor.extract_all()
