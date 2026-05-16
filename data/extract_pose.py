"""
Pose Skeleton Extraction using DWPose for ControlNet Training Pipeline.

A pose skeleton is a stick figure rendering that shows detected human body keypoints
including shoulders, elbows, wrists, hips, knees, and ankles connected by colored
limb segments. DWPose is a whole-body pose estimation model that detects eighteen
body keypoints from a single RGB image and draws colored stick figures representing
the spatial limb connections between adjacent joints on a black background, producing
conditioning maps suitable for ControlNet pose-guided image generation.
"""

import logging
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PoseMapExtractor:
    """Extracts pose skeletons using DWPose from controlnet_aux.

    Processes source images to generate 512x512 RGB pose skeleton maps.
    Supports checkpoint/resume by skipping already-processed images.
    """

    def __init__(
        self,
        input_dir: str = "data/raw/images",
        output_dir: str = "data/pose",
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track which images had keypoints detected for display preference
        self._detected_paths: List[Path] = []
        self._all_processed_paths: List[Path] = []

        # Load DWPose model
        self._load_model()

    def _load_model(self) -> None:
        """Load DWPose model from controlnet_aux library."""
        from controlnet_aux import DWposeDetector

        self.detector = DWposeDetector.from_pretrained("lllyasviel/Annotators")
        logger.info("DWPose model loaded successfully")

    def extract_all(self) -> None:
        """Process all images, skipping already-processed ones."""
        # Gather source images
        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        source_images = sorted(
            p for p in self.input_dir.iterdir()
            if p.suffix.lower() in image_extensions
        )

        if not source_images:
            logger.warning(f"No images found in {self.input_dir}")
            return

        for image_path in tqdm(source_images, desc="Extracting pose maps"):
            output_path = self.output_dir / f"{image_path.stem}.png"

            # Skip if output already exists (checkpoint/resume)
            if output_path.exists():
                continue

            try:
                image = Image.open(image_path).convert("RGB")
                pose_map = self._process_image(image)

                # Save pose map
                pose_image = Image.fromarray(pose_map)
                pose_image.save(output_path, format="PNG")

                # Track for display
                self._all_processed_paths.append(output_path)
                if not self._is_blank(pose_map):
                    self._detected_paths.append(output_path)

            except Exception as e:
                logger.warning(f"Failed to process {image_path.name}: {e}")
                continue

        # Display sample results
        self._display_samples()

    def _process_image(self, image: Image.Image) -> np.ndarray:
        """Run DWPose inference on a single image.

        Args:
            image: PIL Image in RGB mode.

        Returns:
            512x512 RGB numpy array. Colored stick figure on black background,
            or blank black image if no keypoints detected.
        """
        # Run DWPose inference — returns a PIL Image with skeleton on black background
        pose_result = self.detector(
            image,
            detect_resolution=512,
            image_resolution=512,
            output_type="pil",
        )

        # Convert result to numpy array
        pose_array = np.array(pose_result)

        # Ensure correct shape (512x512 RGB)
        if pose_array.shape != (512, 512, 3):
            # Resize if needed
            pose_pil = Image.fromarray(pose_array).resize((512, 512))
            pose_array = np.array(pose_pil)

        # Check if any keypoints were detected (non-black pixels present)
        if self._is_blank(pose_array):
            # No keypoints detected — return explicit blank black image
            return np.zeros((512, 512, 3), dtype=np.uint8)

        return pose_array.astype(np.uint8)

    def _is_blank(self, pose_map: np.ndarray) -> bool:
        """Check if a pose map is blank (all zeros / no keypoints detected)."""
        return np.all(pose_map == 0)

    def _display_samples(self) -> None:
        """Display 3 pose skeletons (preferring detected poses) with originals."""
        import matplotlib.pyplot as plt

        # Prefer images where keypoints were detected
        if len(self._detected_paths) >= 3:
            sample_paths = self._detected_paths[:3]
        elif self._detected_paths:
            # Use all detected + fill from all processed
            sample_paths = list(self._detected_paths)
            for p in self._all_processed_paths:
                if p not in sample_paths:
                    sample_paths.append(p)
                if len(sample_paths) >= 3:
                    break
        else:
            sample_paths = self._all_processed_paths[:3]

        if not sample_paths:
            logger.info("No processed images to display")
            return

        num_samples = len(sample_paths)
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))

        if num_samples == 1:
            axes = [axes]

        for i, pose_path in enumerate(sample_paths):
            # Load original image
            original_path = self.input_dir / f"{pose_path.stem}.png"
            if original_path.exists():
                original = Image.open(original_path)
            else:
                # Try other extensions
                original = None
                for ext in [".jpg", ".jpeg", ".bmp", ".tiff"]:
                    alt_path = self.input_dir / f"{pose_path.stem}{ext}"
                    if alt_path.exists():
                        original = Image.open(alt_path)
                        break
                if original is None:
                    original = Image.new("RGB", (512, 512), (128, 128, 128))

            # Load pose map
            pose_map = Image.open(pose_path)

            axes[i][0].imshow(original)
            axes[i][0].set_title(f"Original: {pose_path.stem}")
            axes[i][0].axis("off")

            axes[i][1].imshow(pose_map)
            axes[i][1].set_title(f"Pose: {pose_path.stem}")
            axes[i][1].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    extractor = PoseMapExtractor()
    extractor.extract_all()
