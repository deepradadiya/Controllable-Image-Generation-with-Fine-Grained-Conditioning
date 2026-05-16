"""
Dataset Downloader for ControlNet Training Pipeline.

Downloads the fusing/fill50k dataset from HuggingFace and saves the first 5000
examples locally as PNG images with associated text prompts. Includes image
validation, retry logic for network errors, and checkpoint/resume support
via file-existence checks.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class DownloadStats:
    """Statistics from the dataset download process."""

    total_valid: int
    total_invalid: int
    avg_width: float
    avg_height: float
    first_prompt: str


class DatasetDownloader:
    """Downloads fusing/fill50k from HuggingFace and saves locally.

    Saves images as zero-padded PNGs and prompts as a JSON mapping.
    Supports checkpoint/resume by skipping already-saved images.
    """

    def __init__(
        self,
        output_dir: str = "data/raw",
        num_samples: int = 5000,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ):
        self.output_dir = Path(output_dir)
        self.num_samples = num_samples
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Setup directories
        self.images_dir = self.output_dir / "images"
        self.prompts_path = self.output_dir / "prompts.json"
        self.bad_files_log = self.output_dir / "bad_files.log"

    def download(self) -> DownloadStats:
        """Download dataset, save images and prompts.

        Returns:
            DownloadStats with counts of valid/invalid images,
            average dimensions, and first prompt text.
        """
        # Create output directories
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset with retry logic
        dataset = self._load_dataset_with_retry()

        # Select first num_samples examples
        dataset = dataset.select(range(min(self.num_samples, len(dataset))))

        # Process and save
        prompts = {}
        valid_count = 0
        invalid_count = 0
        total_width = 0
        total_height = 0
        first_prompt: Optional[str] = None

        for idx in tqdm(range(len(dataset)), desc="Downloading dataset"):
            example = dataset[idx]
            filename = self._format_filename(idx)

            # Extract image and prompt
            image = example["image"]
            prompt = example["text"]

            if first_prompt is None:
                first_prompt = prompt

            # Save image (with checkpoint: skip if exists)
            image_path = self.images_dir / filename
            if image_path.exists():
                # Already saved, validate and count
                if self._validate_image(image, filename):
                    valid_count += 1
                    total_width += image.width
                    total_height += image.height
                    prompts[filename] = prompt
                else:
                    invalid_count += 1
                continue

            # Save and validate
            if self._save_image(image, idx):
                if self._validate_image(image, filename):
                    valid_count += 1
                    total_width += image.width
                    total_height += image.height
                    prompts[filename] = prompt
                else:
                    invalid_count += 1
            else:
                invalid_count += 1

        # Save prompts JSON
        with open(self.prompts_path, "w") as f:
            json.dump(prompts, f, indent=2)

        # Compute statistics
        avg_width = total_width / valid_count if valid_count > 0 else 0.0
        avg_height = total_height / valid_count if valid_count > 0 else 0.0

        stats = DownloadStats(
            total_valid=valid_count,
            total_invalid=invalid_count,
            avg_width=avg_width,
            avg_height=avg_height,
            first_prompt=first_prompt or "",
        )

        # Print final statistics
        print(f"\n{'='*50}")
        print("Download Complete - Statistics:")
        print(f"  Valid images: {stats.total_valid}")
        print(f"  Invalid images: {stats.total_invalid}")
        print(f"  Average width: {stats.avg_width:.1f} px")
        print(f"  Average height: {stats.avg_height:.1f} px")
        print(f"  First prompt: {stats.first_prompt}")
        print(f"{'='*50}")

        return stats

    def _load_dataset_with_retry(self):
        """Load dataset from HuggingFace with retry logic on network errors."""
        from datasets import load_dataset

        last_error = None
        for attempt in range(self.max_retries):
            try:
                dataset = load_dataset("fusing/fill50k", split="train")
                return dataset
            except (ConnectionError, OSError, TimeoutError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Network error on attempt {attempt + 1}/{self.max_retries}: {e}. "
                        f"Retrying in {self.retry_delay} seconds..."
                    )
                    time.sleep(self.retry_delay)
                else:
                    raise ConnectionError(
                        f"Failed to download dataset after {self.max_retries} attempts. "
                        f"Last error: {last_error}"
                    ) from last_error

    def _save_image(self, image: Image.Image, index: int) -> bool:
        """Save single image as zero-padded PNG.

        Args:
            image: PIL Image to save.
            index: Sequential index for filename.

        Returns:
            True if image was saved successfully, False otherwise.
        """
        filename = self._format_filename(index)
        image_path = self.images_dir / filename

        try:
            image.save(image_path, format="PNG")
            return True
        except (OSError, IOError) as e:
            self._log_bad_file(filename, f"save failed: {e}")
            return False

    def _validate_image(self, image: Image.Image, filename: str) -> bool:
        """Check image is not corrupted and has valid dimensions.

        Args:
            image: PIL Image to validate.
            filename: Filename for logging purposes.

        Returns:
            True if image has width > 0 and height > 0, False otherwise.
        """
        try:
            width, height = image.size
            if width > 0 and height > 0:
                return True
            else:
                self._log_bad_file(
                    filename,
                    f"invalid dimensions: width={width} height={height}",
                )
                return False
        except Exception as e:
            self._log_bad_file(filename, f"validation error: {e}")
            return False

    def _log_bad_file(self, filename: str, reason: str) -> None:
        """Log a failed image to bad_files.log.

        Args:
            filename: The filename that failed.
            reason: Description of the failure.
        """
        with open(self.bad_files_log, "a") as f:
            f.write(f"{filename} {reason}\n")

    @staticmethod
    def _format_filename(index: int) -> str:
        """Format index as zero-padded filename.

        Args:
            index: Integer index in range [0, 4999].

        Returns:
            Filename string like '00000.png'.
        """
        return f"{index:05d}.png"


if __name__ == "__main__":
    downloader = DatasetDownloader()
    stats = downloader.download()
