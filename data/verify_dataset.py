"""
Dataset Verification for ControlNet Training Pipeline.

Verifies completeness and integrity of all pipeline outputs by checking that
every sample has a source image, depth map, pose skeleton, edge map, and a
valid text prompt. Reports incomplete samples with specific details and prints
a final summary of samples ready for training.
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class SampleStatus:
    """Status of a single sample's completeness check."""

    stem: str
    has_source: bool = False
    has_depth: bool = False
    has_pose: bool = False
    has_edge: bool = False
    has_prompt: bool = False
    prompt_valid: bool = False
    all_loadable: bool = False


@dataclass
class VerificationReport:
    """Report summarizing dataset verification results."""

    total_samples: int = 0
    complete_samples: int = 0
    incomplete_samples: int = 0
    missing_details: Dict[str, List[str]] = field(default_factory=dict)


class DatasetVerifier:
    """Verifies completeness of all pipeline outputs.

    Checks that every sample enumerated from data/raw/images/ has
    corresponding depth, pose, and edge maps, plus a valid prompt entry.
    """

    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.images_dir = self.data_root / "raw" / "images"
        self.depth_dir = self.data_root / "depth"
        self.pose_dir = self.data_root / "pose"
        self.edges_dir = self.data_root / "edges"
        self.prompts_path = self.data_root / "raw" / "prompts.json"

    def verify(self) -> VerificationReport:
        """Check all samples have source image, depth, pose, edge, and prompt.

        Returns:
            VerificationReport with complete/incomplete counts and details.

        Raises:
            SystemExit: If prompts.json is missing or contains invalid JSON.
        """
        # Load prompts.json — halt if missing or invalid
        prompts = self._load_prompts()

        # Enumerate samples by listing filenames in data/raw/images/
        stems = self._enumerate_samples()

        report = VerificationReport(total_samples=len(stems))

        for stem in sorted(stems):
            status = self._check_sample(stem, prompts)
            issues = self._collect_issues(status)

            if issues:
                report.incomplete_samples += 1
                report.missing_details[stem] = issues
                print(f"INCOMPLETE: {stem} — {', '.join(issues)}")
            else:
                report.complete_samples += 1

        # Print final summary
        print(f"\n{report.complete_samples} samples ready for training")

        return report

    def _load_prompts(self) -> Dict[str, str]:
        """Load and validate prompts.json.

        Returns:
            Dictionary mapping filenames to prompt strings.

        Raises:
            SystemExit: If file is missing or contains invalid JSON.
        """
        if not self.prompts_path.exists():
            print(f"ERROR: prompts file not found: {self.prompts_path}")
            sys.exit(1)

        try:
            with open(self.prompts_path, "r") as f:
                prompts = json.load(f)
        except json.JSONDecodeError as e:
            print(f"ERROR: invalid JSON in prompts file: {e}")
            sys.exit(1)

        if not isinstance(prompts, dict):
            print("ERROR: prompts.json must contain a JSON object")
            sys.exit(1)

        return prompts

    def _enumerate_samples(self) -> List[str]:
        """List all sample stems from data/raw/images/.

        Returns:
            List of filename stems (without extension).
        """
        if not self.images_dir.exists():
            return []

        stems = [p.stem for p in self.images_dir.glob("*.png")]
        return stems

    def _check_sample(self, stem: str, prompts: Dict[str, str]) -> SampleStatus:
        """Check a single sample for all required files.

        Args:
            stem: Filename stem (e.g., '00042').
            prompts: Loaded prompts dictionary.

        Returns:
            SampleStatus with flags for each required component.
        """
        status = SampleStatus(stem=stem)
        filename = f"{stem}.png"

        # Check source image exists
        source_path = self.images_dir / filename
        status.has_source = source_path.exists()

        # Check depth map
        depth_path = self.depth_dir / filename
        status.has_depth = self._validate_image_file(depth_path)

        # Check pose skeleton
        pose_path = self.pose_dir / filename
        status.has_pose = self._validate_image_file(pose_path)

        # Check edge map
        edge_path = self.edges_dir / filename
        status.has_edge = self._validate_image_file(edge_path)

        # Check prompt entry
        status.has_prompt = filename in prompts
        if status.has_prompt:
            prompt_value = prompts[filename]
            status.prompt_valid = (
                isinstance(prompt_value, str)
                and len(prompt_value) >= 5
            )
        else:
            status.prompt_valid = False

        # All loadable if all conditioning maps passed validation
        status.all_loadable = status.has_depth and status.has_pose and status.has_edge

        return status

    def _validate_image_file(self, path: Path) -> bool:
        """Check file is loadable, >0 bytes, >=256x256.

        Args:
            path: Path to the image file to validate.

        Returns:
            True if file exists, is loadable, >0 bytes, and >=256x256 pixels.
        """
        if not path.exists():
            return False

        # Check file size > 0 bytes
        if path.stat().st_size == 0:
            return False

        # Try to load and check dimensions
        try:
            with Image.open(path) as img:
                img.verify()
            # Re-open after verify (verify can leave file in bad state)
            with Image.open(path) as img:
                width, height = img.size
                if width < 256 or height < 256:
                    return False
        except Exception:
            return False

        return True

    def _collect_issues(self, status: SampleStatus) -> List[str]:
        """Collect list of issues for a sample.

        Args:
            status: SampleStatus from checking a sample.

        Returns:
            List of human-readable issue descriptions. Empty if sample is complete.
        """
        issues = []

        if not status.has_source:
            issues.append("missing source image")
        if not status.has_depth:
            issues.append("missing/invalid depth map")
        if not status.has_pose:
            issues.append("missing/invalid pose skeleton")
        if not status.has_edge:
            issues.append("missing/invalid edge map")
        if not status.has_prompt:
            issues.append("missing prompt entry")
        elif not status.prompt_valid:
            issues.append("invalid prompt (too short or empty)")

        return issues


if __name__ == "__main__":
    verifier = DatasetVerifier()
    report = verifier.verify()
