"""
Configuration and shared utilities for the evaluation pipeline.

Provides the EvaluationConfig dataclass with all configurable parameters,
and helper functions for output directory management and test prompt loading.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation pipeline.

    Contains all configurable parameters for FID computation,
    condition alignment measurement, and visual grid generation.
    """

    output_dir: str = "evaluation/results"
    num_fid_samples: int = 1000
    num_alignment_samples: int = 100
    num_grid_prompts: int = 20
    batch_size: int = 32
    condition_types: List[str] = field(default_factory=lambda: ["depth", "pose", "edge"])
    coco_val_dir: str = "data/raw/coco_val2017"
    checkpoint_dir: str = "models/trained"
    guidance_scale: float = 7.5
    num_inference_steps: int = 20
    seed: int = 42


def ensure_output_dir(output_dir: str = "evaluation/results") -> Path:
    """Create the output directory if it doesn't exist.

    Args:
        output_dir: Path to the output directory. Defaults to 'evaluation/results/'.

    Returns:
        Path object for the created/existing output directory.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_test_prompts() -> List[str]:
    """Return a list of 20 diverse test prompts for evaluation.

    The prompts cover a variety of subjects, styles, and compositions
    suitable for evaluating image generation quality across different
    conditioning types (depth, pose, edge).

    Returns:
        List of 20 diverse text prompts for image generation.
    """
    return [
        "A woman standing in a sunlit garden with colorful flowers",
        "A man sitting on a wooden bench in a park",
        "A cat sleeping on a windowsill with curtains",
        "A modern kitchen with stainless steel appliances",
        "A person riding a bicycle on a city street",
        "A dog running through a grassy field",
        "A cozy living room with a fireplace and bookshelves",
        "A dancer performing on a stage with dramatic lighting",
        "A mountain landscape with a lake in the foreground",
        "A chef preparing food in a restaurant kitchen",
        "A child playing with toys in a colorful bedroom",
        "A street scene with cars and pedestrians at sunset",
        "A yoga practitioner in a warrior pose on a beach",
        "An old stone bridge over a flowing river",
        "A musician playing guitar on a street corner",
        "A modern office space with large windows and plants",
        "A couple walking hand in hand through autumn leaves",
        "A still life arrangement of fruits on a wooden table",
        "A surfer riding a wave in the ocean",
        "A medieval castle on a hilltop under cloudy skies",
    ]
