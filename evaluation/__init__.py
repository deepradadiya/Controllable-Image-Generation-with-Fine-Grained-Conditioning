"""
Evaluation pipeline for ControlNet adapters.

This package provides tools for evaluating trained ControlNet adapters using:
- FID score computation against COCO 2017 validation set
- Condition alignment measurement (SSIM, correlation, keypoint distance)
- Visual comparison grid generation
- Pipeline orchestration with CLI interface
"""

from evaluation.config import EvaluationConfig, ensure_output_dir, load_test_prompts

__all__ = [
    "EvaluationConfig",
    "ensure_output_dir",
    "load_test_prompts",
]
