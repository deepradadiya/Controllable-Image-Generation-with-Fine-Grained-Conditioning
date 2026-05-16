"""
Evaluation and Metrics Module

This module contains evaluation metrics and monitoring components:
- FID score computation using InceptionV3
- Condition alignment metrics for spatial conditioning
- Visual evaluation and comparison grids
- Training visualization and monitoring utilities
"""

from .compute_fid import FIDCalculator
from .condition_alignment import (
    AlignmentResult,
    BatchAlignmentResult,
    ConditionAlignmentEvaluator,
    DepthAlignmentEvaluator,
    EdgeAlignmentEvaluator,
    PoseAlignmentEvaluator,
)
from .visual_grid import (
    EvaluationSample,
    VisualGridGenerator,
    VisualQualityAssessor,
    VisualQualityMetrics,
    generate_evaluation_report,
    generate_visual_comparison,
)

__all__ = [
    "FIDCalculator",
    "AlignmentResult",
    "BatchAlignmentResult",
    "ConditionAlignmentEvaluator",
    "DepthAlignmentEvaluator",
    "PoseAlignmentEvaluator",
    "EdgeAlignmentEvaluator",
    "EvaluationSample",
    "VisualGridGenerator",
    "VisualQualityAssessor",
    "VisualQualityMetrics",
    "generate_visual_comparison",
    "generate_evaluation_report",
]