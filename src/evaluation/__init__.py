"""
Evaluation and Metrics Module

This module contains evaluation metrics and monitoring components:
- FID score computation using InceptionV3
- Condition alignment metrics for spatial conditioning
- Visual evaluation and comparison grids
- Training visualization and monitoring utilities
"""

from .compute_fid import FIDCalculator
from .condition_alignment import ConditionAlignmentMetrics
from .visual_grid import VisualGridGenerator

__all__ = [
    "FIDCalculator",
    "ConditionAlignmentMetrics",
    "VisualGridGenerator"
]