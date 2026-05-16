"""
Training System Module

This module contains the training infrastructure for ControlNet models:
- Memory-optimized training orchestrator with T4 GPU constraints
- Diffusion loss computation with conditioning
- Training scripts for depth, pose, and edge conditioning
- Gradient checkpointing and mixed precision support
"""

from .trainer import ControlNetTrainer
from .losses import DiffusionLoss, ControlNetLoss

__all__ = [
    "ControlNetTrainer",
    "DiffusionLoss",
    "ControlNetLoss",
]