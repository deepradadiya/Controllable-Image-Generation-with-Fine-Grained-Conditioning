"""
Training System Module

This module contains the training infrastructure for ControlNet models:
- Memory-optimized training orchestrator with T4 GPU constraints
- Diffusion loss computation with conditioning
- Training scripts for depth, pose, and edge conditioning
- Gradient checkpointing and mixed precision support
"""

from .trainer import ControlNetTrainer
from .losses import DiffusionLoss
from .train_depth import train_depth_controlnet
from .train_pose import train_pose_controlnet
from .train_edge import train_edge_controlnet

__all__ = [
    "ControlNetTrainer",
    "DiffusionLoss",
    "train_depth_controlnet",
    "train_pose_controlnet", 
    "train_edge_controlnet"
]