"""
Model Architecture Module

This module contains the core ControlNet architecture implementation:
- ControlNet adapter architecture with encoder blocks
- UNet wrapper for ControlNet integration with SD1.5
- Model configuration and serialization utilities
- HuggingFace compatible model saving/loading
"""

from .controlnet import ControlNetModel
from .controlnet import create_controlnet_from_config

__all__ = [
    "ControlNetModel",
    "create_controlnet_from_config"
]