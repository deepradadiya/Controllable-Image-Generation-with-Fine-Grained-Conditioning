"""
Inference Module

This module contains inference pipeline components:
- End-to-end inference pipeline combining SD1.5 with ControlNet
- DDIM sampling with ControlNet guidance
- Model loading and compatibility verification
- Batch inference and parameter controls
"""

from .pipeline import ControlNetInferencePipeline

__all__ = [
    "ControlNetInferencePipeline"
]