"""
Application Module

This module contains the web application components:
- Gradio web interface for HuggingFace Spaces
- Interactive generation controls and model management
- Model lifecycle management for demo deployment
"""

from .controls import (
    GenerationParameters,
    build_controls_interface,
    create_generation_controls,
    create_generation_trigger,
    create_pipeline_generate_fn,
    create_side_by_side_display,
    launch_controls_demo,
    parse_image_size,
)
from .gradio_app import create_gradio_app
from .model_manager import ModelManager, ManagerConfig, ModelLoadError, ModelStatus, ModelInfo

__all__ = [
    "create_gradio_app",
    "ModelManager",
    "ManagerConfig",
    "ModelLoadError",
    "ModelStatus",
    "ModelInfo",
    "GenerationParameters",
    "build_controls_interface",
    "create_generation_controls",
    "create_generation_trigger",
    "create_pipeline_generate_fn",
    "create_side_by_side_display",
    "launch_controls_demo",
    "parse_image_size",
]
