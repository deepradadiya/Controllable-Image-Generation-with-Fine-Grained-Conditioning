"""
Application Module

This module contains the web application and inference components:
- Gradio web interface for HuggingFace Spaces
- End-to-end inference pipeline
- Interactive generation controls and model management
"""

from .gradio_app import create_gradio_app
from .inference_pipeline import ControlNetInferencePipeline

__all__ = [
    "create_gradio_app",
    "ControlNetInferencePipeline"
]