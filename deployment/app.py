"""
HuggingFace Space Entry Point for ControlNet Image Generation

This module serves as the entry point for the HuggingFace Space deployment.
It imports and launches the Gradio application from the main source package.

Usage:
    This file is automatically executed by HuggingFace Spaces when the Space starts.
    It can also be run locally for testing:
        python deployment/app.py
"""

import sys
from pathlib import Path

# Add the project root to the Python path so that src/ imports work correctly
# in the HuggingFace Space environment
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.app.gradio_app import create_gradio_app


def main():
    """Launch the Gradio application for HuggingFace Space deployment."""
    app = create_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
