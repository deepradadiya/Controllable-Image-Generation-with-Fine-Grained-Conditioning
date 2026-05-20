"""
HuggingFace Space Entry Point for ControlNet Image Generation

This module serves as the entry point for the HuggingFace Space deployment.
It imports and launches the enhanced Gradio application from the main source package.

HuggingFace Spaces with the Gradio SDK expect a module-level `demo` variable
that is a `gr.Blocks` (or `gr.Interface`) instance. The Space runtime will
automatically call `demo.launch()` when the Space starts.

This file also supports standalone execution for local testing:
    python deployment/app.py

Requirements Addressed:
- 9.1: HuggingFace Space with SDK type "gradio" and public visibility
- 9.2: Gradio application code file uploaded to the Space repository
"""

import sys
from pathlib import Path

# Add the project root to the Python path so that src/ imports work correctly
# in the HuggingFace Space environment where the source tree is available
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.app.gradio_app import create_gradio_app

# Create the Gradio app instance at module level.
# HuggingFace Spaces with gradio SDK auto-detect this variable and launch it.
demo = create_gradio_app()


def main():
    """Launch the Gradio application for local testing or direct execution."""
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
