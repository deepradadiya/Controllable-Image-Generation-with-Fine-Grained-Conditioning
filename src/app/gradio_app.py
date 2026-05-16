"""
Gradio Web Interface for ControlNet Image Generation

This module implements a HuggingFace Space compatible web interface for interactive
ControlNet image generation. Users can upload images, select condition types,
provide text prompts, and generate conditioned images with real-time feedback.

Key Features:
- HuggingFace Spaces compatible Gradio Blocks interface
- Image upload with automatic condition map extraction
- Dropdown for condition type selection (depth, pose, edge)
- Text prompt input for guiding image generation
- Side-by-side display of condition map and generated image
- Error handling for missing models or failed extractions

Requirements Addressed:
- 8.1: HuggingFace Space compatible web interface
- 8.2: Image upload with condition type selection
- 8.3: Automatic condition map extraction
"""

import logging
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Condition type options
CONDITION_TYPES = ["depth", "pose", "edge"]

# Default generation parameters
DEFAULT_NUM_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_CONDITIONING_STRENGTH = 1.0


def _extract_condition_map(
    image: Image.Image, condition_type: str
) -> Tuple[Optional[np.ndarray], Optional[Image.Image], str]:
    """
    Extract a condition map from the uploaded image based on the selected type.

    Args:
        image: Source PIL Image uploaded by the user.
        condition_type: One of 'depth', 'pose', or 'edge'.

    Returns:
        Tuple of (raw_condition_map_array, display_image, status_message).
        On failure, raw array and display image may be None.
    """
    if image is None:
        return None, None, "No image provided."

    try:
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        if condition_type == "depth":
            from src.data.extract_depth import DepthExtractor, DepthExtractionConfig

            config = DepthExtractionConfig(
                device="auto",
                precision="fp16",
                target_size=(512, 512),
            )
            extractor = DepthExtractor(config)
            result = extractor.extract(image)

            if not result.success:
                return None, None, f"Depth extraction failed: {result.error_message}"

            # Convert single-channel depth map to displayable image
            depth_map = result.depth_map  # shape (H, W, 1), range [0, 1]
            display_array = (depth_map[:, :, 0] * 255).astype(np.uint8)
            display_image = Image.fromarray(display_array, mode="L").convert("RGB")
            return depth_map, display_image, "Depth map extracted successfully."

        elif condition_type == "pose":
            from src.data.extract_pose import PoseExtractor

            extractor = PoseExtractor(
                prefer_dwpose=True,
                fallback_to_mediapipe=True,
                speed_critical=False,
            )
            result = extractor.extract(image)

            if result is None:
                return None, None, "Pose extraction returned None."

            # result is a numpy array (H, W, 3) with rendered skeleton
            if isinstance(result, np.ndarray):
                pose_map = result
            elif isinstance(result, Image.Image):
                pose_map = np.array(result)
            else:
                return None, None, f"Unexpected pose result type: {type(result)}"

            # Normalize to [0, 1] for consistency
            if pose_map.dtype == np.uint8:
                condition_array = pose_map.astype(np.float32) / 255.0
            else:
                condition_array = pose_map.astype(np.float32)

            display_image = Image.fromarray(
                (condition_array * 255).astype(np.uint8)
            )
            return condition_array, display_image, "Pose skeleton extracted successfully."

        elif condition_type == "edge":
            from src.data.extract_edges import CannyEdgeExtractor, EdgeExtractionConfig

            config = EdgeExtractionConfig(
                adaptive_threshold=True,
                output_channels=3,
                normalize_output=True,
            )
            extractor = CannyEdgeExtractor(config)
            result = extractor.extract(image)

            if not result.success:
                return None, None, f"Edge extraction failed: {result.error_message}"

            edge_map = result.edge_map  # shape (H, W, 3), range [0, 1]
            display_image = Image.fromarray(
                (edge_map * 255).astype(np.uint8)
            )
            return edge_map, display_image, "Edge map extracted successfully."

        else:
            return None, None, f"Unknown condition type: {condition_type}"

    except ImportError as e:
        error_msg = (
            f"Missing dependency for {condition_type} extraction: {e}. "
            f"Please install required packages."
        )
        logger.error(error_msg)
        return None, None, error_msg
    except Exception as e:
        error_msg = f"Condition extraction failed: {e}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return None, None, error_msg


def _generate_image(
    source_image: Optional[Image.Image],
    condition_type: str,
    prompt: str,
    num_steps: int,
    guidance_scale: float,
    conditioning_strength: float,
) -> Tuple[Optional[Image.Image], Optional[Image.Image], str]:
    """
    Generate an image using the ControlNet inference pipeline.

    Args:
        source_image: Uploaded source image for condition extraction.
        condition_type: Type of conditioning (depth, pose, edge).
        prompt: Text prompt for generation.
        num_steps: Number of inference steps.
        guidance_scale: Classifier-free guidance scale.
        conditioning_strength: ControlNet conditioning strength.

    Returns:
        Tuple of (condition_map_display, generated_image, status_message).
    """
    if source_image is None:
        return None, None, "Please upload a source image."

    if not prompt or not prompt.strip():
        return None, None, "Please enter a text prompt."

    # Step 1: Extract condition map
    condition_array, condition_display, extract_status = _extract_condition_map(
        source_image, condition_type
    )

    if condition_array is None:
        return None, None, f"Condition extraction failed: {extract_status}"

    # Step 2: Run inference pipeline
    try:
        from src.inference.pipeline import (
            ControlNetInferencePipeline,
            GenerationParams,
            InferenceConfig,
        )

        config = InferenceConfig(
            condition_type=condition_type,
            enable_memory_optimization=True,
        )
        pipeline = ControlNetInferencePipeline(config)

        params = GenerationParams(
            prompt=prompt.strip(),
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            conditioning_scale=float(conditioning_strength),
            height=512,
            width=512,
        )

        result = pipeline.generate(
            prompt=prompt.strip(),
            condition_image=condition_array,
            params=params,
            condition_type=condition_type,
        )

        if result.images:
            generated_image = result.images[0]
            status = (
                f"Image generated successfully in {result.generation_time_seconds:.1f}s. "
                f"Seed: {result.seed_used}"
            )
            return condition_display, generated_image, status
        else:
            return condition_display, None, "Generation produced no images."

    except ImportError as e:
        error_msg = f"Missing inference dependency: {e}"
        logger.error(error_msg)
        return condition_display, None, error_msg
    except RuntimeError as e:
        error_msg = f"Model loading or inference failed: {e}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return condition_display, None, error_msg
    except Exception as e:
        error_msg = f"Generation failed: {e}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return condition_display, None, error_msg


def _on_image_upload(
    image: Optional[Image.Image], condition_type: str
) -> Tuple[Optional[Image.Image], str]:
    """
    Callback when an image is uploaded. Automatically extracts the condition map.

    Args:
        image: Uploaded PIL Image.
        condition_type: Selected condition type.

    Returns:
        Tuple of (condition_map_display_image, status_message).
    """
    if image is None:
        return None, "No image uploaded."

    _, display_image, status = _extract_condition_map(image, condition_type)
    return display_image, status


def create_gradio_app() -> "gr.Blocks":
    """
    Create and return the Gradio Blocks application.

    This function builds the complete Gradio interface with:
    - Image upload component
    - Condition type dropdown
    - Text prompt input
    - Generation parameter controls
    - Condition map display
    - Generated image display
    - Status messages

    Returns:
        A Gradio Blocks app instance ready to launch.
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError(
            "Gradio is required for the web interface. "
            "Install it with: pip install gradio"
        )

    with gr.Blocks(
        title="ControlNet Image Generation",
    ) as app:
        gr.Markdown(
            """
            # ControlNet Image Generation Demo

            Generate images conditioned on spatial control maps (depth, pose, edge).
            Upload a source image, select a condition type, enter a text prompt,
            and click **Generate** to create a conditioned image.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Input controls
                source_image = gr.Image(
                    label="Source Image",
                    type="pil",
                    height=300,
                )
                condition_type = gr.Dropdown(
                    choices=CONDITION_TYPES,
                    value="depth",
                    label="Condition Type",
                    info="Select the type of spatial conditioning to apply.",
                )
                prompt = gr.Textbox(
                    label="Text Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=2,
                )

                with gr.Accordion("Generation Parameters", open=False):
                    num_steps = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=DEFAULT_NUM_STEPS,
                        step=1,
                        label="Inference Steps",
                        info="More steps = higher quality but slower.",
                    )
                    guidance_scale = gr.Slider(
                        minimum=1.0,
                        maximum=20.0,
                        value=DEFAULT_GUIDANCE_SCALE,
                        step=0.5,
                        label="Guidance Scale",
                        info="Higher values follow the prompt more closely.",
                    )
                    conditioning_strength = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=DEFAULT_CONDITIONING_STRENGTH,
                        step=0.1,
                        label="Conditioning Strength",
                        info="How strongly the condition map influences generation.",
                    )

                generate_btn = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1):
                # Output displays
                condition_map_output = gr.Image(
                    label="Extracted Condition Map",
                    type="pil",
                    height=300,
                )
                generated_image_output = gr.Image(
                    label="Generated Image",
                    type="pil",
                    height=300,
                )
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2,
                )

        # Wire up automatic condition map extraction on image upload or type change
        source_image.change(
            fn=_on_image_upload,
            inputs=[source_image, condition_type],
            outputs=[condition_map_output, status_output],
        )
        condition_type.change(
            fn=_on_image_upload,
            inputs=[source_image, condition_type],
            outputs=[condition_map_output, status_output],
        )

        # Wire up generation button
        generate_btn.click(
            fn=_generate_image,
            inputs=[
                source_image,
                condition_type,
                prompt,
                num_steps,
                guidance_scale,
                conditioning_strength,
            ],
            outputs=[condition_map_output, generated_image_output, status_output],
        )

    return app


# Standalone entry point for HuggingFace Spaces or local development
if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(share=False)
