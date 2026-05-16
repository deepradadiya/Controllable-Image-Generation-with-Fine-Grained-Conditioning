"""
Interactive Generation Controls for ControlNet Demo Application

This module implements the Gradio-based interactive controls for the ControlNet
image generation demo. It provides sliders for generation parameters, real-time
parameter updates, and a side-by-side display of condition maps and generated images.

Key Features:
- Sliders for num_inference_steps, guidance_scale, and conditioning_strength
- Seed input for reproducible generation
- Image size selection (256x256, 512x512, 768x768)
- Negative prompt text input
- Side-by-side display layout showing condition map and generated image
- Generate button that triggers inference with current parameters
- Status/progress indicator during generation

Requirements satisfied: 8.4, 8.5, 8.6
"""

import logging
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Constants
# ============================================================================

# Inference steps slider range
STEPS_MIN = 10
STEPS_MAX = 100
STEPS_DEFAULT = 50
STEPS_STEP = 1

# Guidance scale slider range
GUIDANCE_SCALE_MIN = 1.0
GUIDANCE_SCALE_MAX = 20.0
GUIDANCE_SCALE_DEFAULT = 7.5
GUIDANCE_SCALE_STEP = 0.5

# Conditioning strength slider range
CONDITIONING_STRENGTH_MIN = 0.0
CONDITIONING_STRENGTH_MAX = 2.0
CONDITIONING_STRENGTH_DEFAULT = 1.0
CONDITIONING_STRENGTH_STEP = 0.05

# Supported image sizes
IMAGE_SIZES = ["256x256", "512x512", "768x768"]
IMAGE_SIZE_DEFAULT = "512x512"

# Seed range
SEED_MIN = 0
SEED_MAX = 2147483647  # Max int32


@dataclass
class GenerationParameters:
    """Container for all generation parameters from the UI controls.

    Attributes:
        num_inference_steps: Number of DDIM sampling steps (10-100)
        guidance_scale: Classifier-free guidance scale (1.0-20.0)
        conditioning_strength: ControlNet conditioning strength (0.0-2.0)
        seed: Random seed for reproducibility (-1 for random)
        image_size: Output image dimensions as "WxH" string
        negative_prompt: Negative text prompt for guidance
    """
    num_inference_steps: int = STEPS_DEFAULT
    guidance_scale: float = GUIDANCE_SCALE_DEFAULT
    conditioning_strength: float = CONDITIONING_STRENGTH_DEFAULT
    seed: int = -1
    image_size: str = IMAGE_SIZE_DEFAULT
    negative_prompt: str = ""

    @property
    def width(self) -> int:
        """Extract width from image_size string."""
        return int(self.image_size.split("x")[0])

    @property
    def height(self) -> int:
        """Extract height from image_size string."""
        return int(self.image_size.split("x")[1])

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary for pipeline integration."""
        return {
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "conditioning_scale": self.conditioning_strength,
            "seed": self.seed if self.seed >= 0 else None,
            "width": self.width,
            "height": self.height,
            "negative_prompt": self.negative_prompt,
        }


def parse_image_size(size_str: str) -> Tuple[int, int]:
    """Parse image size string into (width, height) tuple.

    Args:
        size_str: Size string in format "WxH" (e.g., "512x512")

    Returns:
        Tuple of (width, height)

    Raises:
        ValueError: If size string format is invalid
    """
    try:
        parts = size_str.split("x")
        if len(parts) != 2:
            raise ValueError(f"Invalid size format: {size_str}")
        width = int(parts[0])
        height = int(parts[1])
        if width <= 0 or height <= 0:
            raise ValueError(f"Dimensions must be positive: {size_str}")
        return width, height
    except (ValueError, IndexError) as e:
        raise ValueError(f"Cannot parse image size '{size_str}': {e}")


def create_generation_controls() -> Dict[str, Any]:
    """Create the Gradio UI components for generation parameter controls.

    Returns a dictionary of Gradio components that can be integrated into
    a larger Gradio interface. Components include sliders, inputs, and
    display elements for controlling ControlNet image generation.

    Returns:
        Dictionary mapping component names to Gradio component instances
    """
    components = {}

    with gr.Column():
        gr.Markdown("### Generation Parameters")

        # Number of inference steps slider
        components["num_inference_steps"] = gr.Slider(
            minimum=STEPS_MIN,
            maximum=STEPS_MAX,
            value=STEPS_DEFAULT,
            step=STEPS_STEP,
            label="Inference Steps",
            info="Number of DDIM sampling steps. More steps = higher quality but slower.",
        )

        # Guidance scale slider
        components["guidance_scale"] = gr.Slider(
            minimum=GUIDANCE_SCALE_MIN,
            maximum=GUIDANCE_SCALE_MAX,
            value=GUIDANCE_SCALE_DEFAULT,
            step=GUIDANCE_SCALE_STEP,
            label="Guidance Scale (CFG)",
            info="Classifier-free guidance strength. Higher = more prompt adherence.",
        )

        # Conditioning strength slider
        components["conditioning_strength"] = gr.Slider(
            minimum=CONDITIONING_STRENGTH_MIN,
            maximum=CONDITIONING_STRENGTH_MAX,
            value=CONDITIONING_STRENGTH_DEFAULT,
            step=CONDITIONING_STRENGTH_STEP,
            label="Conditioning Strength",
            info="ControlNet conditioning strength. Higher = stronger spatial control.",
        )

        # Seed input for reproducibility
        components["seed"] = gr.Number(
            value=-1,
            label="Seed",
            info="Random seed for reproducibility. Use -1 for random.",
            precision=0,
        )

        # Image size selection
        components["image_size"] = gr.Dropdown(
            choices=IMAGE_SIZES,
            value=IMAGE_SIZE_DEFAULT,
            label="Image Size",
            info="Output image dimensions (width x height).",
        )

        # Negative prompt input
        components["negative_prompt"] = gr.Textbox(
            value="",
            label="Negative Prompt",
            placeholder="low quality, blurry, distorted, deformed...",
            info="Describe what you don't want in the generated image.",
            lines=2,
        )

    return components


def create_side_by_side_display() -> Dict[str, Any]:
    """Create the side-by-side display layout for condition map and generated image.

    Provides a two-column layout showing the input condition map on the left
    and the generated output image on the right, enabling visual comparison
    of how well the generation follows the conditioning.

    Returns:
        Dictionary mapping display component names to Gradio component instances
    """
    display_components = {}

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### Condition Map")
            display_components["condition_display"] = gr.Image(
                label="Condition Map",
                type="pil",
                interactive=False,
                show_label=False,
            )

        with gr.Column(scale=1):
            gr.Markdown("#### Generated Image")
            display_components["generated_display"] = gr.Image(
                label="Generated Image",
                type="pil",
                interactive=False,
                show_label=False,
            )

    return display_components


def create_generation_trigger() -> Dict[str, Any]:
    """Create the generate button and status/progress indicator.

    Returns:
        Dictionary with generate button and status components
    """
    trigger_components = {}

    with gr.Row():
        trigger_components["generate_btn"] = gr.Button(
            value="Generate Image",
            variant="primary",
            scale=2,
        )
        trigger_components["clear_btn"] = gr.Button(
            value="Clear",
            variant="secondary",
            scale=1,
        )

    # Status/progress indicator
    trigger_components["status"] = gr.Textbox(
        value="Ready",
        label="Status",
        interactive=False,
        max_lines=2,
    )

    return trigger_components


def build_controls_interface(
    generate_fn: Optional[Callable] = None,
) -> gr.Blocks:
    """Build the complete interactive generation controls interface.

    Creates a full Gradio Blocks interface with all generation controls,
    side-by-side display, and generation triggering. This can be used
    standalone or integrated into a larger application.

    Args:
        generate_fn: Optional callback function for image generation.
            If provided, it will be connected to the generate button.
            Expected signature:
                generate_fn(
                    condition_image: PIL.Image,
                    prompt: str,
                    num_inference_steps: int,
                    guidance_scale: float,
                    conditioning_strength: float,
                    seed: int,
                    image_size: str,
                    negative_prompt: str,
                ) -> Tuple[PIL.Image, PIL.Image, str]
            Returns: (condition_map_display, generated_image, status_message)

    Returns:
        Gradio Blocks interface with all controls wired up
    """
    with gr.Blocks(
        title="ControlNet Generation Controls",
    ) as interface:
        gr.Markdown("## ControlNet Image Generation")
        gr.Markdown(
            "Upload a condition map or image, adjust parameters, "
            "and generate controlled images."
        )

        with gr.Row():
            # Left column: Input and controls
            with gr.Column(scale=1):
                # Condition image input
                condition_input = gr.Image(
                    label="Input Condition Map",
                    type="pil",
                    sources=["upload", "clipboard"],
                )

                # Text prompt
                prompt_input = gr.Textbox(
                    value="",
                    label="Text Prompt",
                    placeholder="Describe the image you want to generate...",
                    lines=3,
                )

                # Generation parameter controls
                controls = create_generation_controls()

                # Generate and clear buttons with status
                triggers = create_generation_trigger()

            # Right column: Side-by-side display
            with gr.Column(scale=1):
                display = create_side_by_side_display()

                # Generation metadata
                with gr.Accordion("Generation Info", open=False):
                    generation_info = gr.JSON(
                        label="Parameters Used",
                        value={},
                    )

        # Wire up the generate button if callback provided
        if generate_fn is not None:
            _connect_generate_handler(
                generate_fn=generate_fn,
                condition_input=condition_input,
                prompt_input=prompt_input,
                controls=controls,
                triggers=triggers,
                display=display,
                generation_info=generation_info,
            )

        # Wire up clear button
        triggers["clear_btn"].click(
            fn=_clear_outputs,
            inputs=[],
            outputs=[
                display["condition_display"],
                display["generated_display"],
                triggers["status"],
                generation_info,
            ],
        )

    return interface


def _connect_generate_handler(
    generate_fn: Callable,
    condition_input: gr.Image,
    prompt_input: gr.Textbox,
    controls: Dict[str, Any],
    triggers: Dict[str, Any],
    display: Dict[str, Any],
    generation_info: gr.JSON,
) -> None:
    """Connect the generate button to the generation callback.

    Wires up all input components to the generate function and routes
    outputs to the display components.

    Args:
        generate_fn: The generation callback function
        condition_input: Condition image input component
        prompt_input: Text prompt input component
        controls: Dictionary of control components
        triggers: Dictionary of trigger components (button, status)
        display: Dictionary of display components
        generation_info: JSON component for generation metadata
    """
    # Define the wrapped handler that manages status updates
    def _wrapped_generate(
        condition_image,
        prompt,
        num_inference_steps,
        guidance_scale,
        conditioning_strength,
        seed,
        image_size,
        negative_prompt,
    ):
        """Wrapped generation handler with error handling and status updates."""
        if condition_image is None:
            return (
                None,
                None,
                "Error: Please upload a condition map image.",
                {},
            )

        if not prompt or not prompt.strip():
            return (
                condition_image,
                None,
                "Error: Please enter a text prompt.",
                {},
            )

        start_time = time.time()

        try:
            # Call the actual generation function
            condition_display, generated_image, status = generate_fn(
                condition_image,
                prompt,
                int(num_inference_steps),
                float(guidance_scale),
                float(conditioning_strength),
                int(seed),
                image_size,
                negative_prompt,
            )

            elapsed = time.time() - start_time

            # Build generation info
            info = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(guidance_scale),
                "conditioning_strength": float(conditioning_strength),
                "seed": int(seed),
                "image_size": image_size,
                "generation_time_seconds": round(elapsed, 2),
            }

            status_msg = f"Done in {elapsed:.1f}s. {status}" if status else f"Done in {elapsed:.1f}s"

            return condition_display, generated_image, status_msg, info

        except Exception as e:
            logger.error(f"Generation failed: {e}\n{traceback.format_exc()}")
            elapsed = time.time() - start_time
            return (
                condition_image,
                None,
                f"Error ({elapsed:.1f}s): {str(e)}",
                {"error": str(e)},
            )

    # Connect the button click event
    triggers["generate_btn"].click(
        fn=_wrapped_generate,
        inputs=[
            condition_input,
            prompt_input,
            controls["num_inference_steps"],
            controls["guidance_scale"],
            controls["conditioning_strength"],
            controls["seed"],
            controls["image_size"],
            controls["negative_prompt"],
        ],
        outputs=[
            display["condition_display"],
            display["generated_display"],
            triggers["status"],
            generation_info,
        ],
    )


def _clear_outputs() -> Tuple[None, None, str, Dict]:
    """Clear all output displays and reset status.

    Returns:
        Tuple of (None, None, "Ready", {}) to clear all output components
    """
    return None, None, "Ready", {}


def create_pipeline_generate_fn(pipeline) -> Callable:
    """Create a generation function that wraps the ControlNetInferencePipeline.

    This factory function creates a callback compatible with the controls
    interface that delegates to the actual inference pipeline.

    Args:
        pipeline: An instance of ControlNetInferencePipeline (from src.inference.pipeline)

    Returns:
        A callable suitable for use with build_controls_interface()
    """
    from src.inference.pipeline import GenerationParams

    def generate(
        condition_image: Image.Image,
        prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        conditioning_strength: float,
        seed: int,
        image_size: str,
        negative_prompt: str,
    ) -> Tuple[Optional[Image.Image], Optional[Image.Image], str]:
        """Generate an image using the ControlNet inference pipeline.

        Args:
            condition_image: Input condition map as PIL Image
            prompt: Text prompt for generation
            num_inference_steps: Number of DDIM sampling steps
            guidance_scale: Classifier-free guidance scale
            conditioning_strength: ControlNet conditioning strength
            seed: Random seed (-1 for random)
            image_size: Output size as "WxH" string
            negative_prompt: Negative prompt text

        Returns:
            Tuple of (condition_map_display, generated_image, status_message)
        """
        # Parse image size
        width, height = parse_image_size(image_size)

        # Build generation parameters
        params = GenerationParams(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            conditioning_scale=conditioning_strength,
            height=height,
            width=width,
            seed=seed if seed >= 0 else None,
            num_images=1,
        )

        # Run generation
        result = pipeline.generate(
            prompt=prompt,
            condition_image=condition_image,
            params=params,
        )

        # Extract outputs
        generated_image = result.images[0] if result.images else None
        condition_display = result.condition_map if result.condition_map else condition_image

        status = f"Seed: {result.seed_used}"
        if result.memory_peak_mb > 0:
            status += f" | Peak memory: {result.memory_peak_mb:.0f} MB"

        return condition_display, generated_image, status

    return generate


# ============================================================================
# Standalone Demo (for testing without full pipeline)
# ============================================================================


def _demo_generate_fn(
    condition_image: Image.Image,
    prompt: str,
    num_inference_steps: int,
    guidance_scale: float,
    conditioning_strength: float,
    seed: int,
    image_size: str,
    negative_prompt: str,
) -> Tuple[Image.Image, Image.Image, str]:
    """Demo generation function that returns a placeholder image.

    Used for testing the UI without loading the full inference pipeline.
    Generates a simple gradient image as a placeholder.

    Args:
        condition_image: Input condition map
        prompt: Text prompt (unused in demo)
        num_inference_steps: Steps parameter (unused in demo)
        guidance_scale: Guidance scale (unused in demo)
        conditioning_strength: Conditioning strength (unused in demo)
        seed: Random seed
        image_size: Output size string
        negative_prompt: Negative prompt (unused in demo)

    Returns:
        Tuple of (condition_map, placeholder_image, status)
    """
    width, height = parse_image_size(image_size)

    # Create a placeholder gradient image
    rng = np.random.RandomState(seed if seed >= 0 else None)
    placeholder = rng.randint(0, 255, (height, width, 3), dtype=np.uint8)
    generated = Image.fromarray(placeholder)

    # Resize condition image to match output size
    condition_display = condition_image.resize((width, height), Image.BILINEAR)

    status = (
        f"Demo mode | Steps: {num_inference_steps}, "
        f"CFG: {guidance_scale}, Strength: {conditioning_strength}"
    )

    return condition_display, generated, status


def launch_controls_demo(share: bool = False, server_port: int = 7861) -> None:
    """Launch a standalone demo of the generation controls interface.

    This is useful for testing the UI layout and interactions without
    loading the full ControlNet inference pipeline.

    Args:
        share: Whether to create a public Gradio share link
        server_port: Port number for the local server
    """
    interface = build_controls_interface(generate_fn=_demo_generate_fn)
    interface.launch(share=share, server_port=server_port)


if __name__ == "__main__":
    launch_controls_demo()
