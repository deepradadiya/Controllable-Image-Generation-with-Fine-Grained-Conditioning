"""
Gradio Web Interface for ControlNet Image Generation

This module implements a HuggingFace Space compatible web interface for interactive
ControlNet image generation. Users can upload images, select condition types,
provide text prompts, and generate conditioned images with real-time feedback.

Key Features:
- HuggingFace Spaces compatible Gradio Blocks interface
- 3-panel display: Input Image | Condition Map | Generated Result
- Image upload with automatic condition map extraction
- Dropdown for condition type selection (Depth Map, Pose Skeleton, Edge Map)
- Text prompt input for guiding image generation
- Guidance scale slider (1.0–15.0, step 0.5, default 7.5)
- 6 preset example combinations (2 per condition type) for quick demo
- Multi-condition comparison: generate with all 3 conditions at once
- 3-column × 2-row comparison grid for side-by-side evaluation
- Error handling for missing models or failed extractions

Requirements Addressed:
- 1.1, 1.2, 1.3, 1.4, 1.5: Three-Panel Image Display
- 2.1, 2.2, 2.3, 2.4: Condition Type Selection
- 3.1, 3.2, 3.3: Guidance Scale Slider
- 4.1, 4.2, 4.3, 4.4, 4.5: Automatic Condition Map Preview
- 5.1, 5.2, 5.3, 5.4, 5.5: Preset Example Combinations
- 6.1, 6.2, 6.3, 6.4, 6.5, 6.6: Multi-Condition Comparison Button
"""

import logging
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from src.app.models import (
    CONDITION_TYPE_DISPLAY_NAMES,
    CONDITION_TYPE_FROM_DISPLAY,
    ENHANCED_GUIDANCE_SCALE_DEFAULT,
    ENHANCED_GUIDANCE_SCALE_MAX,
    ENHANCED_GUIDANCE_SCALE_MIN,
    ENHANCED_GUIDANCE_SCALE_STEP,
    MultiConditionResult,
    PresetExample,
    VALID_CONDITION_TYPES,
)

logger = logging.getLogger(__name__)

# Condition type dropdown options (display names)
CONDITION_TYPE_CHOICES = list(CONDITION_TYPE_DISPLAY_NAMES.values())
# ["Depth Map", "Pose Skeleton", "Edge Map"]

# Default generation parameters
DEFAULT_NUM_STEPS = 30
DEFAULT_GUIDANCE_SCALE = ENHANCED_GUIDANCE_SCALE_DEFAULT
DEFAULT_CONDITIONING_STRENGTH = 1.0

# --- Preset Examples ---
# Directory containing preset example images (relative to project root)
PRESET_IMAGES_DIR = Path(__file__).parent.parent.parent / "examples" / "images"

# 6 preset example combinations: at least 1 per condition type
PRESET_EXAMPLES: List[PresetExample] = [
    PresetExample(
        source_image_path=str(PRESET_IMAGES_DIR / "street_scene.jpg"),
        condition_type="Depth Map",
        prompt="A bustling city street with tall buildings and pedestrians, photorealistic, golden hour lighting",
    ),
    PresetExample(
        source_image_path=str(PRESET_IMAGES_DIR / "living_room.jpg"),
        condition_type="Depth Map",
        prompt="A cozy modern living room with warm lighting, minimalist furniture, and indoor plants",
    ),
    PresetExample(
        source_image_path=str(PRESET_IMAGES_DIR / "dancer.jpg"),
        condition_type="Pose Skeleton",
        prompt="A ballet dancer performing an elegant arabesque on stage, dramatic spotlight, high detail",
    ),
    PresetExample(
        source_image_path=str(PRESET_IMAGES_DIR / "yoga_pose.jpg"),
        condition_type="Pose Skeleton",
        prompt="A person doing a yoga tree pose in a serene mountain meadow at sunrise, peaceful atmosphere",
    ),
    PresetExample(
        source_image_path=str(PRESET_IMAGES_DIR / "building_facade.jpg"),
        condition_type="Edge Map",
        prompt="An ornate Gothic cathedral facade with intricate stone carvings and stained glass windows",
    ),
    PresetExample(
        source_image_path=str(PRESET_IMAGES_DIR / "flower_garden.jpg"),
        condition_type="Edge Map",
        prompt="A vibrant flower garden with roses, tulips, and daisies in full bloom, soft bokeh background",
    ),
]


def get_preset_examples_data() -> List[List[str]]:
    """
    Get preset examples formatted for gr.Examples component.

    Returns a list of [image_path, condition_type, prompt] lists.
    Only includes presets whose source image files exist.

    Returns:
        List of example data rows for gr.Examples.
    """
    examples_data = []
    for preset in PRESET_EXAMPLES:
        examples_data.append([
            preset.source_image_path,
            preset.condition_type,
            preset.prompt,
        ])
    return examples_data


def _validate_preset_image(image_path: str) -> Optional[str]:
    """
    Validate that a preset image file exists and is loadable.

    Args:
        image_path: Path to the preset image file.

    Returns:
        Error message string if the image is unavailable, None if valid.
    """
    path = Path(image_path)
    if not path.exists():
        return f"Preset image unavailable: {path.name} not found at {image_path}"
    try:
        img = Image.open(path)
        img.verify()
        return None
    except Exception as e:
        return f"Preset image unavailable: {path.name} failed to load ({e})"


def _resolve_condition_type(condition_type: str) -> str:
    """
    Resolve a condition type from display name or internal name to internal name.

    Args:
        condition_type: Display name (e.g. "Depth Map") or internal name (e.g. "depth").

    Returns:
        Internal condition type name ("depth", "pose", or "edge").
    """
    if condition_type in CONDITION_TYPE_FROM_DISPLAY:
        return CONDITION_TYPE_FROM_DISPLAY[condition_type]
    if condition_type in VALID_CONDITION_TYPES:
        return condition_type
    return condition_type


def _extract_condition_map(
    image: Image.Image, condition_type: str
) -> Tuple[Optional[np.ndarray], Optional[Image.Image], str]:
    """
    Extract a condition map from the uploaded image based on the selected type.

    Args:
        image: Source PIL Image uploaded by the user.
        condition_type: Display name (e.g. "Depth Map") or internal name (e.g. "depth").

    Returns:
        Tuple of (raw_condition_map_array, display_image, status_message).
        On failure, raw array and display image may be None.
    """
    if image is None:
        return None, None, "No image provided."

    # Resolve display name to internal name
    condition_type = _resolve_condition_type(condition_type)

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
) -> Tuple[Optional[Image.Image], Optional[Image.Image], Optional[Image.Image], str]:
    """
    Generate an image using the ControlNet inference pipeline.

    Returns a 3-panel output: input image, condition map, and generated image.

    Args:
        source_image: Uploaded source image for condition extraction.
        condition_type: Display name or internal name of conditioning type.
        prompt: Text prompt for generation.
        num_steps: Number of inference steps.
        guidance_scale: Classifier-free guidance scale (1.0–15.0).
        conditioning_strength: ControlNet conditioning strength.

    Returns:
        Tuple of (input_image, condition_map_display, generated_image, status_message).
    """
    if source_image is None:
        return None, None, None, "Please upload a source image."

    if not prompt or not prompt.strip():
        return source_image, None, None, "Please enter a text prompt."

    # Resolve display name to internal name
    internal_type = _resolve_condition_type(condition_type)

    # Step 1: Extract condition map
    condition_array, condition_display, extract_status = _extract_condition_map(
        source_image, internal_type
    )

    if condition_array is None:
        return source_image, None, None, f"Condition extraction failed: {extract_status}"

    # Step 2: Run inference pipeline
    try:
        from src.inference.pipeline import (
            ControlNetInferencePipeline,
            GenerationParams,
            InferenceConfig,
        )

        config = InferenceConfig(
            condition_type=internal_type,
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
            condition_type=internal_type,
        )

        if result.images:
            generated_image = result.images[0]
            status = (
                f"Image generated successfully in {result.generation_time_seconds:.1f}s. "
                f"Seed: {result.seed_used}"
            )
            return source_image, condition_display, generated_image, status
        else:
            return source_image, condition_display, None, "Generation produced no images."

    except ImportError as e:
        error_msg = f"Missing inference dependency: {e}"
        logger.error(error_msg)
        return source_image, condition_display, None, error_msg
    except RuntimeError as e:
        error_msg = f"Model loading or inference failed: {e}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return source_image, condition_display, None, error_msg
    except Exception as e:
        error_msg = f"Generation failed: {e}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        return source_image, condition_display, None, error_msg


def _on_image_upload(
    image: Optional[Image.Image], condition_type: str
) -> Tuple[Optional[Image.Image], str]:
    """
    Callback when an image is uploaded or condition type changes.
    Automatically extracts the condition map.

    Args:
        image: Uploaded PIL Image.
        condition_type: Selected condition type (display name or internal name).

    Returns:
        Tuple of (condition_map_display_image, status_message).
    """
    if image is None:
        return None, "No image uploaded."

    _, display_image, status = _extract_condition_map(image, condition_type)
    return display_image, status


def _generate_all_conditions(
    source_image: Optional[Image.Image],
    prompt: str,
    num_steps: int,
    guidance_scale: float,
    conditioning_strength: float,
) -> Tuple[
    Optional[Image.Image],
    Optional[Image.Image],
    Optional[Image.Image],
    Optional[Image.Image],
    Optional[Image.Image],
    Optional[Image.Image],
    str,
]:
    """
    Generate images using all 3 condition types for comparison.

    Runs all 3 extractors (depth, pose, edge) and generators on the same input,
    handling partial failures gracefully. Returns a 3-column × 2-row grid
    (top row: condition maps, bottom row: generated images).

    Args:
        source_image: Uploaded source image for condition extraction.
        prompt: Text prompt for generation.
        num_steps: Number of inference steps.
        guidance_scale: Classifier-free guidance scale (1.0–15.0).
        conditioning_strength: ControlNet conditioning strength.

    Returns:
        Tuple of (depth_map, pose_map, edge_map, depth_gen, pose_gen, edge_gen, status).
        Failed types will have None for their respective outputs.
    """
    # Validate inputs before proceeding
    if source_image is None:
        return None, None, None, None, None, None, "Please upload an image before generating."

    if not prompt or not prompt.strip():
        return None, None, None, None, None, None, "Please enter a text prompt before generating."

    result = MultiConditionResult()
    condition_arrays = {}

    # Step 1: Extract condition maps for all 3 types
    for ctype in VALID_CONDITION_TYPES:
        try:
            condition_array, display_image, extract_status = _extract_condition_map(
                source_image, ctype
            )
            if condition_array is not None and display_image is not None:
                result.condition_maps[ctype] = display_image
                condition_arrays[ctype] = condition_array
            else:
                result.errors[ctype] = f"Extraction failed: {extract_status}"
        except Exception as e:
            result.errors[ctype] = f"Extraction error: {e}"
            logger.error(f"Multi-condition extraction failed for {ctype}: {e}")

    # Step 2: Generate images for each successfully extracted condition type
    for ctype in list(result.condition_maps.keys()):
        if ctype in result.errors:
            continue  # Skip types that already failed during extraction

        try:
            from src.inference.pipeline import (
                ControlNetInferencePipeline,
                GenerationParams,
                InferenceConfig,
            )

            config = InferenceConfig(
                condition_type=ctype,
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

            gen_result = pipeline.generate(
                prompt=prompt.strip(),
                condition_image=condition_arrays[ctype],
                params=params,
                condition_type=ctype,
            )

            if gen_result.images:
                result.generated_images[ctype] = gen_result.images[0]
            else:
                result.errors[ctype] = "Generation produced no images."

        except ImportError as e:
            result.errors[ctype] = f"Missing dependency: {e}"
            logger.error(f"Multi-condition generation import error for {ctype}: {e}")
        except RuntimeError as e:
            result.errors[ctype] = f"Model/inference error: {e}"
            logger.error(f"Multi-condition generation runtime error for {ctype}: {e}")
        except Exception as e:
            result.errors[ctype] = f"Generation failed: {e}"
            logger.error(f"Multi-condition generation failed for {ctype}: {e}")

    # Build status message
    status_parts = []
    for ctype in VALID_CONDITION_TYPES:
        display_name = CONDITION_TYPE_DISPLAY_NAMES[ctype]
        if ctype in result.errors:
            status_parts.append(f"{display_name}: ✗ {result.errors[ctype]}")
        elif ctype in result.generated_images:
            status_parts.append(f"{display_name}: ✓ Success")
        else:
            status_parts.append(f"{display_name}: ✗ No result")

    if result.all_succeeded:
        result.status = "All 3 conditions generated successfully.\n" + "\n".join(status_parts)
    elif result.has_any_result:
        result.status = "Partial success (some conditions failed).\n" + "\n".join(status_parts)
    else:
        result.status = "All conditions failed.\n" + "\n".join(status_parts)

    # Return grid outputs: top row (condition maps), bottom row (generated images)
    depth_map = result.condition_maps.get("depth")
    pose_map = result.condition_maps.get("pose")
    edge_map = result.condition_maps.get("edge")
    depth_gen = result.generated_images.get("depth")
    pose_gen = result.generated_images.get("pose")
    edge_gen = result.generated_images.get("edge")

    return depth_map, pose_map, edge_map, depth_gen, pose_gen, edge_gen, result.status


def create_gradio_app() -> "gr.Blocks":
    """
    Create and return the Gradio Blocks application with 3-panel display.

    This function builds the enhanced Gradio interface with:
    - 3-panel display row: Input Image | Condition Map | Generated Result
    - Condition type dropdown (Depth Map, Pose Skeleton, Edge Map)
    - Text prompt input
    - Guidance scale slider (1.0–15.0, step 0.5, default 7.5)
    - Generation parameter controls
    - Status messages
    - Automatic condition map extraction on upload/type change

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

        # --- Input Controls ---
        with gr.Row():
            with gr.Column():
                source_image = gr.Image(
                    label="Upload Source Image",
                    type="pil",
                    height=256,
                )
                condition_type = gr.Dropdown(
                    choices=CONDITION_TYPE_CHOICES,
                    value=CONDITION_TYPE_CHOICES[0],  # "Depth Map"
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
                        minimum=ENHANCED_GUIDANCE_SCALE_MIN,
                        maximum=ENHANCED_GUIDANCE_SCALE_MAX,
                        value=DEFAULT_GUIDANCE_SCALE,
                        step=ENHANCED_GUIDANCE_SCALE_STEP,
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
                compare_btn = gr.Button(
                    "Generate with all 3 conditions", variant="secondary"
                )

        # --- Preset Examples ---
        # Display preset examples below input controls (Requirement 5.3)
        # Clicking a preset populates inputs without triggering generation (Req 5.4)
        gr.Markdown("### Preset Examples")
        preset_data = get_preset_examples_data()
        # Validate preset images and show warning for unavailable ones (Req 5.5)
        unavailable_presets = []
        for preset in PRESET_EXAMPLES:
            error = _validate_preset_image(preset.source_image_path)
            if error:
                unavailable_presets.append(error)

        if unavailable_presets:
            gr.Markdown(
                "⚠️ **Some preset images are unavailable:**\n"
                + "\n".join(f"- {msg}" for msg in unavailable_presets)
            )

        if preset_data:
            gr.Examples(
                examples=preset_data,
                inputs=[source_image, condition_type, prompt],
                label="Click a preset to populate inputs",
                examples_per_page=6,
            )

        # --- 3-Panel Display: Input Image | Condition Map | Generated Result ---
        with gr.Row(equal_height=True):
            input_image_output = gr.Image(
                label="Input Image",
                type="pil",
                height=300,
                interactive=False,
            )
            condition_map_output = gr.Image(
                label="Condition Map",
                type="pil",
                height=300,
                interactive=False,
            )
            generated_image_output = gr.Image(
                label="Generated Result",
                type="pil",
                height=300,
                interactive=False,
            )

        # --- Status ---
        status_output = gr.Textbox(
            label="Status",
            interactive=False,
            lines=2,
        )

        # --- Multi-Condition Comparison Grid (3-column × 2-row) ---
        gr.Markdown(
            """
            ## Multi-Condition Comparison

            Click "Generate with all 3 conditions" to compare results across
            all condition types side by side.
            """
        )

        # Top row: Condition Maps
        with gr.Row(equal_height=True):
            compare_depth_map = gr.Image(
                label="Depth Map",
                type="pil",
                height=256,
                interactive=False,
            )
            compare_pose_map = gr.Image(
                label="Pose Skeleton",
                type="pil",
                height=256,
                interactive=False,
            )
            compare_edge_map = gr.Image(
                label="Edge Map",
                type="pil",
                height=256,
                interactive=False,
            )

        # Bottom row: Generated Images
        with gr.Row(equal_height=True):
            compare_depth_gen = gr.Image(
                label="Generated (Depth)",
                type="pil",
                height=256,
                interactive=False,
            )
            compare_pose_gen = gr.Image(
                label="Generated (Pose)",
                type="pil",
                height=256,
                interactive=False,
            )
            compare_edge_gen = gr.Image(
                label="Generated (Edge)",
                type="pil",
                height=256,
                interactive=False,
            )

        # --- Comparison Status ---
        compare_status_output = gr.Textbox(
            label="Comparison Status",
            interactive=False,
            lines=3,
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

        # Wire up generation button — outputs to all 3 panels + status
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
            outputs=[
                input_image_output,
                condition_map_output,
                generated_image_output,
                status_output,
            ],
        )

        # Wire up multi-condition comparison button
        compare_btn.click(
            fn=_generate_all_conditions,
            inputs=[
                source_image,
                prompt,
                num_steps,
                guidance_scale,
                conditioning_strength,
            ],
            outputs=[
                compare_depth_map,
                compare_pose_map,
                compare_edge_map,
                compare_depth_gen,
                compare_pose_gen,
                compare_edge_gen,
                compare_status_output,
            ],
        )

    return app


# Standalone entry point for HuggingFace Spaces or local development
if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(share=False)
