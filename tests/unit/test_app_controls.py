"""
Unit tests for the interactive generation controls module (src/app/controls.py).

Tests the Gradio UI components, parameter parsing, generation callback wiring,
and side-by-side display layout for the ControlNet demo application.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.app.controls import (
    CONDITIONING_STRENGTH_DEFAULT,
    CONDITIONING_STRENGTH_MAX,
    CONDITIONING_STRENGTH_MIN,
    GUIDANCE_SCALE_DEFAULT,
    GUIDANCE_SCALE_MAX,
    GUIDANCE_SCALE_MIN,
    IMAGE_SIZE_DEFAULT,
    IMAGE_SIZES,
    STEPS_DEFAULT,
    STEPS_MAX,
    STEPS_MIN,
    GenerationParameters,
    _clear_outputs,
    _demo_generate_fn,
    build_controls_interface,
    create_generation_controls,
    create_generation_trigger,
    create_pipeline_generate_fn,
    create_side_by_side_display,
    parse_image_size,
)


# ============================================================================
# Tests for GenerationParameters
# ============================================================================


class TestGenerationParameters:
    """Tests for the GenerationParameters dataclass."""

    def test_default_values(self):
        """Test default parameter values are set correctly."""
        params = GenerationParameters()
        assert params.num_inference_steps == STEPS_DEFAULT
        assert params.guidance_scale == GUIDANCE_SCALE_DEFAULT
        assert params.conditioning_strength == CONDITIONING_STRENGTH_DEFAULT
        assert params.seed == -1
        assert params.image_size == IMAGE_SIZE_DEFAULT
        assert params.negative_prompt == ""

    def test_custom_values(self):
        """Test creating parameters with custom values."""
        params = GenerationParameters(
            num_inference_steps=30,
            guidance_scale=10.0,
            conditioning_strength=1.5,
            seed=42,
            image_size="768x768",
            negative_prompt="blurry, low quality",
        )
        assert params.num_inference_steps == 30
        assert params.guidance_scale == 10.0
        assert params.conditioning_strength == 1.5
        assert params.seed == 42
        assert params.image_size == "768x768"
        assert params.negative_prompt == "blurry, low quality"

    def test_width_property(self):
        """Test width extraction from image_size string."""
        params = GenerationParameters(image_size="768x512")
        assert params.width == 768

    def test_height_property(self):
        """Test height extraction from image_size string."""
        params = GenerationParameters(image_size="768x512")
        assert params.height == 512

    def test_to_dict(self):
        """Test conversion to dictionary for pipeline integration."""
        params = GenerationParameters(
            num_inference_steps=25,
            guidance_scale=8.0,
            conditioning_strength=0.7,
            seed=123,
            image_size="256x256",
            negative_prompt="ugly",
        )
        d = params.to_dict()
        assert d["num_inference_steps"] == 25
        assert d["guidance_scale"] == 8.0
        assert d["conditioning_scale"] == 0.7
        assert d["seed"] == 123
        assert d["width"] == 256
        assert d["height"] == 256
        assert d["negative_prompt"] == "ugly"

    def test_to_dict_random_seed(self):
        """Test that seed=-1 maps to None in dict (random)."""
        params = GenerationParameters(seed=-1)
        d = params.to_dict()
        assert d["seed"] is None

    def test_to_dict_valid_seed(self):
        """Test that positive seed is preserved in dict."""
        params = GenerationParameters(seed=42)
        d = params.to_dict()
        assert d["seed"] == 42


# ============================================================================
# Tests for parse_image_size
# ============================================================================


class TestParseImageSize:
    """Tests for the parse_image_size utility function."""

    def test_valid_512x512(self):
        """Test parsing standard 512x512 size."""
        w, h = parse_image_size("512x512")
        assert w == 512
        assert h == 512

    def test_valid_256x256(self):
        """Test parsing 256x256 size."""
        w, h = parse_image_size("256x256")
        assert w == 256
        assert h == 256

    def test_valid_768x768(self):
        """Test parsing 768x768 size."""
        w, h = parse_image_size("768x768")
        assert w == 768
        assert h == 768

    def test_non_square(self):
        """Test parsing non-square dimensions."""
        w, h = parse_image_size("768x512")
        assert w == 768
        assert h == 512

    def test_invalid_format_no_x(self):
        """Test that missing 'x' separator raises ValueError."""
        with pytest.raises(ValueError):
            parse_image_size("512")

    def test_invalid_format_multiple_x(self):
        """Test that multiple 'x' separators raises ValueError."""
        with pytest.raises(ValueError):
            parse_image_size("512x512x3")

    def test_invalid_non_numeric(self):
        """Test that non-numeric values raise ValueError."""
        with pytest.raises(ValueError):
            parse_image_size("abcxdef")

    def test_invalid_zero_dimension(self):
        """Test that zero dimensions raise ValueError."""
        with pytest.raises(ValueError):
            parse_image_size("0x512")

    def test_invalid_negative_dimension(self):
        """Test that negative dimensions raise ValueError."""
        with pytest.raises(ValueError):
            parse_image_size("-1x512")


# ============================================================================
# Tests for Gradio component creation
# ============================================================================


class TestCreateGenerationControls:
    """Tests for create_generation_controls function."""

    def test_returns_dict_with_expected_keys(self):
        """Test that all expected control components are created."""
        import gradio as gr

        with gr.Blocks():
            controls = create_generation_controls()

        assert "num_inference_steps" in controls
        assert "guidance_scale" in controls
        assert "conditioning_strength" in controls
        assert "seed" in controls
        assert "image_size" in controls
        assert "negative_prompt" in controls

    def test_slider_components_are_sliders(self):
        """Test that slider controls are Gradio Slider instances."""
        import gradio as gr

        with gr.Blocks():
            controls = create_generation_controls()

        assert isinstance(controls["num_inference_steps"], gr.Slider)
        assert isinstance(controls["guidance_scale"], gr.Slider)
        assert isinstance(controls["conditioning_strength"], gr.Slider)

    def test_seed_is_number_component(self):
        """Test that seed control is a Gradio Number instance."""
        import gradio as gr

        with gr.Blocks():
            controls = create_generation_controls()

        assert isinstance(controls["seed"], gr.Number)

    def test_image_size_is_dropdown(self):
        """Test that image size control is a Gradio Dropdown instance."""
        import gradio as gr

        with gr.Blocks():
            controls = create_generation_controls()

        assert isinstance(controls["image_size"], gr.Dropdown)

    def test_negative_prompt_is_textbox(self):
        """Test that negative prompt control is a Gradio Textbox instance."""
        import gradio as gr

        with gr.Blocks():
            controls = create_generation_controls()

        assert isinstance(controls["negative_prompt"], gr.Textbox)


class TestCreateSideBySideDisplay:
    """Tests for create_side_by_side_display function."""

    def test_returns_dict_with_display_keys(self):
        """Test that display components are created with expected keys."""
        import gradio as gr

        with gr.Blocks():
            display = create_side_by_side_display()

        assert "condition_display" in display
        assert "generated_display" in display

    def test_display_components_are_images(self):
        """Test that display components are Gradio Image instances."""
        import gradio as gr

        with gr.Blocks():
            display = create_side_by_side_display()

        assert isinstance(display["condition_display"], gr.Image)
        assert isinstance(display["generated_display"], gr.Image)


class TestCreateGenerationTrigger:
    """Tests for create_generation_trigger function."""

    def test_returns_dict_with_trigger_keys(self):
        """Test that trigger components are created with expected keys."""
        import gradio as gr

        with gr.Blocks():
            triggers = create_generation_trigger()

        assert "generate_btn" in triggers
        assert "clear_btn" in triggers
        assert "status" in triggers

    def test_generate_button_is_primary(self):
        """Test that generate button has primary variant."""
        import gradio as gr

        with gr.Blocks():
            triggers = create_generation_trigger()

        assert isinstance(triggers["generate_btn"], gr.Button)

    def test_status_is_textbox(self):
        """Test that status indicator is a Textbox."""
        import gradio as gr

        with gr.Blocks():
            triggers = create_generation_trigger()

        assert isinstance(triggers["status"], gr.Textbox)


# ============================================================================
# Tests for clear outputs
# ============================================================================


class TestClearOutputs:
    """Tests for the _clear_outputs helper function."""

    def test_returns_correct_tuple(self):
        """Test that clear returns None images, Ready status, and empty dict."""
        result = _clear_outputs()
        assert result == (None, None, "Ready", {})

    def test_returns_four_elements(self):
        """Test that clear returns exactly 4 elements."""
        result = _clear_outputs()
        assert len(result) == 4


# ============================================================================
# Tests for demo generate function
# ============================================================================


class TestDemoGenerateFn:
    """Tests for the _demo_generate_fn placeholder function."""

    def test_returns_three_elements(self):
        """Test that demo function returns (condition, generated, status)."""
        condition = Image.new("RGB", (64, 64), color="red")
        result = _demo_generate_fn(
            condition_image=condition,
            prompt="test prompt",
            num_inference_steps=20,
            guidance_scale=7.5,
            conditioning_strength=1.0,
            seed=42,
            image_size="256x256",
            negative_prompt="",
        )
        assert len(result) == 3

    def test_returns_pil_images(self):
        """Test that demo function returns PIL Image objects."""
        condition = Image.new("RGB", (64, 64), color="blue")
        cond_display, generated, status = _demo_generate_fn(
            condition_image=condition,
            prompt="test",
            num_inference_steps=20,
            guidance_scale=7.5,
            conditioning_strength=1.0,
            seed=42,
            image_size="512x512",
            negative_prompt="",
        )
        assert isinstance(cond_display, Image.Image)
        assert isinstance(generated, Image.Image)

    def test_output_matches_requested_size(self):
        """Test that generated image matches requested dimensions."""
        condition = Image.new("RGB", (64, 64), color="green")
        cond_display, generated, _ = _demo_generate_fn(
            condition_image=condition,
            prompt="test",
            num_inference_steps=20,
            guidance_scale=7.5,
            conditioning_strength=1.0,
            seed=42,
            image_size="256x256",
            negative_prompt="",
        )
        assert generated.size == (256, 256)
        assert cond_display.size == (256, 256)

    def test_reproducibility_with_same_seed(self):
        """Test that same seed produces same output."""
        condition = Image.new("RGB", (64, 64), color="white")
        _, gen1, _ = _demo_generate_fn(
            condition_image=condition,
            prompt="test",
            num_inference_steps=20,
            guidance_scale=7.5,
            conditioning_strength=1.0,
            seed=123,
            image_size="256x256",
            negative_prompt="",
        )
        _, gen2, _ = _demo_generate_fn(
            condition_image=condition,
            prompt="test",
            num_inference_steps=20,
            guidance_scale=7.5,
            conditioning_strength=1.0,
            seed=123,
            image_size="256x256",
            negative_prompt="",
        )
        assert np.array_equal(np.array(gen1), np.array(gen2))

    def test_different_seeds_produce_different_output(self):
        """Test that different seeds produce different outputs."""
        condition = Image.new("RGB", (64, 64), color="white")
        _, gen1, _ = _demo_generate_fn(
            condition_image=condition,
            prompt="test",
            num_inference_steps=20,
            guidance_scale=7.5,
            conditioning_strength=1.0,
            seed=1,
            image_size="256x256",
            negative_prompt="",
        )
        _, gen2, _ = _demo_generate_fn(
            condition_image=condition,
            prompt="test",
            num_inference_steps=20,
            guidance_scale=7.5,
            conditioning_strength=1.0,
            seed=2,
            image_size="256x256",
            negative_prompt="",
        )
        assert not np.array_equal(np.array(gen1), np.array(gen2))

    def test_status_contains_parameters(self):
        """Test that status message includes parameter info."""
        condition = Image.new("RGB", (64, 64), color="white")
        _, _, status = _demo_generate_fn(
            condition_image=condition,
            prompt="test",
            num_inference_steps=30,
            guidance_scale=9.0,
            conditioning_strength=0.8,
            seed=42,
            image_size="512x512",
            negative_prompt="",
        )
        assert "30" in status
        assert "9.0" in status
        assert "0.8" in status


# ============================================================================
# Tests for build_controls_interface
# ============================================================================


class TestBuildControlsInterface:
    """Tests for the build_controls_interface function."""

    def test_returns_gradio_blocks(self):
        """Test that function returns a Gradio Blocks instance."""
        import gradio as gr

        interface = build_controls_interface()
        assert isinstance(interface, gr.Blocks)

    def test_with_generate_fn(self):
        """Test that interface can be built with a generate callback."""
        import gradio as gr

        def mock_generate(*args):
            return None, None, "done"

        interface = build_controls_interface(generate_fn=mock_generate)
        assert isinstance(interface, gr.Blocks)

    def test_without_generate_fn(self):
        """Test that interface can be built without a generate callback."""
        import gradio as gr

        interface = build_controls_interface(generate_fn=None)
        assert isinstance(interface, gr.Blocks)


# ============================================================================
# Tests for create_pipeline_generate_fn
# ============================================================================


class TestCreatePipelineGenerateFn:
    """Tests for the create_pipeline_generate_fn factory."""

    def test_returns_callable(self):
        """Test that factory returns a callable function."""
        mock_pipeline = MagicMock()
        fn = create_pipeline_generate_fn(mock_pipeline)
        assert callable(fn)

    def test_calls_pipeline_generate(self):
        """Test that the returned function calls pipeline.generate."""
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [Image.new("RGB", (512, 512))]
        mock_result.condition_map = Image.new("RGB", (512, 512))
        mock_result.seed_used = 42
        mock_result.memory_peak_mb = 1024.0
        mock_pipeline.generate.return_value = mock_result

        fn = create_pipeline_generate_fn(mock_pipeline)

        condition = Image.new("RGB", (512, 512))
        cond_display, generated, status = fn(
            condition_image=condition,
            prompt="a cat",
            num_inference_steps=20,
            guidance_scale=7.5,
            conditioning_strength=1.0,
            seed=42,
            image_size="512x512",
            negative_prompt="ugly",
        )

        mock_pipeline.generate.assert_called_once()
        assert isinstance(generated, Image.Image)
        assert "42" in status

    def test_handles_random_seed(self):
        """Test that seed=-1 passes None to pipeline."""
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.images = [Image.new("RGB", (512, 512))]
        mock_result.condition_map = None
        mock_result.seed_used = 99999
        mock_result.memory_peak_mb = 0.0
        mock_pipeline.generate.return_value = mock_result

        fn = create_pipeline_generate_fn(mock_pipeline)

        condition = Image.new("RGB", (512, 512))
        fn(
            condition_image=condition,
            prompt="test",
            num_inference_steps=20,
            guidance_scale=7.5,
            conditioning_strength=1.0,
            seed=-1,
            image_size="512x512",
            negative_prompt="",
        )

        # Verify the params passed to generate have seed=None
        call_kwargs = mock_pipeline.generate.call_args[1]
        assert call_kwargs["params"].seed is None


# ============================================================================
# Tests for constants validation
# ============================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_steps_range_valid(self):
        """Test inference steps range is valid."""
        assert STEPS_MIN > 0
        assert STEPS_MAX > STEPS_MIN
        assert STEPS_MIN <= STEPS_DEFAULT <= STEPS_MAX

    def test_guidance_scale_range_valid(self):
        """Test guidance scale range is valid."""
        assert GUIDANCE_SCALE_MIN > 0
        assert GUIDANCE_SCALE_MAX > GUIDANCE_SCALE_MIN
        assert GUIDANCE_SCALE_MIN <= GUIDANCE_SCALE_DEFAULT <= GUIDANCE_SCALE_MAX

    def test_conditioning_strength_range_valid(self):
        """Test conditioning strength range is valid."""
        assert CONDITIONING_STRENGTH_MIN >= 0.0
        assert CONDITIONING_STRENGTH_MAX > CONDITIONING_STRENGTH_MIN
        assert CONDITIONING_STRENGTH_MIN <= CONDITIONING_STRENGTH_DEFAULT <= CONDITIONING_STRENGTH_MAX

    def test_image_sizes_valid(self):
        """Test all image sizes can be parsed."""
        for size in IMAGE_SIZES:
            w, h = parse_image_size(size)
            assert w > 0
            assert h > 0

    def test_default_image_size_in_choices(self):
        """Test default image size is in the choices list."""
        assert IMAGE_SIZE_DEFAULT in IMAGE_SIZES
