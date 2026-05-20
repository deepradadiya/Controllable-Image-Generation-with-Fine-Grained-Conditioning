"""
Unit tests for the multi-condition comparison feature in gradio_app.py.

Tests the _generate_all_conditions() function including:
- Input validation (missing image, missing prompt)
- Partial failure handling (one or more condition types fail)
- Successful generation across all condition types
- Status message formatting

Requirements Addressed: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.app.gradio_app import _generate_all_conditions


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_image():
    """Create a simple test RGB image."""
    return Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB")


@pytest.fixture
def mock_condition_array():
    """Create a mock condition array."""
    return np.zeros((64, 64, 3), dtype=np.float32)


@pytest.fixture
def mock_display_image():
    """Create a mock display image."""
    return Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB")


# ============================================================================
# Tests for input validation (Requirement 6.5)
# ============================================================================


class TestMultiConditionInputValidation:
    """Tests that validate inputs are checked before proceeding."""

    def test_no_image_returns_error(self):
        """Requirement 6.5: error when no image uploaded."""
        results = _generate_all_conditions(
            source_image=None,
            prompt="A test prompt",
            num_steps=30,
            guidance_scale=7.5,
            conditioning_strength=1.0,
        )
        # All image outputs should be None
        assert results[0] is None  # depth_map
        assert results[1] is None  # pose_map
        assert results[2] is None  # edge_map
        assert results[3] is None  # depth_gen
        assert results[4] is None  # pose_gen
        assert results[5] is None  # edge_gen
        # Status should indicate missing image
        assert "upload" in results[6].lower() or "image" in results[6].lower()

    def test_no_prompt_returns_error(self, sample_image):
        """Requirement 6.5: error when no prompt provided."""
        results = _generate_all_conditions(
            source_image=sample_image,
            prompt="",
            num_steps=30,
            guidance_scale=7.5,
            conditioning_strength=1.0,
        )
        assert results[0] is None
        assert results[3] is None
        assert "prompt" in results[6].lower()

    def test_whitespace_only_prompt_returns_error(self, sample_image):
        """Requirement 6.5: whitespace-only prompt treated as empty."""
        results = _generate_all_conditions(
            source_image=sample_image,
            prompt="   ",
            num_steps=30,
            guidance_scale=7.5,
            conditioning_strength=1.0,
        )
        assert results[0] is None
        assert "prompt" in results[6].lower()

    def test_none_prompt_returns_error(self, sample_image):
        """Requirement 6.5: None prompt treated as missing."""
        results = _generate_all_conditions(
            source_image=sample_image,
            prompt=None,
            num_steps=30,
            guidance_scale=7.5,
            conditioning_strength=1.0,
        )
        assert results[0] is None
        assert "prompt" in results[6].lower()


# ============================================================================
# Tests for partial failure handling (Requirement 6.6)
# ============================================================================


class TestMultiConditionPartialFailure:
    """Tests that partial failures are handled gracefully."""

    @patch("src.app.gradio_app._extract_condition_map")
    def test_one_extraction_fails_others_continue(
        self, mock_extract, sample_image, mock_condition_array, mock_display_image
    ):
        """Requirement 6.6: if one type fails, others still produce results."""

        def side_effect(image, condition_type):
            from src.app.gradio_app import _resolve_condition_type

            ctype = _resolve_condition_type(condition_type)
            if ctype == "pose":
                return None, None, "Pose extraction failed: model not found"
            return mock_condition_array, mock_display_image, f"{ctype} extracted"

        mock_extract.side_effect = side_effect

        # Mock the inference pipeline to avoid actual model loading
        mock_gen_result = MagicMock()
        mock_gen_result.images = [mock_display_image]

        with patch(
            "src.app.gradio_app.ControlNetInferencePipeline", create=True
        ) as mock_pipeline_cls, patch(
            "src.inference.pipeline.ControlNetInferencePipeline", create=True
        ), patch(
            "src.inference.pipeline.GenerationParams", create=True
        ), patch(
            "src.inference.pipeline.InferenceConfig", create=True
        ):
            # Patch the import inside the function
            import importlib
            import types

            mock_pipeline_module = types.ModuleType("src.inference.pipeline")
            mock_pipeline_module.ControlNetInferencePipeline = MagicMock(
                return_value=MagicMock(generate=MagicMock(return_value=mock_gen_result))
            )
            mock_pipeline_module.GenerationParams = MagicMock(return_value=MagicMock())
            mock_pipeline_module.InferenceConfig = MagicMock(return_value=MagicMock())

            with patch.dict("sys.modules", {"src.inference.pipeline": mock_pipeline_module}):
                results = _generate_all_conditions(
                    source_image=sample_image,
                    prompt="A test prompt",
                    num_steps=30,
                    guidance_scale=7.5,
                    conditioning_strength=1.0,
                )

        # Depth and edge should have condition maps
        assert results[0] is not None  # depth_map
        assert results[1] is None  # pose_map (failed)
        assert results[2] is not None  # edge_map

        # Status should mention the failure
        status = results[6]
        assert "Pose" in status or "pose" in status

    @patch("src.app.gradio_app._extract_condition_map")
    def test_all_extractions_fail(self, mock_extract, sample_image):
        """When all extractions fail, status reports all failures."""
        mock_extract.return_value = (None, None, "Extraction failed")

        results = _generate_all_conditions(
            source_image=sample_image,
            prompt="A test prompt",
            num_steps=30,
            guidance_scale=7.5,
            conditioning_strength=1.0,
        )

        # All outputs should be None
        for i in range(6):
            assert results[i] is None

        # Status should indicate all failed
        status = results[6]
        assert "failed" in status.lower() or "✗" in status


# ============================================================================
# Tests for successful generation (Requirements 6.2, 6.3, 6.4)
# ============================================================================


class TestMultiConditionSuccess:
    """Tests for successful multi-condition generation."""

    @patch("src.app.gradio_app._extract_condition_map")
    def test_all_conditions_succeed(
        self, mock_extract, sample_image, mock_condition_array, mock_display_image
    ):
        """Requirement 6.2, 6.3: all 3 extractors and generators run."""
        mock_extract.return_value = (
            mock_condition_array,
            mock_display_image,
            "Extracted successfully",
        )

        mock_gen_result = MagicMock()
        mock_gen_result.images = [mock_display_image]

        import types

        mock_pipeline_module = types.ModuleType("src.inference.pipeline")
        mock_pipeline_module.ControlNetInferencePipeline = MagicMock(
            return_value=MagicMock(generate=MagicMock(return_value=mock_gen_result))
        )
        mock_pipeline_module.GenerationParams = MagicMock(return_value=MagicMock())
        mock_pipeline_module.InferenceConfig = MagicMock(return_value=MagicMock())

        with patch.dict("sys.modules", {"src.inference.pipeline": mock_pipeline_module}):
            results = _generate_all_conditions(
                source_image=sample_image,
                prompt="A beautiful landscape",
                num_steps=30,
                guidance_scale=7.5,
                conditioning_strength=1.0,
            )

        # All condition maps should be present
        assert results[0] is not None  # depth_map
        assert results[1] is not None  # pose_map
        assert results[2] is not None  # edge_map

        # All generated images should be present
        assert results[3] is not None  # depth_gen
        assert results[4] is not None  # pose_gen
        assert results[5] is not None  # edge_gen

        # Status should indicate success
        status = results[6]
        assert "success" in status.lower() or "✓" in status

    @patch("src.app.gradio_app._extract_condition_map")
    def test_status_reports_per_type_results(
        self, mock_extract, sample_image, mock_condition_array, mock_display_image
    ):
        """Requirement 6.4: results displayed per condition type."""
        mock_extract.return_value = (
            mock_condition_array,
            mock_display_image,
            "Extracted",
        )

        mock_gen_result = MagicMock()
        mock_gen_result.images = [mock_display_image]

        import types

        mock_pipeline_module = types.ModuleType("src.inference.pipeline")
        mock_pipeline_module.ControlNetInferencePipeline = MagicMock(
            return_value=MagicMock(generate=MagicMock(return_value=mock_gen_result))
        )
        mock_pipeline_module.GenerationParams = MagicMock(return_value=MagicMock())
        mock_pipeline_module.InferenceConfig = MagicMock(return_value=MagicMock())

        with patch.dict("sys.modules", {"src.inference.pipeline": mock_pipeline_module}):
            results = _generate_all_conditions(
                source_image=sample_image,
                prompt="test",
                num_steps=30,
                guidance_scale=7.5,
                conditioning_strength=1.0,
            )

        status = results[6]
        # Status should mention each condition type
        assert "Depth Map" in status
        assert "Pose Skeleton" in status
        assert "Edge Map" in status
