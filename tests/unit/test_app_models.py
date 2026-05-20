"""
Unit tests for the enhanced data models module (src/app/models.py).

Tests validation logic, dataclass construction, and helper methods for
EnhancedGenerationParams, PresetExample, MultiConditionResult,
PublishConfig, and ModelCardMetadata.
"""

import sys
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.app.models import (
    CONDITION_TYPE_DISPLAY_NAMES,
    CONDITION_TYPE_FROM_DISPLAY,
    ENHANCED_GUIDANCE_SCALE_DEFAULT,
    ENHANCED_GUIDANCE_SCALE_MAX,
    ENHANCED_GUIDANCE_SCALE_MIN,
    PRESET_PROMPT_MAX_LENGTH,
    VALID_CONDITION_TYPES,
    EnhancedGenerationParams,
    ModelCardMetadata,
    MultiConditionResult,
    PresetExample,
    PublishConfig,
    validate_condition_type,
    validate_guidance_scale,
    validate_preset_prompt,
)


# ============================================================================
# Tests for validation helpers
# ============================================================================


class TestValidateGuidanceScale:
    """Tests for validate_guidance_scale function."""

    def test_valid_minimum(self):
        assert validate_guidance_scale(1.0) == 1.0

    def test_valid_maximum(self):
        assert validate_guidance_scale(15.0) == 15.0

    def test_valid_default(self):
        assert validate_guidance_scale(7.5) == 7.5

    def test_valid_mid_range(self):
        assert validate_guidance_scale(10.0) == 10.0

    def test_below_minimum_raises(self):
        with pytest.raises(ValueError, match="guidance_scale must be between"):
            validate_guidance_scale(0.5)

    def test_above_maximum_raises(self):
        with pytest.raises(ValueError, match="guidance_scale must be between"):
            validate_guidance_scale(15.5)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="guidance_scale must be between"):
            validate_guidance_scale(-1.0)


class TestValidatePresetPrompt:
    """Tests for validate_preset_prompt function."""

    def test_valid_short_prompt(self):
        prompt = "A beautiful landscape"
        assert validate_preset_prompt(prompt) == prompt

    def test_valid_max_length_prompt(self):
        prompt = "x" * 200
        assert validate_preset_prompt(prompt) == prompt

    def test_empty_prompt(self):
        assert validate_preset_prompt("") == ""

    def test_exceeds_max_length_raises(self):
        prompt = "x" * 201
        with pytest.raises(ValueError, match="at most 200 characters"):
            validate_preset_prompt(prompt)


class TestValidateConditionType:
    """Tests for validate_condition_type function."""

    def test_valid_internal_names(self):
        for ct in ["depth", "pose", "edge"]:
            assert validate_condition_type(ct) == ct

    def test_valid_display_names(self):
        for name in ["Depth Map", "Pose Skeleton", "Edge Map"]:
            assert validate_condition_type(name) == name

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="condition_type must be one of"):
            validate_condition_type("invalid")


# ============================================================================
# Tests for EnhancedGenerationParams
# ============================================================================


class TestEnhancedGenerationParams:
    """Tests for the EnhancedGenerationParams dataclass."""

    def test_basic_construction(self):
        params = EnhancedGenerationParams(
            prompt="A cat sitting on a chair",
            condition_type="depth",
        )
        assert params.prompt == "A cat sitting on a chair"
        assert params.condition_type == "depth"
        assert params.guidance_scale == 7.5
        assert params.num_inference_steps == 30
        assert params.width == 512
        assert params.height == 512

    def test_custom_guidance_scale(self):
        params = EnhancedGenerationParams(
            prompt="test",
            condition_type="pose",
            guidance_scale=12.0,
        )
        assert params.guidance_scale == 12.0

    def test_invalid_guidance_scale_raises(self):
        with pytest.raises(ValueError):
            EnhancedGenerationParams(
                prompt="test",
                condition_type="depth",
                guidance_scale=20.0,
            )

    def test_invalid_condition_type_raises(self):
        with pytest.raises(ValueError):
            EnhancedGenerationParams(
                prompt="test",
                condition_type="invalid",
            )

    def test_to_dict(self):
        params = EnhancedGenerationParams(
            prompt="A landscape",
            condition_type="edge",
            guidance_scale=5.0,
            seed=42,
        )
        d = params.to_dict()
        assert d["prompt"] == "A landscape"
        assert d["condition_type"] == "edge"
        assert d["guidance_scale"] == 5.0
        assert d["seed"] == 42
        assert d["width"] == 512
        assert d["height"] == 512


# ============================================================================
# Tests for PresetExample
# ============================================================================


class TestPresetExample:
    """Tests for the PresetExample dataclass."""

    def test_basic_construction(self):
        preset = PresetExample(
            source_image_path="examples/cat.jpg",
            condition_type="Depth Map",
            prompt="A fluffy cat",
        )
        assert preset.source_image_path == "examples/cat.jpg"
        assert preset.condition_type == "Depth Map"
        assert preset.prompt == "A fluffy cat"

    def test_internal_condition_type_property(self):
        preset = PresetExample(
            source_image_path="test.jpg",
            condition_type="Pose Skeleton",
            prompt="A person dancing",
        )
        assert preset.internal_condition_type == "pose"

    def test_prompt_too_long_raises(self):
        with pytest.raises(ValueError, match="at most 200 characters"):
            PresetExample(
                source_image_path="test.jpg",
                condition_type="Edge Map",
                prompt="x" * 201,
            )

    def test_invalid_condition_type_raises(self):
        with pytest.raises(ValueError):
            PresetExample(
                source_image_path="test.jpg",
                condition_type="Invalid Type",
                prompt="test",
            )


# ============================================================================
# Tests for MultiConditionResult
# ============================================================================


class TestMultiConditionResult:
    """Tests for the MultiConditionResult dataclass."""

    def test_empty_result(self):
        result = MultiConditionResult()
        assert result.condition_maps == {}
        assert result.generated_images == {}
        assert result.errors == {}
        assert result.status == ""
        assert result.all_succeeded is True
        assert result.has_any_result is False

    def test_successful_types(self):
        from unittest.mock import MagicMock

        mock_img = MagicMock()
        result = MultiConditionResult(
            generated_images={"depth": mock_img, "pose": mock_img},
            errors={"edge": "Extraction failed"},
        )
        assert "depth" in result.successful_types
        assert "pose" in result.successful_types
        assert "edge" not in result.successful_types

    def test_failed_types(self):
        result = MultiConditionResult(
            errors={"pose": "Model not loaded"},
        )
        assert result.failed_types == ["pose"]
        assert result.all_succeeded is False

    def test_has_any_result(self):
        from unittest.mock import MagicMock

        result = MultiConditionResult(
            generated_images={"depth": MagicMock()},
        )
        assert result.has_any_result is True


# ============================================================================
# Tests for PublishConfig
# ============================================================================


class TestPublishConfig:
    """Tests for the PublishConfig dataclass."""

    def test_default_construction(self):
        config = PublishConfig()
        assert config.hf_username == "deepradadiya"
        assert config.base_model_id == "runwayml/stable-diffusion-v1-5"
        assert config.condition_types == ["depth", "pose", "edge"]
        assert config.deploy_space is False

    def test_get_adapter_repo_id(self):
        config = PublishConfig()
        assert config.get_adapter_repo_id("depth") == "deepradadiya/controlnet-sd15-depth"
        assert config.get_adapter_repo_id("pose") == "deepradadiya/controlnet-sd15-pose"
        assert config.get_adapter_repo_id("edge") == "deepradadiya/controlnet-sd15-edge"

    def test_get_combined_repo_id(self):
        config = PublishConfig()
        assert config.get_combined_repo_id() == "deepradadiya/controlnet-sd15-multi"

    def test_get_space_repo_id(self):
        config = PublishConfig()
        assert config.get_space_repo_id() == "deepradadiya/controlnet-demo"

    def test_get_adapter_weight_path(self):
        config = PublishConfig(model_dir=Path("/tmp/models"))
        assert config.get_adapter_weight_path("depth") == Path("/tmp/models/depth/model.safetensors")

    def test_get_adapter_config_path(self):
        config = PublishConfig(model_dir=Path("/tmp/models"))
        assert config.get_adapter_config_path("pose") == Path("/tmp/models/pose/config.json")

    def test_invalid_condition_type_raises(self):
        with pytest.raises(ValueError, match="Invalid condition type"):
            PublishConfig(condition_types=["depth", "invalid"])

    def test_custom_username(self):
        config = PublishConfig(hf_username="testuser")
        assert config.get_adapter_repo_id("depth") == "testuser/controlnet-sd15-depth"

    def test_get_missing_adapters_all_missing(self, tmp_path):
        config = PublishConfig(model_dir=tmp_path)
        missing = config.get_missing_adapters()
        assert set(missing) == {"depth", "pose", "edge"}

    def test_get_missing_adapters_some_present(self, tmp_path):
        # Create depth adapter file
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()
        (depth_dir / "model.safetensors").touch()

        config = PublishConfig(model_dir=tmp_path)
        missing = config.get_missing_adapters()
        assert "depth" not in missing
        assert "pose" in missing
        assert "edge" in missing


# ============================================================================
# Tests for ModelCardMetadata
# ============================================================================


class TestModelCardMetadata:
    """Tests for the ModelCardMetadata dataclass."""

    def test_basic_construction(self):
        meta = ModelCardMetadata(
            condition_type="depth",
            repo_id="deepradadiya/controlnet-sd15-depth",
        )
        assert meta.condition_type == "depth"
        assert meta.base_model == "runwayml/stable-diffusion-v1-5"
        assert meta.license == "apache-2.0"
        assert "controlnet" in meta.tags

    def test_has_metrics_false_when_none(self):
        meta = ModelCardMetadata(
            condition_type="pose",
            repo_id="test/repo",
        )
        assert meta.has_metrics is False

    def test_has_metrics_true_with_fid(self):
        meta = ModelCardMetadata(
            condition_type="edge",
            repo_id="test/repo",
            fid_score=45.2,
        )
        assert meta.has_metrics is True

    def test_has_visual_grid(self):
        meta = ModelCardMetadata(
            condition_type="depth",
            repo_id="test/repo",
            visual_grid_path="grids/depth_grid.png",
        )
        assert meta.has_visual_grid is True

    def test_no_visual_grid(self):
        meta = ModelCardMetadata(
            condition_type="depth",
            repo_id="test/repo",
        )
        assert meta.has_visual_grid is False

    def test_alignment_metric_name_depth(self):
        meta = ModelCardMetadata(condition_type="depth", repo_id="test/repo")
        assert meta.alignment_metric_name == "Pearson correlation"

    def test_alignment_metric_name_pose(self):
        meta = ModelCardMetadata(condition_type="pose", repo_id="test/repo")
        assert meta.alignment_metric_name == "normalized keypoint distance"

    def test_alignment_metric_name_edge(self):
        meta = ModelCardMetadata(condition_type="edge", repo_id="test/repo")
        assert meta.alignment_metric_name == "SSIM"

    def test_invalid_condition_type_raises(self):
        with pytest.raises(ValueError, match="Invalid condition_type"):
            ModelCardMetadata(
                condition_type="invalid",
                repo_id="test/repo",
            )
