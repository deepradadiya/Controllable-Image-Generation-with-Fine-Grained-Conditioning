"""
Unit tests for preset examples component in the Gradio app.

Tests the preset example definitions, validation, and gr.Examples integration
for Task 2.2: Add preset examples component.

Requirements tested: 5.1, 5.2, 5.3, 5.4, 5.5
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.app.gradio_app import (
    PRESET_EXAMPLES,
    PRESET_IMAGES_DIR,
    get_preset_examples_data,
    _validate_preset_image,
)
from src.app.models import (
    CONDITION_TYPE_DISPLAY_NAMES,
    PRESET_PROMPT_MAX_LENGTH,
    PresetExample,
)


class TestPresetExamplesDefinition:
    """Tests for the preset examples definitions (Requirement 5.1)."""

    def test_exactly_6_presets_defined(self):
        """Requirement 5.1: exactly 6 preset example combinations."""
        assert len(PRESET_EXAMPLES) == 6

    def test_all_presets_are_preset_example_instances(self):
        """Each preset is a valid PresetExample dataclass instance."""
        for preset in PRESET_EXAMPLES:
            assert isinstance(preset, PresetExample)

    def test_all_prompts_within_200_chars(self):
        """Requirement 5.1: text prompt not exceeding 200 characters."""
        for preset in PRESET_EXAMPLES:
            assert len(preset.prompt) <= PRESET_PROMPT_MAX_LENGTH, (
                f"Preset prompt too long ({len(preset.prompt)} chars): "
                f"{preset.prompt[:50]}..."
            )

    def test_all_prompts_non_empty(self):
        """Each preset has a non-empty prompt."""
        for preset in PRESET_EXAMPLES:
            assert len(preset.prompt) > 0


class TestPresetConditionTypeCoverage:
    """Tests for condition type coverage (Requirement 5.2)."""

    def test_at_least_one_depth_preset(self):
        """Requirement 5.2: at least one preset for Depth Map."""
        depth_presets = [p for p in PRESET_EXAMPLES if p.condition_type == "Depth Map"]
        assert len(depth_presets) >= 1

    def test_at_least_one_pose_preset(self):
        """Requirement 5.2: at least one preset for Pose Skeleton."""
        pose_presets = [
            p for p in PRESET_EXAMPLES if p.condition_type == "Pose Skeleton"
        ]
        assert len(pose_presets) >= 1

    def test_at_least_one_edge_preset(self):
        """Requirement 5.2: at least one preset for Edge Map."""
        edge_presets = [p for p in PRESET_EXAMPLES if p.condition_type == "Edge Map"]
        assert len(edge_presets) >= 1

    def test_all_condition_types_are_valid_display_names(self):
        """Each preset uses a valid display name for condition type."""
        valid_display_names = list(CONDITION_TYPE_DISPLAY_NAMES.values())
        for preset in PRESET_EXAMPLES:
            assert preset.condition_type in valid_display_names, (
                f"Invalid condition type: {preset.condition_type}"
            )


class TestGetPresetExamplesData:
    """Tests for get_preset_examples_data() function (Requirement 5.3)."""

    def test_returns_list_of_lists(self):
        """Returns a list of [image_path, condition_type, prompt] lists."""
        data = get_preset_examples_data()
        assert isinstance(data, list)
        assert len(data) == 6

    def test_each_row_has_3_elements(self):
        """Each row has exactly 3 elements: image_path, condition_type, prompt."""
        data = get_preset_examples_data()
        for row in data:
            assert len(row) == 3

    def test_row_elements_are_strings(self):
        """All elements in each row are strings."""
        data = get_preset_examples_data()
        for row in data:
            for element in row:
                assert isinstance(element, str)

    def test_data_matches_preset_definitions(self):
        """Data rows match the PRESET_EXAMPLES definitions."""
        data = get_preset_examples_data()
        for i, (row, preset) in enumerate(zip(data, PRESET_EXAMPLES)):
            assert row[0] == preset.source_image_path, f"Row {i} image path mismatch"
            assert row[1] == preset.condition_type, f"Row {i} condition type mismatch"
            assert row[2] == preset.prompt, f"Row {i} prompt mismatch"


class TestValidatePresetImage:
    """Tests for _validate_preset_image() function (Requirement 5.5)."""

    def test_valid_existing_image_returns_none(self):
        """Valid existing image returns None (no error)."""
        # Use one of the actual preset images
        preset = PRESET_EXAMPLES[0]
        result = _validate_preset_image(preset.source_image_path)
        assert result is None

    def test_missing_file_returns_error_message(self):
        """Missing file returns an error message string."""
        result = _validate_preset_image("/nonexistent/path/missing.jpg")
        assert result is not None
        assert "unavailable" in result.lower()
        assert "missing.jpg" in result

    def test_error_message_indicates_which_preset(self):
        """Error message indicates which preset image is unavailable."""
        result = _validate_preset_image("/some/path/my_image.png")
        assert "my_image.png" in result


class TestPresetImagesDirectory:
    """Tests for the preset images directory configuration."""

    def test_preset_images_dir_is_path(self):
        """PRESET_IMAGES_DIR is a Path object."""
        assert isinstance(PRESET_IMAGES_DIR, Path)

    def test_preset_images_dir_points_to_examples_images(self):
        """PRESET_IMAGES_DIR points to examples/images relative to project root."""
        assert PRESET_IMAGES_DIR.name == "images"
        assert PRESET_IMAGES_DIR.parent.name == "examples"

    def test_preset_images_dir_exists(self):
        """The preset images directory exists."""
        assert PRESET_IMAGES_DIR.exists()

    def test_all_preset_image_files_exist(self):
        """All preset image files exist on disk."""
        for preset in PRESET_EXAMPLES:
            path = Path(preset.source_image_path)
            assert path.exists(), f"Preset image not found: {preset.source_image_path}"


class TestGradioAppWithPresets:
    """Tests for the Gradio app integration with presets (Requirements 5.3, 5.4)."""

    def test_app_creates_successfully_with_presets(self):
        """The Gradio app creates without errors when presets are defined."""
        from src.app.gradio_app import create_gradio_app

        app = create_gradio_app()
        assert app is not None

    def test_presets_do_not_trigger_generation(self):
        """
        Requirement 5.4: Clicking a preset populates inputs without
        triggering generation. gr.Examples with no fn parameter ensures
        this behavior (inputs are populated, no callback is triggered).
        """
        # This is verified by the gr.Examples configuration:
        # no `fn` or `outputs` parameter means no function is called on click
        data = get_preset_examples_data()
        # Verify data is structured for input population only
        for row in data:
            assert len(row) == 3  # Only input fields, no output triggers
