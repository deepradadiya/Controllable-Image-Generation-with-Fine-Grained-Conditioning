"""
Unit tests for evaluation/visual_grid.py

Tests the EvaluationGridGenerator class including:
- Initialization with default and custom parameters
- Grid generation with correct dimensions and layout
- Column headers and row labels rendering
- Image generation calls (with and without ControlNet)
- Text truncation for long prompts
"""

from unittest.mock import MagicMock, patch, call

import pytest
from PIL import Image, ImageDraw

from evaluation.visual_grid import EvaluationGridGenerator


@pytest.fixture
def mock_pipeline():
    """Create a mock ControlNetPipeline that returns dummy images."""
    pipeline = MagicMock()
    # Return a 512x512 RGB image for any call
    pipeline.return_value = Image.new("RGB", (512, 512), (128, 128, 128))
    return pipeline


@pytest.fixture
def generator(mock_pipeline):
    """Create an EvaluationGridGenerator with a mock pipeline."""
    return EvaluationGridGenerator(
        pipeline=mock_pipeline,
        cell_size=(256, 256),
        output_dir="evaluation/results",
    )


@pytest.fixture
def sample_data():
    """Create sample test data for grid generation."""
    num_samples = 3
    original_images = [
        Image.new("RGB", (512, 512), (255, 0, 0)) for _ in range(num_samples)
    ]
    condition_maps = [
        Image.new("RGB", (512, 512), (0, 255, 0)) for _ in range(num_samples)
    ]
    prompts = [
        "A woman standing in a sunlit garden",
        "A cat sleeping on a windowsill",
        "A mountain landscape with a lake",
    ]
    return original_images, condition_maps, prompts


class TestEvaluationGridGeneratorInit:
    """Tests for EvaluationGridGenerator initialization."""

    def test_default_cell_size(self, mock_pipeline):
        """Should use default cell_size of (256, 256)."""
        gen = EvaluationGridGenerator(pipeline=mock_pipeline)
        assert gen.cell_size == (256, 256)

    def test_custom_cell_size(self, mock_pipeline):
        """Should accept custom cell_size."""
        gen = EvaluationGridGenerator(
            pipeline=mock_pipeline, cell_size=(512, 512)
        )
        assert gen.cell_size == (512, 512)

    def test_default_output_dir(self, mock_pipeline):
        """Should use default output_dir of 'evaluation/results'."""
        gen = EvaluationGridGenerator(pipeline=mock_pipeline)
        assert str(gen.output_dir) == "evaluation/results"

    def test_custom_output_dir(self, mock_pipeline):
        """Should accept custom output_dir."""
        gen = EvaluationGridGenerator(
            pipeline=mock_pipeline, output_dir="/tmp/custom_output"
        )
        assert str(gen.output_dir) == "/tmp/custom_output"

    def test_stores_pipeline_reference(self, mock_pipeline):
        """Should store the pipeline reference."""
        gen = EvaluationGridGenerator(pipeline=mock_pipeline)
        assert gen.pipeline is mock_pipeline

    def test_column_headers_defined(self, generator):
        """Should have the correct 4 column headers."""
        assert generator.COLUMN_HEADERS == [
            "Original", "Condition Map", "With ControlNet", "Without ControlNet"
        ]

    def test_num_columns_is_four(self, generator):
        """Should have 4 columns."""
        assert generator.NUM_COLUMNS == 4


class TestGenerateGrid:
    """Tests for the generate_grid method."""

    def test_returns_pil_image(self, generator, sample_data):
        """Should return a PIL Image."""
        original_images, condition_maps, prompts = sample_data
        result = generator.generate_grid(
            condition_type="depth",
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
            num_rows=3,
            seed=42,
        )
        assert isinstance(result, Image.Image)

    def test_grid_is_rgb_mode(self, generator, sample_data):
        """Should produce an RGB image."""
        original_images, condition_maps, prompts = sample_data
        result = generator.generate_grid(
            condition_type="depth",
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
            num_rows=3,
            seed=42,
        )
        assert result.mode == "RGB"

    def test_grid_width_accounts_for_columns_and_labels(self, generator, sample_data):
        """Grid width should include row labels, 4 cells, and padding."""
        original_images, condition_maps, prompts = sample_data
        result = generator.generate_grid(
            condition_type="depth",
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
            num_rows=3,
            seed=42,
        )
        cell_w = generator.cell_size[0]
        expected_width = (
            generator.row_label_width
            + generator.padding
            + generator.NUM_COLUMNS * cell_w
            + (generator.NUM_COLUMNS + 1) * generator.padding
        )
        assert result.width == expected_width

    def test_grid_height_accounts_for_rows_and_header(self, generator, sample_data):
        """Grid height should include header, rows, and padding."""
        original_images, condition_maps, prompts = sample_data
        num_rows = 3
        result = generator.generate_grid(
            condition_type="depth",
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
            num_rows=num_rows,
            seed=42,
        )
        cell_h = generator.cell_size[1]
        expected_height = (
            generator.header_height
            + generator.padding
            + num_rows * cell_h
            + (num_rows + 1) * generator.padding
        )
        assert result.height == expected_height

    def test_calls_pipeline_with_controlnet(self, generator, sample_data, mock_pipeline):
        """Should call the pipeline for 'With ControlNet' generation."""
        original_images, condition_maps, prompts = sample_data
        generator.generate_grid(
            condition_type="depth",
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
            num_rows=3,
            seed=42,
        )
        # Pipeline should be called for both with and without ControlNet
        # 3 rows * 2 calls per row = 6 total calls
        assert mock_pipeline.call_count == 6

    def test_pipeline_called_with_correct_condition_type(
        self, generator, sample_data, mock_pipeline
    ):
        """Should pass the correct condition_type to the pipeline."""
        original_images, condition_maps, prompts = sample_data
        generator.generate_grid(
            condition_type="edge",
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
            num_rows=1,
            seed=42,
        )
        # Check that condition_type='edge' was passed in calls
        for call_args in mock_pipeline.call_args_list:
            assert call_args.kwargs["condition_type"] == "edge"

    def test_pipeline_called_with_seed(self, generator, sample_data, mock_pipeline):
        """Should pass seeds to the pipeline for reproducibility."""
        original_images, condition_maps, prompts = sample_data
        generator.generate_grid(
            condition_type="depth",
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
            num_rows=2,
            seed=100,
        )
        # First row should use seed=100, second row seed=101
        seeds_used = [c.kwargs["seed"] for c in mock_pipeline.call_args_list]
        # With ControlNet row 0: seed=100, Without ControlNet row 0: seed=100
        # With ControlNet row 1: seed=101, Without ControlNet row 1: seed=101
        assert 100 in seeds_used
        assert 101 in seeds_used

    def test_num_rows_limited_by_data(self, generator, mock_pipeline):
        """Should limit rows to available data even if num_rows is larger."""
        original_images = [Image.new("RGB", (512, 512), (255, 0, 0))]
        condition_maps = [Image.new("RGB", (512, 512), (0, 255, 0))]
        prompts = ["A single prompt"]

        result = generator.generate_grid(
            condition_type="depth",
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
            num_rows=20,  # Request 20 but only 1 available
            seed=42,
        )
        # Should only have 1 row worth of height
        cell_h = generator.cell_size[1]
        expected_height = (
            generator.header_height
            + generator.padding
            + 1 * cell_h
            + (1 + 1) * generator.padding
        )
        assert result.height == expected_height

    def test_single_row_grid(self, generator, mock_pipeline):
        """Should work correctly with a single row."""
        original_images = [Image.new("RGB", (256, 256), (100, 100, 100))]
        condition_maps = [Image.new("RGB", (256, 256), (200, 200, 200))]
        prompts = ["Test prompt"]

        result = generator.generate_grid(
            condition_type="pose",
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
            num_rows=1,
            seed=0,
        )
        assert isinstance(result, Image.Image)
        assert result.width > 0
        assert result.height > 0

    def test_custom_cell_size_affects_grid_dimensions(self, mock_pipeline):
        """Grid dimensions should scale with cell_size."""
        gen = EvaluationGridGenerator(
            pipeline=mock_pipeline, cell_size=(128, 128)
        )
        original_images = [Image.new("RGB", (512, 512), (255, 0, 0))]
        condition_maps = [Image.new("RGB", (512, 512), (0, 255, 0))]
        prompts = ["Test"]

        result = gen.generate_grid(
            condition_type="depth",
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
            num_rows=1,
            seed=42,
        )
        # Width should use 128px cells
        expected_width = (
            gen.row_label_width
            + gen.padding
            + 4 * 128
            + 5 * gen.padding
        )
        assert result.width == expected_width


class TestTextTruncation:
    """Tests for text truncation helper."""

    def test_short_text_not_truncated(self, generator):
        """Short text should not be modified."""
        font = generator._get_font(generator.label_font_size)
        result = generator._truncate_text("Short", font, 200)
        assert result == "Short"

    def test_long_text_truncated_with_ellipsis(self, generator):
        """Long text should be truncated with '...' appended."""
        font = generator._get_font(generator.label_font_size)
        long_text = "A very long prompt that definitely exceeds the maximum width allowed"
        result = generator._truncate_text(long_text, font, 50)
        assert result.endswith("...")
        assert len(result) < len(long_text)

    def test_truncated_text_fits_within_max_width(self, generator):
        """Truncated text should fit within the specified max_width."""
        font = generator._get_font(generator.label_font_size)
        long_text = "A very long prompt that should be truncated to fit"
        max_width = 80
        result = generator._truncate_text(long_text, font, max_width)

        # Measure the result
        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)
        bbox = draw.textbbox((0, 0), result, font=font)
        result_width = bbox[2] - bbox[0]
        assert result_width <= max_width


class TestGenerateWithoutControlnet:
    """Tests for the _generate_without_controlnet method."""

    def test_uses_zero_condition_map(self, generator, mock_pipeline):
        """Should pass a black (zero) condition map to nullify ControlNet."""
        generator._generate_without_controlnet(
            prompt="test prompt",
            condition_type="depth",
            seed=42,
        )
        call_kwargs = mock_pipeline.call_args.kwargs
        condition_image = call_kwargs["condition_image"]
        # Verify it's a black image (all zeros)
        import numpy as np
        arr = np.array(condition_image)
        assert arr.max() == 0

    def test_passes_correct_seed(self, generator, mock_pipeline):
        """Should pass the seed for reproducibility."""
        generator._generate_without_controlnet(
            prompt="test", condition_type="edge", seed=123
        )
        assert mock_pipeline.call_args.kwargs["seed"] == 123

    def test_passes_correct_prompt(self, generator, mock_pipeline):
        """Should pass the text prompt to the pipeline."""
        generator._generate_without_controlnet(
            prompt="a beautiful sunset", condition_type="depth", seed=0
        )
        assert mock_pipeline.call_args.kwargs["text_prompt"] == "a beautiful sunset"


class TestGenerateCombinedGrid:
    """Tests for the generate_combined_grid method."""

    @pytest.fixture
    def combined_data(self):
        """Create sample data for combined grid generation."""
        num_samples = 3
        original_images = [
            Image.new("RGB", (512, 512), (255, 0, 0)) for _ in range(num_samples)
        ]
        condition_maps = {
            "depth": [Image.new("RGB", (512, 512), (100, 100, 100)) for _ in range(num_samples)],
            "pose": [Image.new("RGB", (512, 512), (0, 255, 0)) for _ in range(num_samples)],
            "edge": [Image.new("RGB", (512, 512), (0, 0, 255)) for _ in range(num_samples)],
        }
        prompts = [
            "A woman standing in a garden",
            "A cat on a windowsill",
            "A mountain landscape",
        ]
        return original_images, condition_maps, prompts

    def test_returns_pil_image(self, generator, combined_data):
        """Should return a PIL Image."""
        original_images, condition_maps, prompts = combined_data
        result = generator.generate_combined_grid(
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
            num_rows=2,
            seed=42,
        )
        assert isinstance(result, Image.Image)

    def test_grid_is_rgb_mode(self, generator, combined_data):
        """Should produce an RGB image."""
        original_images, condition_maps, prompts = combined_data
        result = generator.generate_combined_grid(
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
            num_rows=2,
            seed=42,
        )
        assert result.mode == "RGB"

    def test_calls_pipeline_for_all_condition_types(self, generator, combined_data, mock_pipeline):
        """Should call the pipeline for each condition type."""
        original_images, condition_maps, prompts = combined_data
        generator.generate_combined_grid(
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
            num_rows=2,
            seed=42,
        )
        # 3 condition types * 2 rows * 2 calls per row (with + without) = 12
        assert mock_pipeline.call_count == 12

    def test_handles_subset_of_condition_types(self, generator, mock_pipeline):
        """Should work with only some condition types available."""
        original_images = [Image.new("RGB", (512, 512), (255, 0, 0)) for _ in range(3)]
        condition_maps = {
            "depth": [Image.new("RGB", (512, 512), (100, 100, 100)) for _ in range(3)],
        }
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]

        result = generator.generate_combined_grid(
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
            num_rows=2,
            seed=42,
        )
        assert isinstance(result, Image.Image)
        # Only 1 condition type * 2 rows * 2 calls = 4
        assert mock_pipeline.call_count == 4

    def test_handles_empty_condition_maps(self, generator, mock_pipeline):
        """Should return a placeholder when no condition types are available."""
        original_images = [Image.new("RGB", (512, 512), (255, 0, 0))]
        condition_maps = {}
        prompts = ["Test prompt"]

        result = generator.generate_combined_grid(
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
            num_rows=1,
            seed=42,
        )
        assert isinstance(result, Image.Image)
        # No pipeline calls should be made
        assert mock_pipeline.call_count == 0

    def test_num_rows_limited_by_data(self, generator, combined_data, mock_pipeline):
        """Should limit rows to available data."""
        original_images, condition_maps, prompts = combined_data
        # Request 10 rows but only 3 available
        generator.generate_combined_grid(
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
            num_rows=10,
            seed=42,
        )
        # 3 condition types * 3 rows * 2 calls = 18
        assert mock_pipeline.call_count == 18

    def test_grid_width_matches_single_grid(self, generator, combined_data):
        """Combined grid width should match per-condition grid width."""
        original_images, condition_maps, prompts = combined_data
        combined = generator.generate_combined_grid(
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
            num_rows=2,
            seed=42,
        )
        single = generator.generate_grid(
            condition_type="depth",
            original_images=original_images,
            condition_maps=condition_maps["depth"],
            prompts=prompts,
            num_rows=2,
            seed=42,
        )
        assert combined.width == single.width


class TestSaveAllGrids:
    """Tests for the save_all_grids method."""

    @pytest.fixture
    def save_data(self):
        """Create sample data for save_all_grids."""
        num_samples = 2
        original_images = [
            Image.new("RGB", (512, 512), (255, 0, 0)) for _ in range(num_samples)
        ]
        condition_maps = {
            "depth": [Image.new("RGB", (512, 512), (100, 100, 100)) for _ in range(num_samples)],
            "pose": [Image.new("RGB", (512, 512), (0, 255, 0)) for _ in range(num_samples)],
            "edge": [Image.new("RGB", (512, 512), (0, 0, 255)) for _ in range(num_samples)],
        }
        prompts = ["A test prompt one", "A test prompt two"]
        return original_images, condition_maps, prompts

    def test_returns_list_of_paths(self, mock_pipeline, save_data, tmp_path):
        """Should return a list of saved file paths."""
        original_images, condition_maps, prompts = save_data
        gen = EvaluationGridGenerator(
            pipeline=mock_pipeline,
            cell_size=(64, 64),
            output_dir=str(tmp_path),
        )
        result = gen.save_all_grids(
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
        )
        assert isinstance(result, list)
        assert len(result) == 4  # 3 per-condition + 1 combined

    def test_saves_per_condition_grids(self, mock_pipeline, save_data, tmp_path):
        """Should save a grid for each condition type."""
        original_images, condition_maps, prompts = save_data
        gen = EvaluationGridGenerator(
            pipeline=mock_pipeline,
            cell_size=(64, 64),
            output_dir=str(tmp_path),
        )
        result = gen.save_all_grids(
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
        )
        expected_files = [
            str(tmp_path / "visual_grid_depth.png"),
            str(tmp_path / "visual_grid_pose.png"),
            str(tmp_path / "visual_grid_edge.png"),
        ]
        for expected in expected_files:
            assert expected in result

    def test_saves_combined_grid(self, mock_pipeline, save_data, tmp_path):
        """Should save the combined grid."""
        original_images, condition_maps, prompts = save_data
        gen = EvaluationGridGenerator(
            pipeline=mock_pipeline,
            cell_size=(64, 64),
            output_dir=str(tmp_path),
        )
        result = gen.save_all_grids(
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
        )
        combined_path = str(tmp_path / "visual_grid_combined.png")
        assert combined_path in result

    def test_files_exist_on_disk(self, mock_pipeline, save_data, tmp_path):
        """Saved files should actually exist on disk."""
        original_images, condition_maps, prompts = save_data
        gen = EvaluationGridGenerator(
            pipeline=mock_pipeline,
            cell_size=(64, 64),
            output_dir=str(tmp_path),
        )
        result = gen.save_all_grids(
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
        )
        from pathlib import Path
        for path_str in result:
            assert Path(path_str).exists()

    def test_saved_files_are_valid_png(self, mock_pipeline, save_data, tmp_path):
        """Saved files should be valid PNG images."""
        original_images, condition_maps, prompts = save_data
        gen = EvaluationGridGenerator(
            pipeline=mock_pipeline,
            cell_size=(64, 64),
            output_dir=str(tmp_path),
        )
        result = gen.save_all_grids(
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
        )
        for path_str in result:
            img = Image.open(path_str)
            assert img.format == "PNG"

    def test_creates_output_directory(self, mock_pipeline, save_data, tmp_path):
        """Should create the output directory if it doesn't exist."""
        original_images, condition_maps, prompts = save_data
        nested_dir = tmp_path / "nested" / "output" / "dir"
        gen = EvaluationGridGenerator(
            pipeline=mock_pipeline,
            cell_size=(64, 64),
            output_dir=str(nested_dir),
        )
        result = gen.save_all_grids(
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
        )
        assert nested_dir.exists()
        assert len(result) == 4

    def test_skips_missing_condition_types(self, mock_pipeline, tmp_path):
        """Should skip condition types not in condition_maps."""
        original_images = [Image.new("RGB", (512, 512), (255, 0, 0)) for _ in range(2)]
        condition_maps = {
            "depth": [Image.new("RGB", (512, 512), (100, 100, 100)) for _ in range(2)],
        }
        prompts = ["Prompt 1", "Prompt 2"]

        gen = EvaluationGridGenerator(
            pipeline=mock_pipeline,
            cell_size=(64, 64),
            output_dir=str(tmp_path),
        )
        result = gen.save_all_grids(
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
        )
        # Should have 1 per-condition + 1 combined = 2
        assert len(result) == 2
        assert str(tmp_path / "visual_grid_depth.png") in result
        assert str(tmp_path / "visual_grid_combined.png") in result

    def test_handles_pipeline_failure_gracefully(self, tmp_path):
        """Should skip condition type if pipeline raises an exception."""
        # Create a pipeline that fails for 'pose' condition type
        failing_pipeline = MagicMock()
        call_count = [0]

        def side_effect(**kwargs):
            call_count[0] += 1
            if kwargs.get("condition_type") == "pose":
                raise RuntimeError("Pose checkpoint not found")
            return Image.new("RGB", (512, 512), (128, 128, 128))

        failing_pipeline.side_effect = side_effect

        original_images = [Image.new("RGB", (512, 512), (255, 0, 0)) for _ in range(2)]
        condition_maps = {
            "depth": [Image.new("RGB", (512, 512), (100, 100, 100)) for _ in range(2)],
            "pose": [Image.new("RGB", (512, 512), (0, 255, 0)) for _ in range(2)],
            "edge": [Image.new("RGB", (512, 512), (0, 0, 255)) for _ in range(2)],
        }
        prompts = ["Prompt 1", "Prompt 2"]

        gen = EvaluationGridGenerator(
            pipeline=failing_pipeline,
            cell_size=(64, 64),
            output_dir=str(tmp_path),
        )
        result = gen.save_all_grids(
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
        )
        # Depth and edge should succeed, pose should be skipped
        # So we get depth grid + edge grid + combined grid = 3
        assert str(tmp_path / "visual_grid_depth.png") in result
        assert str(tmp_path / "visual_grid_pose.png") not in result
        assert str(tmp_path / "visual_grid_edge.png") in result
        assert str(tmp_path / "visual_grid_combined.png") in result
