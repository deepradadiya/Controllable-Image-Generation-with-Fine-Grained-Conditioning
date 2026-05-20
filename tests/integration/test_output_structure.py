"""
Integration Test: Verify Output File Structure

Tests that the evaluation pipeline produces correctly structured output files:
1. metrics.json contains all required fields (metadata, fid_scores, alignment_scores, visual_grids)
2. Visual grid PNG files are valid (can be opened with PIL, format is PNG)
3. FID results table prints correctly (has Model and FID Score columns, separator line)
4. Alignment results table prints correctly (has Condition Type, Mean Score, Std, Target Met columns)

Uses a mock pipeline and runs the evaluation to produce output files in a tmp directory.

Requirements Validated: 3.3, 4.1, 4.2, 1.6, 2.6
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from evaluation.compute_fid import EvaluationFIDCalculator
from evaluation.condition_alignment import EvaluationAlignmentCalculator
from evaluation.config import EvaluationConfig
from evaluation.metrics_writer import build_metrics_dict, save_metrics_json
from evaluation.visual_grid import EvaluationGridGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _create_mock_pipeline():
    """Create a mock ControlNetPipeline that returns random 512x512 images."""
    mock_pipeline = MagicMock()

    def generate_random_image(**kwargs):
        """Return a random 512x512 RGB image."""
        arr = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        return Image.fromarray(arr, "RGB")

    mock_pipeline.side_effect = generate_random_image
    mock_pipeline.return_value = generate_random_image()
    # Make the mock callable and return a new random image each time
    mock_pipeline.__call__ = MagicMock(side_effect=generate_random_image)
    return mock_pipeline


@pytest.fixture
def mock_pipeline():
    """Fixture providing a mock ControlNetPipeline."""
    return _create_mock_pipeline()


@pytest.fixture
def sample_prompts() -> List[str]:
    """Fixture providing sample test prompts."""
    return [
        "A woman standing in a sunlit garden",
        "A cat sleeping on a windowsill",
        "A mountain landscape with a lake",
        "A modern kitchen with appliances",
        "A person riding a bicycle",
    ]


@pytest.fixture
def sample_original_images() -> List[Image.Image]:
    """Fixture providing sample original images."""
    images = []
    for _ in range(5):
        arr = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        images.append(Image.fromarray(arr, "RGB"))
    return images


@pytest.fixture
def sample_condition_maps() -> Dict[str, List[Image.Image]]:
    """Fixture providing sample condition maps for all 3 types."""
    condition_maps = {}
    for condition_type in ["depth", "pose", "edge"]:
        maps = []
        for _ in range(5):
            arr = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            maps.append(Image.fromarray(arr, "RGB"))
        condition_maps[condition_type] = maps
    return condition_maps


@pytest.fixture
def sample_fid_results() -> Dict[str, float]:
    """Fixture providing sample FID results."""
    return {
        "baseline": 45.2,
        "depth": 17.3,
        "pose": 16.8,
        "edge": 17.9,
    }


@pytest.fixture
def sample_alignment_results() -> Dict[str, Tuple[float, float]]:
    """Fixture providing sample alignment results."""
    return {
        "depth": (0.74, 0.08),
        "pose": (0.72, 0.11),
        "edge": (0.76, 0.06),
    }


@pytest.fixture
def sample_config() -> EvaluationConfig:
    """Fixture providing a sample EvaluationConfig."""
    return EvaluationConfig(
        output_dir="evaluation/results",
        num_fid_samples=1000,
        num_alignment_samples=100,
        num_grid_prompts=20,
        batch_size=32,
        condition_types=["depth", "pose", "edge"],
        coco_val_dir="data/raw/coco_val2017",
        checkpoint_dir="models/trained",
        guidance_scale=7.5,
        num_inference_steps=20,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Test: metrics.json contains all required fields
# ---------------------------------------------------------------------------


class TestMetricsJsonStructure:
    """Verify metrics.json contains all required fields per Requirements 4.1, 4.2."""

    def test_metrics_json_created_in_output_dir(
        self,
        tmp_path,
        sample_fid_results,
        sample_alignment_results,
        sample_config,
    ):
        """Confirm metrics.json is created in the output directory."""
        grid_paths = [
            "evaluation/results/visual_grid_depth.png",
            "evaluation/results/visual_grid_pose.png",
            "evaluation/results/visual_grid_edge.png",
            "evaluation/results/visual_grid_combined.png",
        ]

        metrics_dict = build_metrics_dict(
            fid_results=sample_fid_results,
            alignment_results=sample_alignment_results,
            grid_paths=grid_paths,
            config=sample_config,
        )
        metrics_path = save_metrics_json(metrics_dict, str(tmp_path))

        assert Path(metrics_path).exists()
        assert Path(metrics_path).name == "metrics.json"

    def test_metrics_json_has_all_top_level_keys(
        self,
        tmp_path,
        sample_fid_results,
        sample_alignment_results,
        sample_config,
    ):
        """Confirm metrics.json contains metadata, fid_scores, alignment_scores, visual_grids."""
        grid_paths = [
            "evaluation/results/visual_grid_depth.png",
            "evaluation/results/visual_grid_pose.png",
            "evaluation/results/visual_grid_edge.png",
            "evaluation/results/visual_grid_combined.png",
        ]

        metrics_dict = build_metrics_dict(
            fid_results=sample_fid_results,
            alignment_results=sample_alignment_results,
            grid_paths=grid_paths,
            config=sample_config,
        )
        metrics_path = save_metrics_json(metrics_dict, str(tmp_path))

        with open(metrics_path, "r") as f:
            loaded = json.load(f)

        required_keys = {"metadata", "fid_scores", "alignment_scores", "visual_grids"}
        assert required_keys.issubset(set(loaded.keys()))

    def test_metadata_contains_required_fields(
        self,
        tmp_path,
        sample_fid_results,
        sample_alignment_results,
        sample_config,
    ):
        """Confirm metadata has timestamp, num_fid_samples, num_alignment_samples, coco_val_size, inference_config, checkpoint_paths."""
        grid_paths = []
        metrics_dict = build_metrics_dict(
            fid_results=sample_fid_results,
            alignment_results=sample_alignment_results,
            grid_paths=grid_paths,
            config=sample_config,
        )
        metrics_path = save_metrics_json(metrics_dict, str(tmp_path))

        with open(metrics_path, "r") as f:
            loaded = json.load(f)

        metadata = loaded["metadata"]
        required_metadata_fields = {
            "timestamp",
            "num_fid_samples",
            "num_alignment_samples",
            "coco_val_size",
            "inference_config",
            "checkpoint_paths",
        }
        assert required_metadata_fields.issubset(set(metadata.keys()))

        # Verify inference_config sub-fields
        inference_config = metadata["inference_config"]
        assert "guidance_scale" in inference_config
        assert "num_inference_steps" in inference_config
        assert "image_size" in inference_config

        # Verify checkpoint_paths has entries for each condition type
        checkpoint_paths = metadata["checkpoint_paths"]
        for ct in ["depth", "pose", "edge"]:
            assert ct in checkpoint_paths

    def test_fid_scores_contains_all_conditions(
        self,
        tmp_path,
        sample_fid_results,
        sample_alignment_results,
        sample_config,
    ):
        """Confirm fid_scores has baseline_sd15 and per-condition scores."""
        grid_paths = []
        metrics_dict = build_metrics_dict(
            fid_results=sample_fid_results,
            alignment_results=sample_alignment_results,
            grid_paths=grid_paths,
            config=sample_config,
        )
        metrics_path = save_metrics_json(metrics_dict, str(tmp_path))

        with open(metrics_path, "r") as f:
            loaded = json.load(f)

        fid_scores = loaded["fid_scores"]
        assert "baseline_sd15" in fid_scores
        assert "depth" in fid_scores
        assert "pose" in fid_scores
        assert "edge" in fid_scores

        # All values should be numeric
        for key, value in fid_scores.items():
            assert isinstance(value, (int, float))

    def test_alignment_scores_contains_required_fields(
        self,
        tmp_path,
        sample_fid_results,
        sample_alignment_results,
        sample_config,
    ):
        """Confirm alignment_scores has mean, std, num_samples, metric, target_met per condition."""
        grid_paths = []
        metrics_dict = build_metrics_dict(
            fid_results=sample_fid_results,
            alignment_results=sample_alignment_results,
            grid_paths=grid_paths,
            config=sample_config,
        )
        metrics_path = save_metrics_json(metrics_dict, str(tmp_path))

        with open(metrics_path, "r") as f:
            loaded = json.load(f)

        alignment_scores = loaded["alignment_scores"]
        for ct in ["depth", "pose", "edge"]:
            assert ct in alignment_scores
            entry = alignment_scores[ct]
            assert "mean" in entry
            assert "std" in entry
            assert "num_samples" in entry
            assert "metric" in entry
            assert "target_met" in entry
            assert isinstance(entry["mean"], (int, float))
            assert isinstance(entry["std"], (int, float))
            assert isinstance(entry["num_samples"], int)
            assert isinstance(entry["metric"], str)
            assert isinstance(entry["target_met"], bool)

    def test_visual_grids_contains_paths(
        self,
        tmp_path,
        sample_fid_results,
        sample_alignment_results,
        sample_config,
    ):
        """Confirm visual_grids section maps condition types to file paths."""
        grid_paths = [
            "evaluation/results/visual_grid_depth.png",
            "evaluation/results/visual_grid_pose.png",
            "evaluation/results/visual_grid_edge.png",
            "evaluation/results/visual_grid_combined.png",
        ]
        metrics_dict = build_metrics_dict(
            fid_results=sample_fid_results,
            alignment_results=sample_alignment_results,
            grid_paths=grid_paths,
            config=sample_config,
        )
        metrics_path = save_metrics_json(metrics_dict, str(tmp_path))

        with open(metrics_path, "r") as f:
            loaded = json.load(f)

        visual_grids = loaded["visual_grids"]
        assert "depth" in visual_grids
        assert "pose" in visual_grids
        assert "edge" in visual_grids
        assert "combined" in visual_grids

        # All values should be strings (file paths)
        for key, value in visual_grids.items():
            assert isinstance(value, str)
            assert value.endswith(".png")

    def test_metrics_json_has_2_space_indentation(
        self,
        tmp_path,
        sample_fid_results,
        sample_alignment_results,
        sample_config,
    ):
        """Confirm metrics.json is formatted with 2-space indentation."""
        grid_paths = []
        metrics_dict = build_metrics_dict(
            fid_results=sample_fid_results,
            alignment_results=sample_alignment_results,
            grid_paths=grid_paths,
            config=sample_config,
        )
        metrics_path = save_metrics_json(metrics_dict, str(tmp_path))

        with open(metrics_path, "r") as f:
            content = f.read()

        # Check that 2-space indentation is used (not 4-space or tabs)
        # The second line should start with 2 spaces (first level of indentation)
        lines = content.split("\n")
        # Find first indented line
        for line in lines:
            if line.startswith(" "):
                # Should start with exactly 2 spaces (not 4)
                assert line.startswith("  ")
                assert not line.startswith("    ") or line.startswith("    ")
                break


# ---------------------------------------------------------------------------
# Test: Visual grid PNG files are valid
# ---------------------------------------------------------------------------


class TestVisualGridPngFiles:
    """Verify visual grid PNG files are valid per Requirements 3.3."""

    def test_per_condition_grids_are_valid_png(
        self,
        tmp_path,
        mock_pipeline,
        sample_original_images,
        sample_condition_maps,
        sample_prompts,
    ):
        """Confirm visual_grid_depth.png, visual_grid_pose.png, visual_grid_edge.png are valid PNGs."""
        output_dir = str(tmp_path)

        grid_generator = EvaluationGridGenerator(
            pipeline=mock_pipeline,
            cell_size=(64, 64),  # Small cells for fast test
            output_dir=output_dir,
        )

        saved_paths = grid_generator.save_all_grids(
            original_images=sample_original_images,
            condition_maps=sample_condition_maps,
            prompts=sample_prompts,
        )

        # Verify per-condition grid files exist and are valid PNGs
        for condition_type in ["depth", "pose", "edge"]:
            grid_path = tmp_path / f"visual_grid_{condition_type}.png"
            assert grid_path.exists(), f"Grid file not found: {grid_path}"

            # Verify it can be opened with PIL and is PNG format
            img = Image.open(grid_path)
            assert img.format == "PNG", f"Expected PNG format, got {img.format}"
            assert img.size[0] > 0
            assert img.size[1] > 0
            # Verify it's a valid image by accessing pixel data
            img.load()

    def test_combined_grid_is_valid_png(
        self,
        tmp_path,
        mock_pipeline,
        sample_original_images,
        sample_condition_maps,
        sample_prompts,
    ):
        """Confirm visual_grid_combined.png is a valid PNG file."""
        output_dir = str(tmp_path)

        grid_generator = EvaluationGridGenerator(
            pipeline=mock_pipeline,
            cell_size=(64, 64),  # Small cells for fast test
            output_dir=output_dir,
        )

        saved_paths = grid_generator.save_all_grids(
            original_images=sample_original_images,
            condition_maps=sample_condition_maps,
            prompts=sample_prompts,
        )

        combined_path = tmp_path / "visual_grid_combined.png"
        assert combined_path.exists(), f"Combined grid file not found: {combined_path}"

        # Verify it can be opened with PIL and is PNG format
        img = Image.open(combined_path)
        assert img.format == "PNG", f"Expected PNG format, got {img.format}"
        assert img.size[0] > 0
        assert img.size[1] > 0
        # Verify it's a valid image by accessing pixel data
        img.load()

    def test_grid_files_have_expected_paths(
        self,
        tmp_path,
        mock_pipeline,
        sample_original_images,
        sample_condition_maps,
        sample_prompts,
    ):
        """Confirm save_all_grids returns the expected file paths."""
        output_dir = str(tmp_path)

        grid_generator = EvaluationGridGenerator(
            pipeline=mock_pipeline,
            cell_size=(64, 64),
            output_dir=output_dir,
        )

        saved_paths = grid_generator.save_all_grids(
            original_images=sample_original_images,
            condition_maps=sample_condition_maps,
            prompts=sample_prompts,
        )

        # Should have 4 paths: depth, pose, edge, combined
        assert len(saved_paths) == 4

        expected_filenames = {
            "visual_grid_depth.png",
            "visual_grid_pose.png",
            "visual_grid_edge.png",
            "visual_grid_combined.png",
        }
        actual_filenames = {Path(p).name for p in saved_paths}
        assert expected_filenames == actual_filenames


# ---------------------------------------------------------------------------
# Test: FID results table prints correctly
# ---------------------------------------------------------------------------


class TestFidResultsTablePrinting:
    """Verify FID results table prints correctly per Requirement 1.6."""

    def test_fid_table_has_model_and_fid_score_columns(
        self, capsys, mock_pipeline, sample_fid_results
    ):
        """Confirm FID table has Model and FID Score columns."""
        fid_calculator = EvaluationFIDCalculator(
            pipeline=mock_pipeline,
            coco_val_dir="/tmp/fake_coco",
            batch_size=32,
        )

        fid_calculator.print_results_table(sample_fid_results)
        captured = capsys.readouterr()

        assert "Model" in captured.out
        assert "FID Score" in captured.out

    def test_fid_table_has_separator_line(
        self, capsys, mock_pipeline, sample_fid_results
    ):
        """Confirm FID table has a separator line (dashes)."""
        fid_calculator = EvaluationFIDCalculator(
            pipeline=mock_pipeline,
            coco_val_dir="/tmp/fake_coco",
            batch_size=32,
        )

        fid_calculator.print_results_table(sample_fid_results)
        captured = capsys.readouterr()

        # Check for separator line with dashes
        lines = captured.out.strip().split("\n")
        separator_found = any("---" in line for line in lines)
        assert separator_found, f"No separator line found in output:\n{captured.out}"

    def test_fid_table_has_all_model_rows(
        self, capsys, mock_pipeline, sample_fid_results
    ):
        """Confirm FID table has rows for SD1.5 baseline and all ControlNet types."""
        fid_calculator = EvaluationFIDCalculator(
            pipeline=mock_pipeline,
            coco_val_dir="/tmp/fake_coco",
            batch_size=32,
        )

        fid_calculator.print_results_table(sample_fid_results)
        captured = capsys.readouterr()

        assert "SD1.5 Baseline" in captured.out
        assert "Depth ControlNet" in captured.out
        assert "Pose ControlNet" in captured.out
        assert "Edge ControlNet" in captured.out

    def test_fid_table_shows_scores(
        self, capsys, mock_pipeline, sample_fid_results
    ):
        """Confirm FID table displays numeric scores for each model."""
        fid_calculator = EvaluationFIDCalculator(
            pipeline=mock_pipeline,
            coco_val_dir="/tmp/fake_coco",
            batch_size=32,
        )

        fid_calculator.print_results_table(sample_fid_results)
        captured = capsys.readouterr()

        # Check that scores appear in the output
        assert "45.2" in captured.out  # baseline
        assert "17.3" in captured.out  # depth
        assert "16.8" in captured.out  # pose
        assert "17.9" in captured.out  # edge

    def test_fid_table_shows_na_for_missing_conditions(
        self, capsys, mock_pipeline
    ):
        """Confirm FID table shows N/A for missing condition types."""
        fid_calculator = EvaluationFIDCalculator(
            pipeline=mock_pipeline,
            coco_val_dir="/tmp/fake_coco",
            batch_size=32,
        )

        # Only baseline and depth computed
        partial_results = {"baseline": 45.2, "depth": 17.3}
        fid_calculator.print_results_table(partial_results)
        captured = capsys.readouterr()

        assert "N/A" in captured.out


# ---------------------------------------------------------------------------
# Test: Alignment results table prints correctly
# ---------------------------------------------------------------------------


class TestAlignmentResultsTablePrinting:
    """Verify alignment results table prints correctly per Requirement 2.6."""

    def test_alignment_table_has_required_columns(
        self, capsys, mock_pipeline, sample_alignment_results
    ):
        """Confirm alignment table has Condition Type, Mean Score, Std, Target Met columns."""
        alignment_calculator = EvaluationAlignmentCalculator(
            pipeline=mock_pipeline,
        )

        alignment_calculator.print_results_table(sample_alignment_results)
        captured = capsys.readouterr()

        assert "Condition Type" in captured.out
        assert "Mean Score" in captured.out
        assert "Std" in captured.out
        # Target column (may be labeled "Target (0.70)" or similar)
        assert "Target" in captured.out

    def test_alignment_table_has_separator_line(
        self, capsys, mock_pipeline, sample_alignment_results
    ):
        """Confirm alignment table has a separator line."""
        alignment_calculator = EvaluationAlignmentCalculator(
            pipeline=mock_pipeline,
        )

        alignment_calculator.print_results_table(sample_alignment_results)
        captured = capsys.readouterr()

        lines = captured.out.strip().split("\n")
        separator_found = any("---" in line for line in lines)
        assert separator_found, f"No separator line found in output:\n{captured.out}"

    def test_alignment_table_shows_all_condition_types(
        self, capsys, mock_pipeline, sample_alignment_results
    ):
        """Confirm alignment table shows all evaluated condition types."""
        alignment_calculator = EvaluationAlignmentCalculator(
            pipeline=mock_pipeline,
        )

        alignment_calculator.print_results_table(sample_alignment_results)
        captured = capsys.readouterr()

        assert "depth" in captured.out
        assert "pose" in captured.out
        assert "edge" in captured.out

    def test_alignment_table_shows_scores_and_std(
        self, capsys, mock_pipeline, sample_alignment_results
    ):
        """Confirm alignment table displays mean scores and standard deviations."""
        alignment_calculator = EvaluationAlignmentCalculator(
            pipeline=mock_pipeline,
        )

        alignment_calculator.print_results_table(sample_alignment_results)
        captured = capsys.readouterr()

        # Check that scores appear (formatted to 4 decimal places)
        assert "0.74" in captured.out or "0.7400" in captured.out
        assert "0.72" in captured.out or "0.7200" in captured.out
        assert "0.76" in captured.out or "0.7600" in captured.out

    def test_alignment_table_shows_target_met_indicator(
        self, capsys, mock_pipeline, sample_alignment_results
    ):
        """Confirm alignment table shows whether target threshold is met."""
        alignment_calculator = EvaluationAlignmentCalculator(
            pipeline=mock_pipeline,
        )

        alignment_calculator.print_results_table(sample_alignment_results)
        captured = capsys.readouterr()

        # All scores are >= 0.70, so should show checkmark
        assert "✓" in captured.out

    def test_alignment_table_warns_below_target(
        self, capsys, mock_pipeline
    ):
        """Confirm alignment table prints warning when score is below 0.70."""
        alignment_calculator = EvaluationAlignmentCalculator(
            pipeline=mock_pipeline,
        )

        # One condition below target
        results_with_low_score = {
            "depth": (0.74, 0.08),
            "pose": (0.55, 0.15),  # Below 0.70
            "edge": (0.76, 0.06),
        }

        alignment_calculator.print_results_table(results_with_low_score)
        captured = capsys.readouterr()

        # Should show failure indicator for pose
        assert "✗" in captured.out
        # Should print a warning about pose being below target
        assert "WARNING" in captured.out or "warning" in captured.out.lower()
        assert "pose" in captured.out
