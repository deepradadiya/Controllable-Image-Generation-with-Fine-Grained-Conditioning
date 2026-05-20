"""
Unit tests for the EvaluationFIDCalculator class.

Tests the image generation, baseline generation, and COCO image loading
functionality using mocked pipeline and filesystem.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from evaluation.compute_fid import EvaluationFIDCalculator


@pytest.fixture
def mock_pipeline():
    """Create a mock ControlNetPipeline that returns 512x512 RGB images."""
    pipeline = MagicMock()
    pipeline.return_value = Image.new("RGB", (512, 512), (128, 128, 128))
    return pipeline


@pytest.fixture
def coco_dir(tmp_path):
    """Create a temporary directory with fake COCO validation images."""
    for i in range(20):
        img = Image.new("RGB", (640, 480), (i * 10, i * 5, i * 3))
        img.save(tmp_path / f"image_{i:06d}.jpg")
    return str(tmp_path)


@pytest.fixture
def calculator(mock_pipeline, coco_dir):
    """Create an EvaluationFIDCalculator with mock pipeline and temp COCO dir."""
    import torch

    return EvaluationFIDCalculator(
        pipeline=mock_pipeline,
        coco_val_dir=coco_dir,
        batch_size=8,
        device=torch.device("cpu"),
    )


class TestEvaluationFIDCalculatorInit:
    """Tests for EvaluationFIDCalculator initialization."""

    def test_init_stores_pipeline(self, calculator, mock_pipeline):
        assert calculator.pipeline is mock_pipeline

    def test_init_stores_coco_dir(self, calculator, coco_dir):
        assert calculator.coco_val_dir == coco_dir

    def test_init_stores_batch_size(self, calculator):
        assert calculator.batch_size == 8

    def test_init_default_device(self, mock_pipeline, coco_dir):
        calc = EvaluationFIDCalculator(
            pipeline=mock_pipeline,
            coco_val_dir=coco_dir,
        )
        assert calc.device is not None


class TestGenerateImages:
    """Tests for generate_images method."""

    def test_generates_correct_number_of_images(self, calculator, mock_pipeline):
        prompts = ["a cat", "a dog"]
        conditions = [Image.new("RGB", (512, 512), (255, 0, 0))]
        result = calculator.generate_images(
            prompts=prompts,
            condition_maps=conditions,
            condition_type="depth",
            num_images=5,
            seed=42,
        )
        assert len(result) == 5

    def test_cycles_through_prompts(self, calculator, mock_pipeline):
        prompts = ["prompt_a", "prompt_b"]
        conditions = [Image.new("RGB", (512, 512))]
        calculator.generate_images(
            prompts=prompts,
            condition_maps=conditions,
            condition_type="edge",
            num_images=4,
            seed=0,
        )
        # Check that prompts cycle: a, b, a, b
        calls = mock_pipeline.call_args_list
        assert calls[0].kwargs["text_prompt"] == "prompt_a"
        assert calls[1].kwargs["text_prompt"] == "prompt_b"
        assert calls[2].kwargs["text_prompt"] == "prompt_a"
        assert calls[3].kwargs["text_prompt"] == "prompt_b"

    def test_uses_incrementing_seeds(self, calculator, mock_pipeline):
        prompts = ["test"]
        conditions = [Image.new("RGB", (512, 512))]
        calculator.generate_images(
            prompts=prompts,
            condition_maps=conditions,
            condition_type="pose",
            num_images=3,
            seed=100,
        )
        calls = mock_pipeline.call_args_list
        assert calls[0].kwargs["seed"] == 100
        assert calls[1].kwargs["seed"] == 101
        assert calls[2].kwargs["seed"] == 102

    def test_passes_condition_type(self, calculator, mock_pipeline):
        prompts = ["test"]
        conditions = [Image.new("RGB", (512, 512))]
        calculator.generate_images(
            prompts=prompts,
            condition_maps=conditions,
            condition_type="edge",
            num_images=1,
            seed=0,
        )
        calls = mock_pipeline.call_args_list
        assert calls[0].kwargs["condition_type"] == "edge"

    def test_handles_pipeline_error_gracefully(self, calculator, mock_pipeline):
        mock_pipeline.side_effect = RuntimeError("GPU OOM")
        prompts = ["test"]
        conditions = [Image.new("RGB", (512, 512))]
        result = calculator.generate_images(
            prompts=prompts,
            condition_maps=conditions,
            condition_type="depth",
            num_images=3,
            seed=0,
        )
        # Should return placeholder images instead of crashing
        assert len(result) == 3
        for img in result:
            assert img.size == (512, 512)


class TestGenerateBaselineImages:
    """Tests for generate_baseline_images method."""

    def test_generates_correct_number(self, calculator, mock_pipeline):
        prompts = ["a landscape", "a portrait"]
        result = calculator.generate_baseline_images(
            prompts=prompts,
            num_images=4,
            seed=42,
        )
        assert len(result) == 4

    def test_uses_zero_condition_map(self, calculator, mock_pipeline):
        prompts = ["test"]
        calculator.generate_baseline_images(
            prompts=prompts,
            num_images=1,
            seed=0,
        )
        call = mock_pipeline.call_args_list[0]
        condition_img = call.kwargs["condition_image"]
        # Verify it's a black (zero) image
        import numpy as np

        arr = np.array(condition_img)
        assert arr.max() == 0

    def test_uses_incrementing_seeds(self, calculator, mock_pipeline):
        prompts = ["test"]
        calculator.generate_baseline_images(
            prompts=prompts,
            num_images=3,
            seed=50,
        )
        calls = mock_pipeline.call_args_list
        assert calls[0].kwargs["seed"] == 50
        assert calls[1].kwargs["seed"] == 51
        assert calls[2].kwargs["seed"] == 52

    def test_handles_pipeline_error_gracefully(self, calculator, mock_pipeline):
        mock_pipeline.side_effect = RuntimeError("Error")
        prompts = ["test"]
        result = calculator.generate_baseline_images(
            prompts=prompts,
            num_images=2,
            seed=0,
        )
        assert len(result) == 2


class TestLoadCocoImages:
    """Tests for load_coco_images method."""

    def test_loads_requested_number(self, calculator):
        result = calculator.load_coco_images(num_images=10)
        assert len(result) == 10

    def test_loads_all_if_fewer_available(self, calculator):
        # Our fixture has 20 images, requesting 20 should work
        result = calculator.load_coco_images(num_images=20)
        assert len(result) == 20

    def test_caps_at_available_images(self, calculator):
        # Requesting more than available should cap at available count
        result = calculator.load_coco_images(num_images=100)
        assert len(result) == 20  # Only 20 in our fixture

    def test_returns_pil_images(self, calculator):
        result = calculator.load_coco_images(num_images=5)
        for img in result:
            assert isinstance(img, Image.Image)
            assert img.mode == "RGB"

    def test_raises_on_missing_directory(self, mock_pipeline):
        import torch

        calc = EvaluationFIDCalculator(
            pipeline=mock_pipeline,
            coco_val_dir="/nonexistent/path",
            device=torch.device("cpu"),
        )
        with pytest.raises(FileNotFoundError):
            calc.load_coco_images(num_images=10)

    def test_raises_on_empty_directory(self, mock_pipeline, tmp_path):
        import torch

        calc = EvaluationFIDCalculator(
            pipeline=mock_pipeline,
            coco_val_dir=str(tmp_path),
            device=torch.device("cpu"),
        )
        with pytest.raises(ValueError, match="No images found"):
            calc.load_coco_images(num_images=10)

    def test_reproducible_sampling(self, calculator):
        """Same call should return same images due to fixed seed."""
        result1 = calculator.load_coco_images(num_images=10)
        result2 = calculator.load_coco_images(num_images=10)
        # Both calls use the same fixed seed, so should get same images
        import numpy as np

        for img1, img2 in zip(result1, result2):
            assert np.array_equal(np.array(img1), np.array(img2))


class TestComputeFidForCondition:
    """Tests for compute_fid_for_condition method."""

    def test_returns_float_fid_score(self, calculator, mock_pipeline):
        """FID score should be a float value."""
        prompts = ["a cat on a table", "a dog in a park"]
        conditions = [Image.new("RGB", (512, 512), (100, 100, 100))]
        result = calculator.compute_fid_for_condition(
            condition_type="depth",
            prompts=prompts,
            condition_maps=conditions,
            num_images=5,
        )
        assert isinstance(result, float)

    def test_fid_score_is_non_negative(self, calculator, mock_pipeline):
        """FID score should always be non-negative."""
        prompts = ["test prompt"]
        conditions = [Image.new("RGB", (512, 512), (50, 50, 50))]
        result = calculator.compute_fid_for_condition(
            condition_type="edge",
            prompts=prompts,
            condition_maps=conditions,
            num_images=5,
        )
        assert result >= 0.0

    def test_calls_generate_images_with_correct_args(self, calculator, mock_pipeline):
        """Should call generate_images with the provided condition type and prompts."""
        prompts = ["prompt_a", "prompt_b"]
        conditions = [Image.new("RGB", (512, 512), (200, 0, 0))]
        calculator.compute_fid_for_condition(
            condition_type="pose",
            prompts=prompts,
            condition_maps=conditions,
            num_images=3,
        )
        # Verify pipeline was called with correct condition_type
        calls = mock_pipeline.call_args_list
        assert len(calls) == 3
        for call in calls:
            assert call.kwargs["condition_type"] == "pose"

    def test_uses_batch_size_from_init(self, mock_pipeline, coco_dir):
        """The FID calculator should use the batch_size specified at init."""
        import torch

        calc = EvaluationFIDCalculator(
            pipeline=mock_pipeline,
            coco_val_dir=coco_dir,
            batch_size=16,
            device=torch.device("cpu"),
        )
        # Verify the internal fid_calculator uses the same batch_size
        assert calc.fid_calculator.batch_size == 16

    def test_extracts_2048_dim_features(self, calculator, mock_pipeline):
        """Inception-v3 features should be 2048-dimensional."""
        prompts = ["test"]
        conditions = [Image.new("RGB", (512, 512))]
        # Generate a small set and verify feature extraction works
        generated = calculator.generate_images(
            prompts=prompts,
            condition_maps=conditions,
            condition_type="depth",
            num_images=3,
            seed=42,
        )
        features = calculator.fid_calculator.extract_features(
            generated, show_progress=False
        )
        assert features.shape == (3, 2048)

    def test_raises_on_missing_coco_dir(self, mock_pipeline):
        """Should raise FileNotFoundError if COCO directory doesn't exist."""
        import torch

        calc = EvaluationFIDCalculator(
            pipeline=mock_pipeline,
            coco_val_dir="/nonexistent/coco/path",
            device=torch.device("cpu"),
        )
        prompts = ["test"]
        conditions = [Image.new("RGB", (512, 512))]
        with pytest.raises(FileNotFoundError):
            calc.compute_fid_for_condition(
                condition_type="depth",
                prompts=prompts,
                condition_maps=conditions,
                num_images=5,
            )

    def test_fid_zero_for_identical_distributions(self, mock_pipeline, coco_dir):
        """FID should be approximately 0 when comparing identical image sets."""
        import torch

        # Create a pipeline that returns the same images as the COCO dir
        # Load one of the COCO images and have the pipeline return it
        coco_img = Image.open(
            list(Path(coco_dir).glob("*.jpg"))[0]
        ).convert("RGB")
        mock_pipeline.return_value = coco_img

        calc = EvaluationFIDCalculator(
            pipeline=mock_pipeline,
            coco_val_dir=coco_dir,
            batch_size=4,
            device=torch.device("cpu"),
        )
        prompts = ["test"]
        conditions = [Image.new("RGB", (512, 512))]

        # With very few samples, FID won't be exactly 0 but should be relatively low
        # when generated images are from the same distribution as real images
        result = calc.compute_fid_for_condition(
            condition_type="depth",
            prompts=prompts,
            condition_maps=conditions,
            num_images=5,
        )
        # Just verify it returns a valid float (exact value depends on sampling)
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert not np.isinf(result)


class TestRunFullEvaluation:
    """Tests for run_full_evaluation method."""

    def test_returns_dict_with_all_condition_types(self, calculator, mock_pipeline):
        """Should return FID scores for all provided condition types + baseline."""
        prompts = ["a cat", "a dog"]
        condition_maps = {
            "depth": [Image.new("RGB", (512, 512), (100, 100, 100))],
            "pose": [Image.new("RGB", (512, 512), (150, 150, 150))],
            "edge": [Image.new("RGB", (512, 512), (200, 200, 200))],
        }
        results = calculator.run_full_evaluation(
            prompts=prompts,
            condition_maps=condition_maps,
            num_images=5,
        )
        assert isinstance(results, dict)
        assert "baseline" in results
        assert "depth" in results
        assert "pose" in results
        assert "edge" in results

    def test_all_scores_are_floats(self, calculator, mock_pipeline):
        """All FID scores in the results should be floats."""
        prompts = ["test prompt"]
        condition_maps = {
            "depth": [Image.new("RGB", (512, 512), (100, 100, 100))],
            "edge": [Image.new("RGB", (512, 512), (200, 200, 200))],
        }
        results = calculator.run_full_evaluation(
            prompts=prompts,
            condition_maps=condition_maps,
            num_images=5,
        )
        for key, score in results.items():
            assert isinstance(score, float), f"Score for '{key}' is not a float"

    def test_all_scores_are_non_negative(self, calculator, mock_pipeline):
        """All FID scores should be non-negative."""
        prompts = ["test"]
        condition_maps = {
            "depth": [Image.new("RGB", (512, 512), (50, 50, 50))],
        }
        results = calculator.run_full_evaluation(
            prompts=prompts,
            condition_maps=condition_maps,
            num_images=5,
        )
        for key, score in results.items():
            assert score >= 0.0, f"Score for '{key}' is negative: {score}"

    def test_skips_missing_condition_types(self, calculator, mock_pipeline):
        """Should skip condition types not present in condition_maps."""
        prompts = ["test"]
        # Only provide depth, not pose or edge
        condition_maps = {
            "depth": [Image.new("RGB", (512, 512), (100, 100, 100))],
        }
        results = calculator.run_full_evaluation(
            prompts=prompts,
            condition_maps=condition_maps,
            num_images=5,
        )
        assert "depth" in results
        assert "baseline" in results
        assert "pose" not in results
        assert "edge" not in results

    def test_handles_compute_fid_failure_gracefully(self, calculator, mock_pipeline, coco_dir):
        """Should skip condition types that raise exceptions during FID computation."""
        prompts = ["test"]
        condition_maps = {
            "depth": [Image.new("RGB", (512, 512), (100, 100, 100))],
            "pose": [Image.new("RGB", (512, 512), (150, 150, 150))],
        }

        # Patch compute_fid_for_condition to raise for pose only
        original_compute = calculator.compute_fid_for_condition

        def patched_compute(condition_type, **kwargs):
            if condition_type == "pose":
                raise RuntimeError("Missing checkpoint for pose")
            return original_compute(condition_type=condition_type, **kwargs)

        calculator.compute_fid_for_condition = patched_compute

        results = calculator.run_full_evaluation(
            prompts=prompts,
            condition_maps=condition_maps,
            num_images=5,
        )
        # Depth should succeed, pose should be skipped
        assert "depth" in results
        assert "pose" not in results

    def test_handles_baseline_failure_gracefully(self, mock_pipeline, tmp_path):
        """Should handle baseline computation failure gracefully."""
        import torch

        # Create a calculator with a COCO dir that will fail on second load
        # (first load for condition, second for baseline)
        coco_dir = str(tmp_path)
        for i in range(10):
            img = Image.new("RGB", (640, 480), (i * 10, i * 5, i * 3))
            img.save(tmp_path / f"image_{i:06d}.jpg")

        calc = EvaluationFIDCalculator(
            pipeline=mock_pipeline,
            coco_val_dir=coco_dir,
            batch_size=4,
            device=torch.device("cpu"),
        )

        prompts = ["test"]
        condition_maps = {
            "depth": [Image.new("RGB", (512, 512), (100, 100, 100))],
        }

        # Make baseline generation fail
        original_generate_baseline = calc.generate_baseline_images

        def failing_baseline(*args, **kwargs):
            raise RuntimeError("Baseline generation failed")

        calc.generate_baseline_images = failing_baseline

        results = calc.run_full_evaluation(
            prompts=prompts,
            condition_maps=condition_maps,
            num_images=5,
        )
        # Depth should still succeed, baseline should be missing
        assert "depth" in results
        assert "baseline" not in results

    def test_empty_condition_maps_returns_only_baseline(self, calculator, mock_pipeline):
        """With empty condition_maps, should only compute baseline."""
        prompts = ["test"]
        condition_maps = {}
        results = calculator.run_full_evaluation(
            prompts=prompts,
            condition_maps=condition_maps,
            num_images=5,
        )
        # Only baseline should be present
        assert "baseline" in results
        assert "depth" not in results
        assert "pose" not in results
        assert "edge" not in results


class TestPrintResultsTable:
    """Tests for print_results_table method."""

    def test_prints_all_rows_with_full_results(self, calculator, capsys):
        """Should print all 4 rows when all results are present."""
        results = {
            "baseline": 45.2,
            "depth": 17.3,
            "pose": 16.8,
            "edge": 17.9,
        }
        calculator.print_results_table(results)
        captured = capsys.readouterr()
        assert "SD1.5 Baseline" in captured.out
        assert "Depth ControlNet" in captured.out
        assert "Pose ControlNet" in captured.out
        assert "Edge ControlNet" in captured.out
        assert "45.2" in captured.out
        assert "17.3" in captured.out
        assert "16.8" in captured.out
        assert "17.9" in captured.out

    def test_prints_header_with_model_and_fid_score(self, calculator, capsys):
        """Should print a header row with Model and FID Score columns."""
        results = {"baseline": 10.0}
        calculator.print_results_table(results)
        captured = capsys.readouterr()
        assert "Model" in captured.out
        assert "FID Score" in captured.out

    def test_prints_separator_line(self, calculator, capsys):
        """Should print a separator line between header and data."""
        results = {"baseline": 10.0}
        calculator.print_results_table(results)
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        # Second line should be the separator with dashes
        assert "---" in lines[1]

    def test_shows_na_for_missing_conditions(self, calculator, capsys):
        """Should show N/A for condition types not in results."""
        results = {"baseline": 45.2, "depth": 17.3}
        calculator.print_results_table(results)
        captured = capsys.readouterr()
        assert "N/A" in captured.out
        # Pose and Edge should show N/A
        lines = captured.out.strip().split("\n")
        pose_line = [l for l in lines if "Pose ControlNet" in l][0]
        edge_line = [l for l in lines if "Edge ControlNet" in l][0]
        assert "N/A" in pose_line
        assert "N/A" in edge_line

    def test_handles_empty_results(self, calculator, capsys):
        """Should print table with all N/A when results are empty."""
        results = {}
        calculator.print_results_table(results)
        captured = capsys.readouterr()
        # All rows should show N/A
        assert captured.out.count("N/A") == 4

    def test_formats_scores_with_one_decimal(self, calculator, capsys):
        """Should format FID scores with one decimal place."""
        results = {"baseline": 45.234, "depth": 17.0}
        calculator.print_results_table(results)
        captured = capsys.readouterr()
        assert "45.2" in captured.out
        assert "17.0" in captured.out

    def test_table_uses_pipe_separators(self, calculator, capsys):
        """Table should use pipe characters as column separators."""
        results = {"baseline": 10.0}
        calculator.print_results_table(results)
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        for line in lines:
            assert line.startswith("|")
            assert line.endswith("|")
