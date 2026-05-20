"""
Unit tests for EvaluationAlignmentCalculator batch evaluation and reporting methods.

Tests evaluate_condition_type, run_full_evaluation, and print_results_table.
"""

import io
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from evaluation.condition_alignment import EvaluationAlignmentCalculator


def create_mock_pipeline():
    """Create a mock ControlNetPipeline that returns random 512x512 images."""
    mock_pipeline = MagicMock()

    def generate_image(**kwargs):
        # Return a random 512x512 RGB image
        seed = kwargs.get("seed", 42)
        rng = np.random.default_rng(seed)
        arr = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
        return Image.fromarray(arr, "RGB")

    mock_pipeline.side_effect = generate_image
    return mock_pipeline


def create_test_condition_maps(num_maps=5, size=(512, 512)):
    """Create test condition map images."""
    maps = []
    for i in range(num_maps):
        rng = np.random.default_rng(i)
        arr = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        maps.append(Image.fromarray(arr, "RGB"))
    return maps


class TestEvaluateConditionType:
    """Tests for evaluate_condition_type method."""

    def test_returns_tuple_of_mean_and_std(self):
        """evaluate_condition_type returns a (mean, std) tuple."""
        mock_pipeline = create_mock_pipeline()
        calculator = EvaluationAlignmentCalculator(pipeline=mock_pipeline)

        prompts = ["A test prompt"]
        condition_maps = create_test_condition_maps(2)

        result = calculator.evaluate_condition_type(
            condition_type="edge",
            prompts=prompts,
            condition_maps=condition_maps,
            num_samples=3,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        mean_score, std_score = result
        assert isinstance(mean_score, float)
        assert isinstance(std_score, float)

    def test_mean_score_in_valid_range(self):
        """Mean score should be in [0, 1] range."""
        mock_pipeline = create_mock_pipeline()
        calculator = EvaluationAlignmentCalculator(pipeline=mock_pipeline)

        prompts = ["A test prompt", "Another prompt"]
        condition_maps = create_test_condition_maps(3)

        for condition_type in ["edge", "depth", "pose"]:
            mean_score, std_score = calculator.evaluate_condition_type(
                condition_type=condition_type,
                prompts=prompts,
                condition_maps=condition_maps,
                num_samples=3,
            )
            assert 0.0 <= mean_score <= 1.0, (
                f"{condition_type} mean score {mean_score} out of range"
            )
            assert std_score >= 0.0, (
                f"{condition_type} std {std_score} is negative"
            )

    def test_invalid_condition_type_raises_error(self):
        """Invalid condition type should raise ValueError."""
        mock_pipeline = create_mock_pipeline()
        calculator = EvaluationAlignmentCalculator(pipeline=mock_pipeline)

        with pytest.raises(ValueError, match="Invalid condition_type"):
            calculator.evaluate_condition_type(
                condition_type="invalid",
                prompts=["test"],
                condition_maps=create_test_condition_maps(1),
                num_samples=1,
            )

    def test_cycles_through_prompts_and_maps(self):
        """Should cycle through prompts and condition maps when fewer than num_samples."""
        mock_pipeline = create_mock_pipeline()
        calculator = EvaluationAlignmentCalculator(pipeline=mock_pipeline)

        prompts = ["Prompt A", "Prompt B"]
        condition_maps = create_test_condition_maps(2)

        # Should not raise even with num_samples > len(prompts)
        result = calculator.evaluate_condition_type(
            condition_type="edge",
            prompts=prompts,
            condition_maps=condition_maps,
            num_samples=5,
        )
        assert result[0] >= 0.0

    def test_handles_pipeline_failure_gracefully(self):
        """Should handle pipeline failures and return results from successful samples."""
        mock_pipeline = MagicMock()
        call_count = [0]

        def generate_with_failures(**kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise RuntimeError("Pipeline failure")
            seed = kwargs.get("seed", 42)
            rng = np.random.default_rng(seed)
            arr = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
            return Image.fromarray(arr, "RGB")

        mock_pipeline.side_effect = generate_with_failures
        calculator = EvaluationAlignmentCalculator(pipeline=mock_pipeline)

        result = calculator.evaluate_condition_type(
            condition_type="edge",
            prompts=["test"],
            condition_maps=create_test_condition_maps(1),
            num_samples=4,
        )
        # Should still return valid results from successful samples
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_all_failures_returns_zero(self):
        """If all samples fail, should return (0.0, 0.0)."""
        mock_pipeline = MagicMock()
        mock_pipeline.side_effect = RuntimeError("Always fails")
        calculator = EvaluationAlignmentCalculator(pipeline=mock_pipeline)

        result = calculator.evaluate_condition_type(
            condition_type="edge",
            prompts=["test"],
            condition_maps=create_test_condition_maps(1),
            num_samples=3,
        )
        assert result == (0.0, 0.0)


class TestRunFullEvaluation:
    """Tests for run_full_evaluation method."""

    def test_evaluates_all_condition_types(self):
        """Should evaluate all condition types in condition_maps."""
        mock_pipeline = create_mock_pipeline()
        calculator = EvaluationAlignmentCalculator(pipeline=mock_pipeline)

        prompts = ["A test prompt"]
        condition_maps = {
            "edge": create_test_condition_maps(2),
            "depth": create_test_condition_maps(2),
        }

        results = calculator.run_full_evaluation(
            prompts=prompts,
            condition_maps=condition_maps,
            num_samples=2,
        )

        assert "edge" in results
        assert "depth" in results
        assert isinstance(results["edge"], tuple)
        assert isinstance(results["depth"], tuple)

    def test_handles_single_condition_type_failure(self):
        """Should skip failed condition types and continue with others."""
        mock_pipeline = MagicMock()

        call_count = [0]

        def generate_with_selective_failure(**kwargs):
            call_count[0] += 1
            condition_type = kwargs.get("condition_type", "")
            if condition_type == "pose":
                raise RuntimeError("Pose model not available")
            seed = kwargs.get("seed", 42)
            rng = np.random.default_rng(seed)
            arr = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
            return Image.fromarray(arr, "RGB")

        mock_pipeline.side_effect = generate_with_selective_failure
        calculator = EvaluationAlignmentCalculator(pipeline=mock_pipeline)

        prompts = ["test"]
        condition_maps = {
            "edge": create_test_condition_maps(2),
            "pose": create_test_condition_maps(2),
        }

        results = calculator.run_full_evaluation(
            prompts=prompts,
            condition_maps=condition_maps,
            num_samples=2,
        )

        # Edge should succeed, pose may have partial results or (0,0)
        assert "edge" in results

    def test_returns_dict_with_tuples(self):
        """Return type should be Dict[str, Tuple[float, float]]."""
        mock_pipeline = create_mock_pipeline()
        calculator = EvaluationAlignmentCalculator(pipeline=mock_pipeline)

        prompts = ["test"]
        condition_maps = {"edge": create_test_condition_maps(2)}

        results = calculator.run_full_evaluation(
            prompts=prompts,
            condition_maps=condition_maps,
            num_samples=2,
        )

        assert isinstance(results, dict)
        for key, value in results.items():
            assert isinstance(key, str)
            assert isinstance(value, tuple)
            assert len(value) == 2
            assert isinstance(value[0], float)
            assert isinstance(value[1], float)


class TestPrintResultsTable:
    """Tests for print_results_table method."""

    def test_prints_table_with_results(self, capsys):
        """Should print a formatted table with results."""
        mock_pipeline = create_mock_pipeline()
        calculator = EvaluationAlignmentCalculator(pipeline=mock_pipeline)

        results = {
            "edge": (0.76, 0.06),
            "depth": (0.74, 0.08),
            "pose": (0.72, 0.11),
        }

        calculator.print_results_table(results)
        captured = capsys.readouterr()

        assert "Condition Alignment Results" in captured.out
        assert "edge" in captured.out
        assert "depth" in captured.out
        assert "pose" in captured.out
        assert "0.7600" in captured.out
        assert "✓" in captured.out

    def test_prints_warning_for_below_target(self, capsys):
        """Should print warning when score is below 0.70."""
        mock_pipeline = create_mock_pipeline()
        calculator = EvaluationAlignmentCalculator(pipeline=mock_pipeline)

        results = {
            "edge": (0.65, 0.10),
            "depth": (0.80, 0.05),
        }

        calculator.print_results_table(results)
        captured = capsys.readouterr()

        assert "WARNING" in captured.out
        assert "edge" in captured.out
        assert "0.6500" in captured.out
        assert "✗" in captured.out
        # depth should show checkmark
        assert "✓" in captured.out

    def test_no_warning_when_all_above_target(self, capsys):
        """Should not print warning when all scores are above 0.70."""
        mock_pipeline = create_mock_pipeline()
        calculator = EvaluationAlignmentCalculator(pipeline=mock_pipeline)

        results = {
            "edge": (0.80, 0.05),
            "depth": (0.75, 0.08),
        }

        calculator.print_results_table(results)
        captured = capsys.readouterr()

        assert "WARNING" not in captured.out

    def test_handles_empty_results(self, capsys):
        """Should handle empty results dict gracefully."""
        mock_pipeline = create_mock_pipeline()
        calculator = EvaluationAlignmentCalculator(pipeline=mock_pipeline)

        results = {}
        calculator.print_results_table(results)
        captured = capsys.readouterr()

        assert "Condition Alignment Results" in captured.out
        assert "WARNING" not in captured.out
