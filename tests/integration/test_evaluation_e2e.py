"""
End-to-End Integration Tests for the Evaluation Pipeline.

Tests the full evaluation flow using a mock ControlNetPipeline that returns
random 512x512 RGB images. Verifies metrics.json creation, PNG grid file
generation, and graceful handling when some condition types have no checkpoints.

Requirements Validated: 1.4, 2.5, 3.3, 4.1, 5.1
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from evaluation.config import EvaluationConfig
from evaluation.run_evaluation import run_evaluation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_random_image(width: int = 512, height: int = 512) -> Image.Image:
    """Create a random RGB image of the given size."""
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _make_mock_pipeline() -> MagicMock:
    """Create a mock ControlNetPipeline that returns random 512x512 images.

    The mock is callable and returns a random PIL Image regardless of inputs.
    """
    mock_pipeline = MagicMock()
    mock_pipeline.side_effect = lambda **kwargs: _create_random_image()
    # Also handle positional-style calls via __call__
    mock_pipeline.__call__ = lambda self, **kwargs: _create_random_image()
    return mock_pipeline


def _build_test_args(output_dir: str, condition_types: List[str]) -> argparse.Namespace:
    """Build a minimal argparse.Namespace for run_evaluation."""
    return argparse.Namespace(
        output_dir=output_dir,
        num_fid_samples=4,
        num_alignment_samples=4,
        num_grid_prompts=2,
        batch_size=2,
        condition_types=condition_types,
        coco_val_dir="data/raw/coco_val2017",
        checkpoint_dir="models/trained",
        seed=42,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pipeline():
    """Provide a mock ControlNetPipeline that returns random 512x512 images."""
    mock = MagicMock()
    mock.side_effect = lambda *args, **kwargs: _create_random_image()
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvaluationEndToEnd:
    """End-to-end integration tests for the evaluation pipeline."""

    @patch("evaluation.run_evaluation.validate_checkpoints")
    @patch("evaluation.run_evaluation.load_controlnet_pipeline")
    def test_full_evaluation_flow_creates_metrics_json(
        self, mock_load_pipeline, mock_validate, tmp_path, mock_pipeline
    ):
        """Test full evaluation flow with mock pipeline creates valid metrics.json.

        Validates: Requirements 1.4, 2.5, 4.1, 5.1
        """
        # Setup mocks
        mock_validate.return_value = ["depth", "edge"]
        mock_load_pipeline.return_value = mock_pipeline

        args = _build_test_args(
            output_dir=str(tmp_path), condition_types=["depth", "pose", "edge"]
        )

        # Run the evaluation pipeline
        result = run_evaluation(args)

        # Verify metrics.json was created
        metrics_path = tmp_path / "metrics.json"
        assert metrics_path.exists(), "metrics.json was not created"

        # Verify metrics.json has valid structure
        with open(metrics_path) as f:
            metrics = json.load(f)

        # Check required top-level keys
        assert "metadata" in metrics, "metrics.json missing 'metadata' key"
        assert "fid_scores" in metrics, "metrics.json missing 'fid_scores' key"
        assert "alignment_scores" in metrics, "metrics.json missing 'alignment_scores' key"
        assert "visual_grids" in metrics, "metrics.json missing 'visual_grids' key"

        # Verify metadata structure
        metadata = metrics["metadata"]
        assert "timestamp" in metadata
        assert "num_fid_samples" in metadata
        assert "num_alignment_samples" in metadata
        assert "inference_config" in metadata
        assert "checkpoint_paths" in metadata

        # Verify inference_config sub-structure
        inference_config = metadata["inference_config"]
        assert "guidance_scale" in inference_config
        assert "num_inference_steps" in inference_config
        assert "image_size" in inference_config

    @patch("evaluation.run_evaluation.validate_checkpoints")
    @patch("evaluation.run_evaluation.load_controlnet_pipeline")
    def test_full_evaluation_flow_creates_grid_files(
        self, mock_load_pipeline, mock_validate, tmp_path, mock_pipeline
    ):
        """Test full evaluation flow creates PNG grid files at expected paths.

        Validates: Requirements 3.3, 5.1
        """
        mock_validate.return_value = ["depth", "edge"]
        mock_load_pipeline.return_value = mock_pipeline

        args = _build_test_args(
            output_dir=str(tmp_path), condition_types=["depth", "edge"]
        )

        result = run_evaluation(args)

        # Verify grid files were created
        grid_paths = result.get("grid_paths", [])
        assert len(grid_paths) > 0, "No grid files were generated"

        # Check that the files actually exist and are valid PNGs
        for grid_path in grid_paths:
            assert os.path.exists(grid_path), f"Grid file not found: {grid_path}"
            # Verify it's a valid PNG by opening it
            img = Image.open(grid_path)
            assert img.format == "PNG" or grid_path.endswith(".png")
            assert img.size[0] > 0 and img.size[1] > 0

    @patch("evaluation.run_evaluation.validate_checkpoints")
    @patch("evaluation.run_evaluation.load_controlnet_pipeline")
    def test_graceful_handling_partial_condition_types(
        self, mock_load_pipeline, mock_validate, tmp_path, mock_pipeline
    ):
        """Test pipeline completes with partial results when some conditions have no checkpoints.

        When validate_checkpoints returns only a subset of condition types,
        the pipeline should still complete successfully with results for the
        available types.

        Validates: Requirements 1.4, 5.1
        """
        # Only "depth" has a valid checkpoint; "pose" and "edge" do not
        mock_validate.return_value = ["depth"]
        mock_load_pipeline.return_value = mock_pipeline

        args = _build_test_args(
            output_dir=str(tmp_path), condition_types=["depth", "pose", "edge"]
        )

        # Pipeline should complete without error
        result = run_evaluation(args)

        # Verify metrics.json was still created
        metrics_path = tmp_path / "metrics.json"
        assert metrics_path.exists(), "metrics.json not created with partial conditions"

        with open(metrics_path) as f:
            metrics = json.load(f)

        # Should still have all required top-level keys
        assert "metadata" in metrics
        assert "fid_scores" in metrics
        assert "alignment_scores" in metrics
        assert "visual_grids" in metrics

        # The result dict should be returned successfully
        assert "fid_results" in result
        assert "alignment_results" in result
        assert "grid_paths" in result

    @patch("evaluation.run_evaluation.validate_checkpoints")
    @patch("evaluation.run_evaluation.load_controlnet_pipeline")
    def test_metrics_json_fid_scores_structure(
        self, mock_load_pipeline, mock_validate, tmp_path, mock_pipeline
    ):
        """Test that fid_scores in metrics.json has valid numeric values.

        Validates: Requirements 1.4, 4.1
        """
        mock_validate.return_value = ["depth", "edge"]
        mock_load_pipeline.return_value = mock_pipeline

        args = _build_test_args(
            output_dir=str(tmp_path), condition_types=["depth", "edge"]
        )

        run_evaluation(args)

        metrics_path = tmp_path / "metrics.json"
        with open(metrics_path) as f:
            metrics = json.load(f)

        fid_scores = metrics["fid_scores"]
        # All FID scores should be numeric (int or float)
        for key, value in fid_scores.items():
            assert isinstance(value, (int, float)), (
                f"FID score for '{key}' is not numeric: {value}"
            )

    @patch("evaluation.run_evaluation.validate_checkpoints")
    @patch("evaluation.run_evaluation.load_controlnet_pipeline")
    def test_metrics_json_alignment_scores_structure(
        self, mock_load_pipeline, mock_validate, tmp_path, mock_pipeline
    ):
        """Test that alignment_scores in metrics.json has valid structure per condition.

        Each condition type entry should have: mean, std, num_samples, metric, target_met.

        Validates: Requirements 2.5, 4.1
        """
        mock_validate.return_value = ["depth", "edge"]
        mock_load_pipeline.return_value = mock_pipeline

        args = _build_test_args(
            output_dir=str(tmp_path), condition_types=["depth", "edge"]
        )

        run_evaluation(args)

        metrics_path = tmp_path / "metrics.json"
        with open(metrics_path) as f:
            metrics = json.load(f)

        alignment_scores = metrics["alignment_scores"]
        for condition_type, scores in alignment_scores.items():
            assert "mean" in scores, f"Missing 'mean' for {condition_type}"
            assert "std" in scores, f"Missing 'std' for {condition_type}"
            assert "num_samples" in scores, f"Missing 'num_samples' for {condition_type}"
            assert "metric" in scores, f"Missing 'metric' for {condition_type}"
            assert "target_met" in scores, f"Missing 'target_met' for {condition_type}"
            # Verify types
            assert isinstance(scores["mean"], (int, float))
            assert isinstance(scores["std"], (int, float))
            assert isinstance(scores["num_samples"], int)
            assert isinstance(scores["metric"], str)
            assert isinstance(scores["target_met"], bool)

    @patch("evaluation.run_evaluation.validate_checkpoints")
    @patch("evaluation.run_evaluation.load_controlnet_pipeline")
    def test_result_dict_structure(
        self, mock_load_pipeline, mock_validate, tmp_path, mock_pipeline
    ):
        """Test that run_evaluation returns a well-structured result dict.

        Validates: Requirements 5.1
        """
        mock_validate.return_value = ["depth"]
        mock_load_pipeline.return_value = mock_pipeline

        args = _build_test_args(
            output_dir=str(tmp_path), condition_types=["depth"]
        )

        result = run_evaluation(args)

        # Verify result dict has expected keys
        assert "fid_results" in result
        assert "alignment_results" in result
        assert "grid_paths" in result
        assert "metrics_path" in result

        # metrics_path should point to an existing file
        if result["metrics_path"]:
            assert os.path.exists(result["metrics_path"])
