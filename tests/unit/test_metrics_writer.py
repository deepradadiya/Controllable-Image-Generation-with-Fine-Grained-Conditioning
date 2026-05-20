"""Unit tests for evaluation/metrics_writer.py."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from evaluation.config import EvaluationConfig
from evaluation.metrics_writer import build_metrics_dict, save_metrics_json


@pytest.fixture
def sample_config():
    """Create a sample EvaluationConfig for testing."""
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


@pytest.fixture
def sample_fid_results():
    """Sample FID results dict."""
    return {
        "baseline": 45.2,
        "depth": 17.3,
        "pose": 16.8,
        "edge": 17.9,
    }


@pytest.fixture
def sample_alignment_results():
    """Sample alignment results dict."""
    return {
        "depth": (0.74, 0.08),
        "pose": (0.72, 0.11),
        "edge": (0.76, 0.06),
    }


@pytest.fixture
def sample_grid_paths():
    """Sample grid file paths."""
    return [
        "evaluation/results/visual_grid_depth.png",
        "evaluation/results/visual_grid_pose.png",
        "evaluation/results/visual_grid_edge.png",
        "evaluation/results/visual_grid_combined.png",
    ]


class TestBuildMetricsDict:
    """Tests for build_metrics_dict function."""

    def test_contains_all_top_level_keys(
        self, sample_fid_results, sample_alignment_results, sample_grid_paths, sample_config
    ):
        """Metrics dict contains all required top-level keys."""
        result = build_metrics_dict(
            sample_fid_results, sample_alignment_results, sample_grid_paths, sample_config
        )
        assert "metadata" in result
        assert "fid_scores" in result
        assert "alignment_scores" in result
        assert "visual_grids" in result

    def test_metadata_has_timestamp(
        self, sample_fid_results, sample_alignment_results, sample_grid_paths, sample_config
    ):
        """Metadata includes an ISO format timestamp ending with Z."""
        result = build_metrics_dict(
            sample_fid_results, sample_alignment_results, sample_grid_paths, sample_config
        )
        ts = result["metadata"]["timestamp"]
        assert ts.endswith("Z")
        # Should be parseable as ISO format (minus the Z)
        from datetime import datetime
        datetime.fromisoformat(ts[:-1])

    def test_metadata_sample_counts(
        self, sample_fid_results, sample_alignment_results, sample_grid_paths, sample_config
    ):
        """Metadata includes correct sample counts from config."""
        result = build_metrics_dict(
            sample_fid_results, sample_alignment_results, sample_grid_paths, sample_config
        )
        meta = result["metadata"]
        assert meta["num_fid_samples"] == 1000
        assert meta["num_alignment_samples"] == 100
        assert meta["coco_val_size"] == 1000

    def test_metadata_inference_config(
        self, sample_fid_results, sample_alignment_results, sample_grid_paths, sample_config
    ):
        """Metadata includes inference configuration."""
        result = build_metrics_dict(
            sample_fid_results, sample_alignment_results, sample_grid_paths, sample_config
        )
        inf_config = result["metadata"]["inference_config"]
        assert inf_config["guidance_scale"] == 7.5
        assert inf_config["num_inference_steps"] == 20
        assert inf_config["image_size"] == 512

    def test_metadata_checkpoint_paths(
        self, sample_fid_results, sample_alignment_results, sample_grid_paths, sample_config
    ):
        """Metadata includes checkpoint paths for all condition types."""
        result = build_metrics_dict(
            sample_fid_results, sample_alignment_results, sample_grid_paths, sample_config
        )
        cp = result["metadata"]["checkpoint_paths"]
        assert "depth" in cp
        assert "pose" in cp
        assert "edge" in cp
        assert "controlnet-sd15-depth" in cp["depth"]

    def test_fid_scores_baseline_renamed(
        self, sample_fid_results, sample_alignment_results, sample_grid_paths, sample_config
    ):
        """FID scores renames 'baseline' key to 'baseline_sd15'."""
        result = build_metrics_dict(
            sample_fid_results, sample_alignment_results, sample_grid_paths, sample_config
        )
        assert "baseline_sd15" in result["fid_scores"]
        assert result["fid_scores"]["baseline_sd15"] == 45.2

    def test_fid_scores_per_condition(
        self, sample_fid_results, sample_alignment_results, sample_grid_paths, sample_config
    ):
        """FID scores include all condition types."""
        result = build_metrics_dict(
            sample_fid_results, sample_alignment_results, sample_grid_paths, sample_config
        )
        assert result["fid_scores"]["depth"] == 17.3
        assert result["fid_scores"]["pose"] == 16.8
        assert result["fid_scores"]["edge"] == 17.9

    def test_alignment_scores_structure(
        self, sample_fid_results, sample_alignment_results, sample_grid_paths, sample_config
    ):
        """Alignment scores have correct structure per condition type."""
        result = build_metrics_dict(
            sample_fid_results, sample_alignment_results, sample_grid_paths, sample_config
        )
        depth_align = result["alignment_scores"]["depth"]
        assert depth_align["mean"] == 0.74
        assert depth_align["std"] == 0.08
        assert depth_align["num_samples"] == 100
        assert depth_align["metric"] == "pearson_correlation"
        assert depth_align["target_met"] is True

    def test_alignment_target_met_false_when_below_threshold(
        self, sample_fid_results, sample_grid_paths, sample_config
    ):
        """target_met is False when mean alignment is below 0.70."""
        alignment_results = {
            "depth": (0.65, 0.10),
            "pose": (0.72, 0.11),
            "edge": (0.76, 0.06),
        }
        result = build_metrics_dict(
            sample_fid_results, alignment_results, sample_grid_paths, sample_config
        )
        assert result["alignment_scores"]["depth"]["target_met"] is False
        assert result["alignment_scores"]["pose"]["target_met"] is True

    def test_alignment_metric_names(
        self, sample_fid_results, sample_alignment_results, sample_grid_paths, sample_config
    ):
        """Each condition type uses the correct metric name."""
        result = build_metrics_dict(
            sample_fid_results, sample_alignment_results, sample_grid_paths, sample_config
        )
        assert result["alignment_scores"]["depth"]["metric"] == "pearson_correlation"
        assert result["alignment_scores"]["pose"]["metric"] == "normalized_keypoint_distance"
        assert result["alignment_scores"]["edge"]["metric"] == "ssim"

    def test_visual_grids_extracted_from_paths(
        self, sample_fid_results, sample_alignment_results, sample_grid_paths, sample_config
    ):
        """Visual grids section maps condition types to file paths."""
        result = build_metrics_dict(
            sample_fid_results, sample_alignment_results, sample_grid_paths, sample_config
        )
        grids = result["visual_grids"]
        assert grids["depth"] == "evaluation/results/visual_grid_depth.png"
        assert grids["pose"] == "evaluation/results/visual_grid_pose.png"
        assert grids["edge"] == "evaluation/results/visual_grid_edge.png"
        assert grids["combined"] == "evaluation/results/visual_grid_combined.png"

    def test_partial_fid_results(
        self, sample_alignment_results, sample_grid_paths, sample_config
    ):
        """Handles partial FID results (missing some condition types)."""
        fid_results = {"baseline": 45.2, "depth": 17.3}
        result = build_metrics_dict(
            fid_results, sample_alignment_results, sample_grid_paths, sample_config
        )
        assert "baseline_sd15" in result["fid_scores"]
        assert "depth" in result["fid_scores"]
        assert "pose" not in result["fid_scores"]
        assert "edge" not in result["fid_scores"]

    def test_empty_grid_paths(
        self, sample_fid_results, sample_alignment_results, sample_config
    ):
        """Handles empty grid paths list."""
        result = build_metrics_dict(
            sample_fid_results, sample_alignment_results, [], sample_config
        )
        assert result["visual_grids"] == {}


class TestSaveMetricsJson:
    """Tests for save_metrics_json function."""

    def test_creates_output_directory(self):
        """Creates the output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, "nested", "results")
            metrics_dict = {"test": "data"}
            save_metrics_json(metrics_dict, output_dir)
            assert os.path.isdir(output_dir)

    def test_writes_metrics_json_file(self):
        """Writes metrics.json to the output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_dict = {"fid_scores": {"depth": 17.3}}
            path = save_metrics_json(metrics_dict, tmpdir)
            assert os.path.isfile(path)
            assert path.endswith("metrics.json")

    def test_json_has_two_space_indentation(self):
        """Output JSON uses 2-space indentation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_dict = {"key": {"nested": "value"}}
            path = save_metrics_json(metrics_dict, tmpdir)
            with open(path, "r") as f:
                content = f.read()
            # 2-space indentation means nested keys are indented by 2 spaces
            assert '  "key"' in content
            assert '    "nested"' in content

    def test_round_trip_fidelity(self):
        """Serialized JSON can be deserialized back to equivalent dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_dict = {
                "metadata": {"timestamp": "2024-01-15T10:30:00Z"},
                "fid_scores": {"baseline_sd15": 45.2, "depth": 17.3},
                "alignment_scores": {
                    "depth": {"mean": 0.74, "std": 0.08}
                },
                "visual_grids": {"depth": "path.png"},
            }
            path = save_metrics_json(metrics_dict, tmpdir)
            with open(path, "r") as f:
                loaded = json.load(f)
            assert loaded == metrics_dict

    def test_returns_file_path(self):
        """Returns the full path to the written file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_metrics_json({"test": True}, tmpdir)
            expected = str(Path(tmpdir) / "metrics.json")
            assert path == expected

    def test_overwrites_existing_file(self):
        """Overwrites an existing metrics.json file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_metrics_json({"version": 1}, tmpdir)
            save_metrics_json({"version": 2}, tmpdir)
            with open(os.path.join(tmpdir, "metrics.json"), "r") as f:
                loaded = json.load(f)
            assert loaded["version"] == 2
