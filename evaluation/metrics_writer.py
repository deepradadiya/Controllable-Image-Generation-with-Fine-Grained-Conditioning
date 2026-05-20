"""
Metrics JSON writer for the evaluation pipeline.

Assembles evaluation results into a structured metrics dictionary and
persists it to evaluation/results/metrics.json with human-readable formatting.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from evaluation.config import EvaluationConfig


# Mapping from condition type to the alignment metric name used
_ALIGNMENT_METRICS = {
    "depth": "pearson_correlation",
    "pose": "normalized_keypoint_distance",
    "edge": "ssim",
}

# Target threshold for alignment scores
_ALIGNMENT_TARGET = 0.70


def build_metrics_dict(
    fid_results: Dict[str, float],
    alignment_results: Dict[str, Tuple[float, float]],
    grid_paths: List[str],
    config: EvaluationConfig,
) -> Dict[str, Any]:
    """Assemble the full metrics dictionary with metadata.

    Combines FID scores, alignment scores, visual grid paths, and pipeline
    metadata into a single structured dictionary matching the metrics JSON schema.

    Args:
        fid_results: Dict mapping condition keys to FID scores.
            Expected keys: 'baseline', 'depth', 'pose', 'edge'.
        alignment_results: Dict mapping condition_type to (mean, std) tuples.
            Expected keys: 'depth', 'pose', 'edge'.
        grid_paths: List of saved grid file paths (e.g.,
            ['evaluation/results/visual_grid_depth.png', ...]).
        config: EvaluationConfig dataclass instance with pipeline parameters.

    Returns:
        Complete metrics dictionary ready for JSON serialization.
    """
    # Build metadata section
    metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "num_fid_samples": config.num_fid_samples,
        "num_alignment_samples": config.num_alignment_samples,
        "coco_val_size": config.num_fid_samples,
        "inference_config": {
            "guidance_scale": config.guidance_scale,
            "num_inference_steps": config.num_inference_steps,
            "image_size": 512,
        },
        "checkpoint_paths": {
            ct: os.path.join(config.checkpoint_dir, f"controlnet-sd15-{ct}")
            for ct in config.condition_types
        },
    }

    # Build fid_scores section
    fid_scores: Dict[str, float] = {}
    if "baseline" in fid_results:
        fid_scores["baseline_sd15"] = fid_results["baseline"]
    for ct in config.condition_types:
        if ct in fid_results:
            fid_scores[ct] = fid_results[ct]

    # Build alignment_scores section
    alignment_scores: Dict[str, Dict[str, Any]] = {}
    for ct, (mean, std) in alignment_results.items():
        alignment_scores[ct] = {
            "mean": mean,
            "std": std,
            "num_samples": config.num_alignment_samples,
            "metric": _ALIGNMENT_METRICS.get(ct, "unknown"),
            "target_met": mean >= _ALIGNMENT_TARGET,
        }

    # Build visual_grids section from grid_paths
    visual_grids: Dict[str, str] = {}
    for path in grid_paths:
        # Extract condition type from filename pattern: visual_grid_{type}.png
        filename = os.path.basename(path)
        if filename.startswith("visual_grid_") and filename.endswith(".png"):
            grid_type = filename[len("visual_grid_"):-len(".png")]
            visual_grids[grid_type] = path

    return {
        "metadata": metadata,
        "fid_scores": fid_scores,
        "alignment_scores": alignment_scores,
        "visual_grids": visual_grids,
    }


def save_metrics_json(metrics_dict: Dict[str, Any], output_dir: str) -> str:
    """Write metrics dictionary to evaluation/results/metrics.json.

    Creates the output directory if it doesn't exist, then serializes the
    metrics dictionary to JSON with 2-space indentation for readability.

    Args:
        metrics_dict: Complete metrics dictionary to serialize.
        output_dir: Path to the output directory (e.g., 'evaluation/results').

    Returns:
        The full path to the written metrics.json file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_file = output_path / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    return str(metrics_file)
