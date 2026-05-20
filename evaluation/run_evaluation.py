"""
Pipeline Orchestrator for ControlNet Evaluation.

Provides the main entry point script that orchestrates FID computation,
condition alignment measurement, and visual grid generation in sequence.
Each module is wrapped in try/except so that a failure in one module does
not abort the entire pipeline.

Usage:
    python -m evaluation.run_evaluation [OPTIONS]

Requirements Validated: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from evaluation.compute_fid import EvaluationFIDCalculator
from evaluation.condition_alignment import EvaluationAlignmentCalculator
from evaluation.config import EvaluationConfig, ensure_output_dir, load_test_prompts
from evaluation.metrics_writer import build_metrics_dict, save_metrics_json
from evaluation.pipeline_loader import (
    load_baseline_pipeline,
    load_controlnet_pipeline,
    validate_checkpoints,
)
from evaluation.visual_grid import EvaluationGridGenerator

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation pipeline.

    Defines CLI arguments for controlling the evaluation pipeline including
    output directory, sample counts, batch size, condition types, and paths
    to data and checkpoints.

    Returns:
        Parsed argument namespace with all evaluation parameters.
    """
    parser = argparse.ArgumentParser(
        description="Run the complete ControlNet evaluation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation/results",
        help="Directory to save evaluation outputs (metrics.json, grids).",
    )
    parser.add_argument(
        "--num_fid_samples",
        type=int,
        default=1000,
        help="Number of images to generate for FID computation.",
    )
    parser.add_argument(
        "--num_alignment_samples",
        type=int,
        default=100,
        help="Number of image-condition pairs for alignment evaluation.",
    )
    parser.add_argument(
        "--num_grid_prompts",
        type=int,
        default=20,
        help="Number of rows (prompts) per visual grid.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for Inception-v3 feature extraction.",
    )
    parser.add_argument(
        "--condition_types",
        type=str,
        nargs="+",
        default=["depth", "pose", "edge"],
        help="Condition types to evaluate.",
    )
    parser.add_argument(
        "--coco_val_dir",
        type=str,
        default="data/raw/coco_val2017",
        help="Path to COCO 2017 validation images directory.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="models/trained",
        help="Directory containing trained ControlNet checkpoints.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()


def _load_condition_maps(
    condition_types: List[str],
    coco_val_dir: str,
    num_images: int = 20,
) -> Tuple[List[Image.Image], Dict[str, List[Image.Image]]]:
    """Load original images and generate placeholder condition maps for evaluation.

    In a full deployment, condition maps would be pre-computed from the dataset.
    This function loads original images from the COCO validation set and creates
    synthetic condition maps (placeholder images) for each condition type.

    Args:
        condition_types: List of condition types to generate maps for.
        coco_val_dir: Path to COCO 2017 validation images.
        num_images: Number of images to load.

    Returns:
        Tuple of (original_images, condition_maps_by_type).
    """
    original_images: List[Image.Image] = []
    coco_path = Path(coco_val_dir)

    if coco_path.exists():
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        image_paths: List[Path] = []
        for ext in image_extensions:
            image_paths.extend(coco_path.glob(f"*{ext}"))
            image_paths.extend(coco_path.glob(f"*{ext.upper()}"))

        image_paths = sorted(image_paths)[:num_images]

        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                original_images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")
                continue

    # If no COCO images available, create placeholder images
    if len(original_images) == 0:
        logger.warning(
            f"No images found in {coco_val_dir}. Using placeholder images."
        )
        for _ in range(num_images):
            original_images.append(Image.new("RGB", (512, 512), (128, 128, 128)))

    # Generate condition maps for each type
    # In production, these would be pre-computed (depth via MiDaS, pose via DWPose, edge via Canny)
    condition_maps: Dict[str, List[Image.Image]] = {}
    for condition_type in condition_types:
        maps: List[Image.Image] = []
        for img in original_images:
            # Create a synthetic condition map from the original image
            maps.append(_create_condition_map(img, condition_type))
        condition_maps[condition_type] = maps

    return original_images, condition_maps


def _create_condition_map(image: Image.Image, condition_type: str) -> Image.Image:
    """Create a condition map from an original image.

    Generates a condition map appropriate for the given condition type:
    - edge: Canny edge detection
    - depth: Grayscale approximation (lightweight proxy)
    - pose: Edge-based skeleton approximation

    Args:
        image: Original PIL Image.
        condition_type: One of 'depth', 'pose', or 'edge'.

    Returns:
        Condition map as a PIL Image.
    """
    import cv2
    import numpy as np

    img_array = np.array(image.convert("RGB"))

    if condition_type == "edge":
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return Image.fromarray(edges).convert("RGB")
    elif condition_type == "depth":
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # Use blurred grayscale as a depth proxy
        depth = cv2.GaussianBlur(gray, (5, 5), 0)
        return Image.fromarray(depth).convert("RGB")
    elif condition_type == "pose":
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # Dilate to simulate skeleton lines
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skeleton = cv2.dilate(edges, kernel, iterations=1)
        return Image.fromarray(skeleton).convert("RGB")
    else:
        # Fallback: return grayscale version
        return image.convert("L").convert("RGB")


def run_evaluation(args: argparse.Namespace) -> Dict:
    """Main evaluation pipeline orchestrator.

    Orchestrates the full evaluation pipeline:
    1. Validates that checkpoints exist for at least one condition type
    2. Loads test data (prompts, original images, condition maps)
    3. Runs FID computation (wrapped in try/except)
    4. Runs condition alignment (wrapped in try/except)
    5. Generates visual grids (wrapped in try/except)
    6. Saves metrics.json with whatever results succeeded
    7. Prints summary with total time, FID scores, alignment scores, and output paths

    Each module failure is caught, logged, and the pipeline continues with
    remaining modules.

    Args:
        args: Parsed command-line arguments namespace containing:
            - output_dir: Output directory path
            - num_fid_samples: Number of FID samples
            - num_alignment_samples: Number of alignment samples
            - num_grid_prompts: Number of grid prompts
            - batch_size: Batch size for feature extraction
            - condition_types: List of condition types to evaluate
            - coco_val_dir: Path to COCO validation images
            - checkpoint_dir: Path to trained checkpoints
            - seed: Random seed

    Returns:
        Dictionary containing evaluation results with keys:
        'fid_results', 'alignment_results', 'grid_paths', 'metrics_path'.
        Values may be empty dicts/lists if modules failed.
    """
    start_time = time.time()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("=" * 70)
    print("ControlNet Evaluation Pipeline")
    print("=" * 70)

    # Build config from args
    config = EvaluationConfig(
        output_dir=args.output_dir,
        num_fid_samples=args.num_fid_samples,
        num_alignment_samples=args.num_alignment_samples,
        num_grid_prompts=args.num_grid_prompts,
        batch_size=args.batch_size,
        condition_types=args.condition_types,
        coco_val_dir=args.coco_val_dir,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
    )

    # Ensure output directory exists
    ensure_output_dir(config.output_dir)

    # -------------------------------------------------------------------------
    # Step 1: Validate checkpoints exist for at least one condition type
    # -------------------------------------------------------------------------
    print("\n[1/6] Validating checkpoints...")
    valid_condition_types = validate_checkpoints(
        checkpoint_dir=config.checkpoint_dir,
        condition_types=config.condition_types,
    )

    if not valid_condition_types:
        print(
            f"\nERROR: No valid checkpoints found for any condition type "
            f"in '{config.checkpoint_dir}'."
        )
        print("Expected checkpoint directories:")
        for ct in config.condition_types:
            expected_path = os.path.join(
                config.checkpoint_dir, f"controlnet-sd15-{ct}"
            )
            print(f"  - {expected_path}")
        print("\nAborting evaluation. Please train at least one ControlNet adapter.")
        sys.exit(1)

    print(f"  Valid condition types: {valid_condition_types}")

    # -------------------------------------------------------------------------
    # Step 2: Load test data (prompts, original images, condition maps)
    # -------------------------------------------------------------------------
    print("\n[2/6] Loading test data...")
    prompts = load_test_prompts()
    original_images, condition_maps = _load_condition_maps(
        condition_types=valid_condition_types,
        coco_val_dir=config.coco_val_dir,
        num_images=config.num_grid_prompts,
    )
    print(f"  Loaded {len(prompts)} prompts, {len(original_images)} images")
    print(f"  Condition maps for: {list(condition_maps.keys())}")

    # Load the pipeline for the first valid condition type
    # (used for FID, alignment, and grid generation)
    print("\n  Loading ControlNet pipeline...")
    pipeline = None
    for ct in valid_condition_types:
        pipeline = load_controlnet_pipeline(
            condition_type=ct,
            checkpoint_dir=config.checkpoint_dir,
        )
        if pipeline is not None:
            print(f"  Pipeline loaded for condition type: {ct}")
            break

    if pipeline is None:
        print("\nERROR: Failed to load pipeline for any valid condition type.")
        print("Aborting evaluation.")
        sys.exit(1)

    # Initialize result containers
    fid_results: Dict[str, float] = {}
    alignment_results: Dict[str, Tuple[float, float]] = {}
    grid_paths: List[str] = []

    # -------------------------------------------------------------------------
    # Step 3: Run FID computation (wrapped in try/except)
    # -------------------------------------------------------------------------
    print("\n[3/6] Running FID computation...")
    try:
        fid_calculator = EvaluationFIDCalculator(
            pipeline=pipeline,
            coco_val_dir=config.coco_val_dir,
            batch_size=config.batch_size,
        )
        fid_results = fid_calculator.run_full_evaluation(
            prompts=prompts,
            condition_maps=condition_maps,
            num_images=config.num_fid_samples,
        )
        fid_calculator.print_results_table(fid_results)
        print("  FID computation completed successfully.")
    except Exception as e:
        logger.error(f"FID computation failed: {e}")
        print(f"  ERROR: FID computation failed: {e}")
        print("  Continuing with remaining modules...")

    # -------------------------------------------------------------------------
    # Step 4: Run condition alignment (wrapped in try/except)
    # -------------------------------------------------------------------------
    print("\n[4/6] Running condition alignment...")
    try:
        alignment_calculator = EvaluationAlignmentCalculator(
            pipeline=pipeline,
        )
        alignment_results = alignment_calculator.run_full_evaluation(
            prompts=prompts,
            condition_maps=condition_maps,
            num_samples=config.num_alignment_samples,
        )
        alignment_calculator.print_results_table(alignment_results)
        print("  Condition alignment completed successfully.")
    except Exception as e:
        logger.error(f"Condition alignment failed: {e}")
        print(f"  ERROR: Condition alignment failed: {e}")
        print("  Continuing with remaining modules...")

    # -------------------------------------------------------------------------
    # Step 5: Generate visual grids (wrapped in try/except)
    # -------------------------------------------------------------------------
    print("\n[5/6] Generating visual grids...")
    try:
        grid_generator = EvaluationGridGenerator(
            pipeline=pipeline,
            output_dir=config.output_dir,
        )
        grid_paths = grid_generator.save_all_grids(
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
        )
        print(f"  Generated {len(grid_paths)} grid files.")
        for path in grid_paths:
            print(f"    - {path}")
    except Exception as e:
        logger.error(f"Visual grid generation failed: {e}")
        print(f"  ERROR: Visual grid generation failed: {e}")
        print("  Continuing with remaining modules...")

    # -------------------------------------------------------------------------
    # Step 6: Save metrics.json with whatever results succeeded
    # -------------------------------------------------------------------------
    print("\n[6/6] Saving metrics...")
    metrics_path = ""
    try:
        metrics_dict = build_metrics_dict(
            fid_results=fid_results,
            alignment_results=alignment_results,
            grid_paths=grid_paths,
            config=config,
        )
        metrics_path = save_metrics_json(metrics_dict, config.output_dir)
        print(f"  Metrics saved to: {metrics_path}")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
        print(f"  ERROR: Failed to save metrics: {e}")

    # -------------------------------------------------------------------------
    # Step 7: Print summary
    # -------------------------------------------------------------------------
    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"\n  Total evaluation time: {total_time:.1f}s ({total_time / 60:.1f} min)")

    # FID scores summary
    print("\n  FID Scores:")
    if fid_results:
        for key, score in fid_results.items():
            label = "SD1.5 Baseline" if key == "baseline" else f"{key.capitalize()} ControlNet"
            print(f"    {label}: {score:.2f}")
    else:
        print("    (not computed)")

    # Alignment scores summary
    print("\n  Alignment Scores:")
    if alignment_results:
        for ct, (mean, std) in alignment_results.items():
            target_met = "✓" if mean >= 0.70 else "✗"
            print(f"    {ct}: mean={mean:.4f}, std={std:.4f} {target_met}")
    else:
        print("    (not computed)")

    # Output paths
    print("\n  Output Files:")
    if metrics_path:
        print(f"    Metrics: {metrics_path}")
    if grid_paths:
        for path in grid_paths:
            print(f"    Grid: {path}")
    if not metrics_path and not grid_paths:
        print("    (no output files generated)")

    print("\n" + "=" * 70)

    return {
        "fid_results": fid_results,
        "alignment_results": alignment_results,
        "grid_paths": grid_paths,
        "metrics_path": metrics_path,
    }


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(args)
