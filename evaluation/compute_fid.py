"""
FID Score Computation for Evaluation Pipeline.

This module implements the EvaluationFIDCalculator class that generates images
using the ControlNetPipeline and computes FID scores against the COCO 2017
validation set. It supports both conditioned generation (with ControlNet adapters)
and baseline generation (vanilla SD1.5 without conditioning).

The class leverages the existing FIDCalculator from src/evaluation/compute_fid.py
for Inception-v3 feature extraction and Fréchet distance computation.

Requirements Validated: 1.1, 1.2, 1.3, 1.4, 1.5, 1.8
"""

import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from model.pipeline import ControlNetPipeline
from src.evaluation.compute_fid import FIDCalculator

logger = logging.getLogger(__name__)

# Supported image extensions for COCO validation set loading
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")


class EvaluationFIDCalculator:
    """
    Computes FID scores for trained ControlNet adapters against COCO validation set.

    Generates images using the ControlNetPipeline, extracts Inception-v3 features
    from both real (COCO) and generated images, and computes the Fréchet distance.
    Also computes a vanilla SD1.5 baseline for comparison.

    The class uses batch processing and fixed seeds for reproducibility and
    memory efficiency on T4 GPUs (15GB VRAM).
    """

    def __init__(
        self,
        pipeline: ControlNetPipeline,
        coco_val_dir: str,
        batch_size: int = 32,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the FID calculator.

        Args:
            pipeline: Loaded ControlNetPipeline for image generation.
            coco_val_dir: Path to COCO 2017 validation images directory.
            batch_size: Batch size for Inception-v3 feature extraction (default 32).
            device: Computation device (default: cuda if available).
        """
        self.pipeline = pipeline
        self.coco_val_dir = coco_val_dir
        self.batch_size = batch_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Initialize the FIDCalculator from src/evaluation for feature extraction
        # and Fréchet distance computation
        self.fid_calculator = FIDCalculator(
            batch_size=self.batch_size,
            device=self.device,
            num_workers=0,  # Use 0 workers for compatibility with PIL images
        )

    def generate_images(
        self,
        prompts: List[str],
        condition_maps: List[Image.Image],
        condition_type: str,
        num_images: int = 1000,
        seed: int = 42,
    ) -> List[Image.Image]:
        """Generate images using the ControlNet pipeline with fixed seeds.

        Generates num_images images by cycling through the provided prompts and
        condition maps. Uses a torch.Generator with incrementing seeds based on
        the provided base seed for reproducibility.

        Args:
            prompts: List of text prompts for generation. Will be cycled if
                fewer than num_images.
            condition_maps: List of condition map images (depth/pose/edge).
                Will be cycled if fewer than num_images.
            condition_type: One of "depth", "pose", or "edge".
            num_images: Number of images to generate (default 1000).
            seed: Base random seed for reproducibility (default 42).

        Returns:
            List of generated PIL Images.
        """
        generated_images: List[Image.Image] = []
        num_prompts = len(prompts)
        num_conditions = len(condition_maps)

        logger.info(
            f"Generating {num_images} images with ControlNet ({condition_type}) "
            f"using {num_prompts} prompts and {num_conditions} condition maps"
        )

        for i in range(num_images):
            # Cycle through prompts and condition maps
            prompt = prompts[i % num_prompts]
            condition_map = condition_maps[i % num_conditions]

            # Use incrementing seed for each image for reproducibility
            image_seed = seed + i

            try:
                image = self.pipeline(
                    text_prompt=prompt,
                    condition_image=condition_map,
                    condition_type=condition_type,
                    seed=image_seed,
                )
                generated_images.append(image)
            except Exception as e:
                logger.warning(
                    f"Failed to generate image {i} with seed {image_seed}: {e}"
                )
                # Generate a blank placeholder to maintain count consistency
                generated_images.append(Image.new("RGB", (512, 512), (0, 0, 0)))

            if (i + 1) % 50 == 0:
                logger.info(f"Generated {i + 1}/{num_images} images")

        logger.info(f"Completed generating {len(generated_images)} images")
        return generated_images

    def generate_baseline_images(
        self,
        prompts: List[str],
        num_images: int = 1000,
        seed: int = 42,
    ) -> List[Image.Image]:
        """Generate images using vanilla SD1.5 (no ControlNet conditioning).

        Generates baseline images by passing a zero-valued condition map to the
        pipeline, effectively disabling ControlNet influence. This produces
        vanilla SD1.5 outputs for FID baseline comparison.

        Uses fixed seeds for reproducibility, matching the same seed sequence
        as generate_images() so results are directly comparable.

        Args:
            prompts: List of text prompts for generation. Will be cycled if
                fewer than num_images.
            num_images: Number of images to generate (default 1000).
            seed: Base random seed for reproducibility (default 42).

        Returns:
            List of generated PIL Images (vanilla SD1.5 outputs).
        """
        generated_images: List[Image.Image] = []
        num_prompts = len(prompts)

        # Create a zero condition map (black image) to disable ControlNet influence
        # The zero convolutions in ControlNet ensure zero input produces zero output
        zero_condition = Image.new("RGB", (512, 512), (0, 0, 0))

        logger.info(
            f"Generating {num_images} baseline images (vanilla SD1.5, no conditioning)"
        )

        for i in range(num_images):
            prompt = prompts[i % num_prompts]
            image_seed = seed + i

            try:
                # Use "depth" as condition_type with zero map — the zero condition
                # means ControlNet contributes nothing, giving vanilla SD1.5 output
                image = self.pipeline(
                    text_prompt=prompt,
                    condition_image=zero_condition,
                    condition_type="depth",
                    seed=image_seed,
                )
                generated_images.append(image)
            except Exception as e:
                logger.warning(
                    f"Failed to generate baseline image {i} with seed {image_seed}: {e}"
                )
                generated_images.append(Image.new("RGB", (512, 512), (0, 0, 0)))

            if (i + 1) % 50 == 0:
                logger.info(f"Generated {i + 1}/{num_images} baseline images")

        logger.info(f"Completed generating {len(generated_images)} baseline images")
        return generated_images

    def load_coco_images(self, num_images: int = 1000) -> List[Image.Image]:
        """Randomly sample images from the COCO 2017 validation set.

        Scans the COCO validation directory for image files and randomly samples
        the requested number. Uses a fixed random seed for reproducibility.

        Args:
            num_images: Number of images to sample (default 1000).

        Returns:
            List of PIL Images loaded from the COCO validation set.

        Raises:
            FileNotFoundError: If the COCO validation directory does not exist.
            ValueError: If the directory contains fewer images than requested.
        """
        coco_path = Path(self.coco_val_dir)

        if not coco_path.exists():
            raise FileNotFoundError(
                f"COCO validation directory not found: {self.coco_val_dir}"
            )

        # Collect all image file paths
        image_paths: List[Path] = []
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(coco_path.glob(f"*{ext}"))
            image_paths.extend(coco_path.glob(f"*{ext.upper()}"))

        if len(image_paths) == 0:
            raise ValueError(
                f"No images found in COCO validation directory: {self.coco_val_dir}"
            )

        if len(image_paths) < num_images:
            logger.warning(
                f"COCO validation set has {len(image_paths)} images, "
                f"but {num_images} were requested. Using all available images."
            )
            num_images = len(image_paths)

        # Randomly sample with fixed seed for reproducibility
        rng = random.Random(42)
        sampled_paths = rng.sample(image_paths, num_images)

        logger.info(
            f"Loading {num_images} images from COCO validation set "
            f"({len(image_paths)} total available)"
        )

        # Load images
        loaded_images: List[Image.Image] = []
        for path in sampled_paths:
            try:
                img = Image.open(path).convert("RGB")
                loaded_images.append(img)
            except Exception as e:
                logger.warning(f"Failed to load image {path}: {e}")
                continue

        if len(loaded_images) == 0:
            raise ValueError(
                f"Failed to load any images from COCO validation directory: "
                f"{self.coco_val_dir}"
            )

        logger.info(f"Successfully loaded {len(loaded_images)} COCO images")
        return loaded_images

    def compute_fid_for_condition(
        self,
        condition_type: str,
        prompts: List[str],
        condition_maps: List[Image.Image],
        num_images: int = 1000,
    ) -> float:
        """Compute FID for a single condition type against COCO validation set.

        This method orchestrates the full FID computation pipeline:
        1. Generates images using the ControlNet pipeline with the given condition
        2. Loads real images from the COCO 2017 validation set
        3. Extracts Inception-v3 features (2048-dim) from both sets using batch
           processing to stay within T4 GPU memory constraints
        4. Computes the Fréchet distance between the two feature distributions

        The batch processing uses self.batch_size (default 32) for feature
        extraction, ensuring peak GPU memory stays within T4 constraints (15GB VRAM).

        Args:
            condition_type: One of "depth", "pose", or "edge".
            prompts: List of text prompts for generation. Will be cycled if
                fewer than num_images.
            condition_maps: List of condition map images for the given condition type.
                Will be cycled if fewer than num_images.
            num_images: Number of images to generate and compare (default 1000).

        Returns:
            FID score as a float (lower is better, 0 means identical distributions).

        Raises:
            FileNotFoundError: If COCO validation directory does not exist.
            ValueError: If no images can be loaded or generated.
        """
        logger.info(
            f"Computing FID for condition type '{condition_type}' "
            f"with {num_images} images"
        )

        # Step 1: Generate images using the ControlNet pipeline
        generated_images = self.generate_images(
            prompts=prompts,
            condition_maps=condition_maps,
            condition_type=condition_type,
            num_images=num_images,
            seed=42,
        )

        # Step 2: Load real images from COCO validation set
        real_images = self.load_coco_images(num_images=num_images)

        # Step 3 & 4: Extract features and compute FID using the FIDCalculator
        # The FIDCalculator handles batch processing internally with self.batch_size
        logger.info(
            f"Extracting Inception-v3 features from {len(real_images)} real "
            f"and {len(generated_images)} generated images "
            f"(batch_size={self.batch_size})"
        )

        # Extract features from real images in batches
        real_features = self.fid_calculator.extract_features(
            real_images, show_progress=True
        )

        # Extract features from generated images in batches
        generated_features = self.fid_calculator.extract_features(
            generated_images, show_progress=True
        )

        # Compute distribution statistics (mean and covariance)
        real_mean, real_cov = self.fid_calculator.compute_statistics(real_features)
        gen_mean, gen_cov = self.fid_calculator.compute_statistics(generated_features)

        # Compute Fréchet distance between the two distributions
        fid_score = self.fid_calculator.calculate_frechet_distance(
            real_mean, real_cov, gen_mean, gen_cov
        )

        logger.info(
            f"FID score for {condition_type}: {fid_score:.3f} "
            f"(real: {len(real_images)} images, generated: {len(generated_images)} images)"
        )

        return fid_score

    def run_full_evaluation(
        self,
        prompts: List[str],
        condition_maps: Dict[str, List[Image.Image]],
        num_images: int = 1000,
    ) -> Dict[str, float]:
        """
        Run FID evaluation for all condition types + baseline.

        Computes FID scores for each condition type (depth, pose, edge) using
        the trained ControlNet adapters, and also computes a baseline FID using
        vanilla SD1.5 (no conditioning). Condition types that fail (e.g., due
        to missing checkpoints) are skipped with a warning.

        Args:
            prompts: List of text prompts for generation.
            condition_maps: Dict mapping condition type ('depth', 'pose', 'edge')
                to a list of condition map images.
            num_images: Number of images to generate per condition type (default 1000).

        Returns:
            Dict with keys from {'baseline', 'depth', 'pose', 'edge'} mapping
            to FID scores (float). Keys are only present for successfully
            computed conditions.
        """
        results: Dict[str, float] = {}

        # Compute FID for each condition type
        for condition_type in ["depth", "pose", "edge"]:
            if condition_type not in condition_maps:
                logger.warning(
                    f"No condition maps provided for '{condition_type}', skipping."
                )
                continue

            try:
                fid_score = self.compute_fid_for_condition(
                    condition_type=condition_type,
                    prompts=prompts,
                    condition_maps=condition_maps[condition_type],
                    num_images=num_images,
                )
                results[condition_type] = fid_score
            except Exception as e:
                logger.warning(
                    f"Failed to compute FID for condition type '{condition_type}': {e}. "
                    f"Skipping this condition type."
                )
                continue

        # Compute baseline FID (vanilla SD1.5, no ControlNet)
        try:
            logger.info("Computing baseline FID (vanilla SD1.5, no conditioning)")
            baseline_images = self.generate_baseline_images(
                prompts=prompts,
                num_images=num_images,
                seed=42,
            )
            real_images = self.load_coco_images(num_images=num_images)

            # Extract features and compute FID for baseline
            real_features = self.fid_calculator.extract_features(
                real_images, show_progress=True
            )
            baseline_features = self.fid_calculator.extract_features(
                baseline_images, show_progress=True
            )

            real_mean, real_cov = self.fid_calculator.compute_statistics(real_features)
            baseline_mean, baseline_cov = self.fid_calculator.compute_statistics(
                baseline_features
            )

            baseline_fid = self.fid_calculator.calculate_frechet_distance(
                real_mean, real_cov, baseline_mean, baseline_cov
            )
            results["baseline"] = baseline_fid
            logger.info(f"Baseline FID score: {baseline_fid:.3f}")
        except Exception as e:
            logger.warning(f"Failed to compute baseline FID: {e}. Skipping baseline.")

        return results

    def print_results_table(self, results: Dict[str, float]) -> None:
        """Print formatted FID results table.

        Prints a markdown-style table with columns for Model and FID Score.
        Rows include SD1.5 baseline and each ControlNet condition type.
        Missing results (skipped conditions) are shown as 'N/A'.

        Args:
            results: Dict mapping condition keys ('baseline', 'depth', 'pose',
                'edge') to FID scores. Missing keys are displayed as 'N/A'.
        """
        # Define the row labels mapping condition keys to display names
        row_definitions = [
            ("baseline", "SD1.5 Baseline"),
            ("depth", "Depth ControlNet"),
            ("pose", "Pose ControlNet"),
            ("edge", "Edge ControlNet"),
        ]

        # Column widths
        model_col_width = 18
        score_col_width = 9

        # Print header
        header = f"| {'Model':<{model_col_width}} | {'FID Score':<{score_col_width}} |"
        separator = f"|{'-' * (model_col_width + 2)}|{'-' * (score_col_width + 2)}|"

        print(header)
        print(separator)

        # Print rows
        for key, display_name in row_definitions:
            if key in results:
                score_str = f"{results[key]:.1f}"
            else:
                score_str = "N/A"
            row = f"| {display_name:<{model_col_width}} | {score_str:<{score_col_width}} |"
            print(row)
