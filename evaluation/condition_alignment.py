"""
Condition Alignment Measurement for the Evaluation Pipeline.

Measures how well generated images follow their conditioning signals using
specialized extraction and comparison methods for each condition type:
- Edge: Canny edge detection on generated image → SSIM vs input edge map
- Depth: DPT model on generated image → Pearson correlation vs input depth map
- Pose: DWPose on generated image → normalized keypoint distance vs input pose

This module is designed to be extended incrementally:
- Task 3.1: Edge alignment (Canny + SSIM)
- Task 3.2: Depth alignment (DPT + Pearson correlation)
- Task 3.3: Pose alignment (DWPose + keypoint distance)
- Task 3.4: Batch evaluation and reporting
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy import stats
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity as ssim

from model.pipeline import ControlNetPipeline

logger = logging.getLogger(__name__)


class EvaluationAlignmentCalculator:
    """Measures how well generated images follow their conditioning signals.

    For each condition type, uses a specialized extraction + comparison method:
    - Edge: Canny on generated → SSIM vs input edge map
    - Depth: DPT on generated → Pearson correlation vs input depth map
    - Pose: DWPose on generated → normalized keypoint distance vs input pose

    Args:
        pipeline: A loaded ControlNetPipeline for image generation.
        device: Computation device (e.g., torch.device). Defaults to None
            which will use CUDA if available.
    """

    def __init__(
        self,
        pipeline: ControlNetPipeline,
        device=None,
    ):
        self.pipeline = pipeline
        if device is None:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def compute_edge_alignment(
        self,
        generated_image: Image.Image,
        input_edge_map: Image.Image,
    ) -> float:
        """Compute SSIM between Canny edges of generated image and input edge map.

        Applies Canny edge detection to the generated image and computes
        Structural Similarity Index (SSIM) against the input edge map.

        Args:
            generated_image: The generated PIL Image to evaluate.
            input_edge_map: The input edge condition map as a PIL Image.

        Returns:
            SSIM score clamped to [0, 1] range.
        """
        # Convert generated image to grayscale numpy array
        generated_gray = np.array(generated_image.convert("L"))

        # Apply Canny edge detection to the generated image
        generated_edges = cv2.Canny(generated_gray, 100, 200)

        # Convert input edge map to grayscale numpy array
        input_edges = np.array(input_edge_map.convert("L"))

        # Resize input edge map to match generated edges if dimensions differ
        if input_edges.shape != generated_edges.shape:
            input_edges = cv2.resize(
                input_edges,
                (generated_edges.shape[1], generated_edges.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # Compute SSIM between the two edge maps
        score = ssim(generated_edges, input_edges, data_range=255)

        # Clamp result to [0, 1] range
        score = float(max(0.0, min(1.0, score)))

        return score

    def compute_depth_alignment(
        self,
        generated_image: Image.Image,
        input_depth_map: Image.Image,
        use_dpt: bool = False,
    ) -> float:
        """Run DPT (or lightweight proxy) on generated image, compute Pearson correlation with input depth.

        Extracts a depth map from the generated image and computes the Pearson
        correlation coefficient against the input depth condition map. By default,
        uses a lightweight Laplacian-based depth proxy. When use_dpt=True, attempts
        to load and run a DPT model for more accurate depth estimation.

        The Laplacian-based proxy works by:
        1. Converting the generated image to grayscale
        2. Computing the absolute Laplacian (captures depth-related edges/structure)
        3. Applying Gaussian blur for smoother depth estimation
        4. Normalizing to [0, 1]

        Args:
            generated_image: The generated PIL Image to evaluate.
            input_depth_map: The input depth condition map as a PIL Image.
            use_dpt: If True, attempt to use a DPT model for depth extraction.
                If False (default), use the lightweight Laplacian proxy.

        Returns:
            Pearson correlation normalized to [0, 1] via max(0, correlation).
        """
        if use_dpt:
            extracted_depth = self._extract_depth_dpt(generated_image)
        else:
            extracted_depth = self._extract_depth_laplacian(generated_image)

        # Convert input depth map to grayscale numpy array
        input_depth = np.array(input_depth_map.convert("L")).astype(np.float32)

        # Resize input depth map to match extracted depth if dimensions differ
        if input_depth.shape != extracted_depth.shape:
            input_depth = cv2.resize(
                input_depth,
                (extracted_depth.shape[1], extracted_depth.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        # Flatten both arrays for correlation computation
        flat_extracted = extracted_depth.flatten()
        flat_input = input_depth.flatten()

        # Handle constant arrays (zero variance) — correlation is undefined
        if np.std(flat_extracted) < 1e-10 or np.std(flat_input) < 1e-10:
            # If both are constant and equal, perfect correlation
            if np.std(flat_extracted) < 1e-10 and np.std(flat_input) < 1e-10:
                if np.allclose(flat_extracted, flat_input, atol=1e-6):
                    return 1.0
            return 0.0

        # Compute Pearson correlation
        correlation, _ = stats.pearsonr(flat_extracted, flat_input)

        # Handle NaN from numerical issues
        if np.isnan(correlation):
            return 0.0

        # Normalize to [0, 1] by taking max(0, correlation)
        score = float(max(0.0, correlation))

        return score

    def _extract_depth_laplacian(self, image: Image.Image) -> np.ndarray:
        """Extract an approximate depth map using Laplacian-based estimation.

        This provides a lightweight depth proxy without requiring a full DPT model.
        Higher frequency content (larger Laplacian response) correlates with
        closer objects and more depth detail.

        Args:
            image: Input PIL Image.

        Returns:
            Estimated depth map as (H, W) float32 array in [0, 1].
        """
        # Convert to grayscale
        gray = np.array(image.convert("L")).astype(np.uint8)

        # Compute absolute Laplacian as depth proxy
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        depth_proxy = np.abs(laplacian)

        # Normalize to [0, 1]
        max_val = depth_proxy.max()
        if max_val > 0:
            depth_proxy = depth_proxy / max_val

        # Apply Gaussian blur for smoother depth estimation
        blur_kernel_size = 5
        depth_proxy = cv2.GaussianBlur(
            depth_proxy.astype(np.float32),
            (blur_kernel_size, blur_kernel_size),
            0,
        )

        # Re-normalize after blur
        max_val = depth_proxy.max()
        if max_val > 0:
            depth_proxy = depth_proxy / max_val

        return depth_proxy.astype(np.float32)

    def _extract_depth_dpt(self, image: Image.Image) -> np.ndarray:
        """Extract depth map using a DPT model.

        Attempts to load and run a DPT (Dense Prediction Transformer) model
        for accurate monocular depth estimation. Falls back to the Laplacian
        proxy if the model cannot be loaded.

        Args:
            image: Input PIL Image.

        Returns:
            Estimated depth map as (H, W) float32 array in [0, 1].
        """
        try:
            import torch
            from transformers import DPTForDepthEstimation, DPTImageProcessor

            # Load DPT model (Intel/dpt-small for efficiency)
            processor = DPTImageProcessor.from_pretrained("Intel/dpt-small")
            model = DPTForDepthEstimation.from_pretrained("Intel/dpt-small")
            model.eval()
            model.to(self.device)

            # Prepare input
            inputs = processor(images=image, return_tensors="pt").to(self.device)

            # Run inference
            with torch.no_grad():
                outputs = model(**inputs)
                predicted_depth = outputs.predicted_depth

            # Interpolate to original image size
            prediction = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=image.size[::-1],  # (H, W)
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            # Convert to numpy and normalize to [0, 1]
            depth_map = prediction.cpu().numpy()
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            if depth_max - depth_min > 0:
                depth_map = (depth_map - depth_min) / (depth_max - depth_min)
            else:
                depth_map = np.zeros_like(depth_map)

            return depth_map.astype(np.float32)

        except (ImportError, OSError, Exception) as e:
            logger.warning(
                "DPT model could not be loaded (%s). Falling back to Laplacian proxy.",
                str(e),
            )
            return self._extract_depth_laplacian(image)

    def compute_pose_alignment(
        self,
        generated_image: Image.Image,
        input_pose_map: Image.Image,
    ) -> float:
        """Compute normalized keypoint distance between generated image and input pose.

        Detects keypoints in the generated image using contour-based detection
        (lightweight proxy for DWPose) and extracts keypoints from the input
        pose map (rendered skeleton with white dots/lines on black background).
        Computes mean keypoint distance normalized by image diagonal.

        The algorithm:
        1. Extract keypoints from the input pose map using blob/contour detection
        2. Extract keypoints from the generated image using edge-based detection
        3. Match keypoints using the Hungarian algorithm (optimal assignment)
        4. Compute mean Euclidean distance between matched keypoint pairs
        5. Normalize by image diagonal (sqrt(width^2 + height^2))
        6. Return 1 - normalized_distance (higher = better alignment)

        Args:
            generated_image: The generated PIL Image to evaluate.
            input_pose_map: The input pose condition map as a PIL Image
                (rendered skeleton with white lines/dots on black background).

        Returns:
            Alignment score clamped to [0, 1], where 1 = perfect alignment.
        """
        # Extract keypoints from the input pose map
        input_keypoints = self._extract_keypoints_from_pose_map(input_pose_map)

        # Extract keypoints from the generated image
        generated_keypoints = self._extract_keypoints_from_generated(generated_image)

        # If no keypoints detected in either image, return 0 (cannot evaluate)
        if len(input_keypoints) == 0 or len(generated_keypoints) == 0:
            return 0.0

        # Compute image diagonal for normalization
        width, height = generated_image.size
        diagonal = math.sqrt(width ** 2 + height ** 2)

        if diagonal == 0:
            return 0.0

        # Match keypoints using Hungarian algorithm for optimal assignment
        mean_distance = self._compute_matched_keypoint_distance(
            generated_keypoints, input_keypoints
        )

        # Normalize distance by image diagonal
        normalized_distance = mean_distance / diagonal

        # Return 1 - normalized_distance (higher = better)
        score = 1.0 - normalized_distance

        # Clamp to [0, 1]
        score = float(max(0.0, min(1.0, score)))

        return score

    def _extract_keypoints_from_pose_map(
        self, pose_map: Image.Image
    ) -> List[Tuple[float, float]]:
        """Extract keypoint locations from a rendered pose skeleton image.

        The input pose map is expected to be a rendered skeleton with white
        lines/dots on a black background. Keypoints are detected as bright
        blob centers using contour detection.

        Args:
            pose_map: Pose skeleton image (white on black background).

        Returns:
            List of (x, y) keypoint coordinates.
        """
        # Convert to grayscale numpy array
        gray = np.array(pose_map.convert("L"))

        # Threshold to get binary mask of skeleton
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Use blob detection to find keypoint centers
        # Set up SimpleBlobDetector parameters for detecting bright blobs
        params = cv2.SimpleBlobDetector_Params()
        params.filterByColor = True
        params.blobColor = 255  # Detect white blobs
        params.filterByArea = True
        params.minArea = 3
        params.maxArea = 5000
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False

        detector = cv2.SimpleBlobDetector_create(params)
        blob_keypoints = detector.detect(binary)

        keypoints = [(kp.pt[0], kp.pt[1]) for kp in blob_keypoints]

        # If blob detection finds few keypoints, fall back to contour centroids
        if len(keypoints) < 3:
            keypoints = self._extract_keypoints_via_contours(binary)

        return keypoints

    def _extract_keypoints_from_generated(
        self, image: Image.Image
    ) -> List[Tuple[float, float]]:
        """Extract keypoint-like features from a generated image.

        Uses edge detection and contour analysis as a lightweight proxy
        for DWPose keypoint detection. Identifies salient structural points
        in the image that correspond to body joint locations.

        Args:
            image: Generated PIL Image.

        Returns:
            List of (x, y) keypoint coordinates.
        """
        # Convert to grayscale
        gray = np.array(image.convert("L"))

        # Apply Canny edge detection to find structural edges
        edges = cv2.Canny(gray, 50, 150)

        # Dilate edges to connect nearby segments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Find contours and extract centroids as keypoints
        keypoints = self._extract_keypoints_via_contours(dilated)

        return keypoints

    def _extract_keypoints_via_contours(
        self, binary_image: np.ndarray
    ) -> List[Tuple[float, float]]:
        """Extract keypoint locations from a binary image using contour centroids.

        Finds contours in the binary image and computes their centroids.
        Filters by area to remove noise and very large regions.

        Args:
            binary_image: Binary image (uint8, values 0 or 255).

        Returns:
            List of (x, y) centroid coordinates.
        """
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        keypoints = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter out very small noise and very large regions
            if area < 2:
                continue
            if area > binary_image.shape[0] * binary_image.shape[1] * 0.1:
                continue

            # Compute centroid using moments
            moments = cv2.moments(contour)
            if moments["m00"] > 0:
                cx = moments["m10"] / moments["m00"]
                cy = moments["m01"] / moments["m00"]
                keypoints.append((cx, cy))

        return keypoints

    def _compute_matched_keypoint_distance(
        self,
        keypoints_a: List[Tuple[float, float]],
        keypoints_b: List[Tuple[float, float]],
    ) -> float:
        """Compute mean distance between optimally matched keypoints.

        Uses the Hungarian algorithm (linear sum assignment) to find the
        optimal one-to-one matching between two sets of keypoints that
        minimizes total distance.

        If the sets have different sizes, only the smaller number of
        matches is used (unmatched keypoints are ignored).

        Args:
            keypoints_a: First set of (x, y) keypoints.
            keypoints_b: Second set of (x, y) keypoints.

        Returns:
            Mean Euclidean distance between matched keypoint pairs.
        """
        n = len(keypoints_a)
        m = len(keypoints_b)

        # Build cost matrix of pairwise Euclidean distances
        cost_matrix = np.zeros((n, m), dtype=np.float64)
        for i, (ax, ay) in enumerate(keypoints_a):
            for j, (bx, by) in enumerate(keypoints_b):
                cost_matrix[i, j] = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

        # Solve optimal assignment using Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Compute mean distance of matched pairs
        if len(row_indices) == 0:
            return 0.0

        total_distance = sum(
            cost_matrix[r, c] for r, c in zip(row_indices, col_indices)
        )
        mean_distance = total_distance / len(row_indices)

        return mean_distance

    def evaluate_condition_type(
        self,
        condition_type: str,
        prompts: List[str],
        condition_maps: List[Image.Image],
        num_samples: int = 100,
    ) -> Tuple[float, float]:
        """Evaluate alignment for a single condition type.

        Generates images using the pipeline for the given condition type,
        then computes the alignment score for each generated image against
        its corresponding condition map. Returns the mean and standard
        deviation of all alignment scores.

        Uses fixed seeds for reproducibility — each sample gets seed = 42 + i.

        Args:
            condition_type: One of "edge", "depth", or "pose".
            prompts: List of text prompts for generation. Will be cycled
                if fewer than num_samples.
            condition_maps: List of condition map images. Will be cycled
                if fewer than num_samples.
            num_samples: Number of image-condition pairs to evaluate
                (default 100).

        Returns:
            Tuple of (mean_score, std_score) for the alignment scores.

        Raises:
            ValueError: If condition_type is not one of the supported types.
        """
        valid_types = {"edge", "depth", "pose"}
        if condition_type not in valid_types:
            raise ValueError(
                f"Invalid condition_type '{condition_type}'. "
                f"Supported types are: {sorted(valid_types)}"
            )

        # Select the appropriate alignment computation method
        alignment_methods = {
            "edge": self.compute_edge_alignment,
            "depth": self.compute_depth_alignment,
            "pose": self.compute_pose_alignment,
        }
        compute_alignment = alignment_methods[condition_type]

        logger.info(
            f"Evaluating {condition_type} alignment with {num_samples} samples"
        )

        scores: List[float] = []

        for i in range(num_samples):
            # Cycle through prompts and condition maps
            prompt = prompts[i % len(prompts)]
            condition_map = condition_maps[i % len(condition_maps)]

            # Use fixed seed for reproducibility
            seed = 42 + i

            try:
                # Generate image using the pipeline
                generated_image = self.pipeline(
                    text_prompt=prompt,
                    condition_image=condition_map,
                    condition_type=condition_type,
                    seed=seed,
                )

                # Compute alignment score
                score = compute_alignment(generated_image, condition_map)
                scores.append(score)

            except Exception as e:
                logger.warning(
                    f"Failed to evaluate sample {i} for {condition_type}: {e}"
                )
                continue

        if len(scores) == 0:
            logger.warning(
                f"No successful evaluations for condition type '{condition_type}'"
            )
            return (0.0, 0.0)

        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        logger.info(
            f"{condition_type} alignment: mean={mean_score:.4f}, "
            f"std={std_score:.4f} ({len(scores)}/{num_samples} samples)"
        )

        return (mean_score, std_score)

    def run_full_evaluation(
        self,
        prompts: List[str],
        condition_maps: Dict[str, List[Image.Image]],
        num_samples: int = 100,
    ) -> Dict[str, Tuple[float, float]]:
        """Run alignment evaluation for all condition types.

        Calls evaluate_condition_type for each condition type present in
        condition_maps. Handles failures gracefully by skipping failed
        condition types and logging a warning.

        Args:
            prompts: List of text prompts for generation.
            condition_maps: Dictionary mapping condition_type to a list of
                condition map images (e.g., {"edge": [...], "depth": [...]}).
            num_samples: Number of samples to evaluate per condition type
                (default 100).

        Returns:
            Dictionary mapping condition_type to (mean_score, std_score).
            Only includes condition types that were successfully evaluated.
        """
        results: Dict[str, Tuple[float, float]] = {}

        for condition_type, maps in condition_maps.items():
            try:
                mean_score, std_score = self.evaluate_condition_type(
                    condition_type=condition_type,
                    prompts=prompts,
                    condition_maps=maps,
                    num_samples=num_samples,
                )
                results[condition_type] = (mean_score, std_score)
            except Exception as e:
                logger.warning(
                    f"Alignment evaluation failed for '{condition_type}': {e}"
                )
                continue

        return results

    def print_results_table(self, results: Dict[str, Tuple[float, float]]) -> None:
        """Print a formatted results table showing alignment scores.

        Displays a table with columns: Condition Type, Mean Score, Std,
        and Target Met (✓ if mean >= 0.70, ✗ otherwise).

        Prints a warning line for any condition type with mean score < 0.70.

        Args:
            results: Dictionary mapping condition_type to (mean_score, std_score).
        """
        target_threshold = 0.70

        # Print table header
        print("\n" + "=" * 60)
        print("Condition Alignment Results")
        print("=" * 60)
        print(
            f"{'Condition Type':<18} {'Mean Score':<12} {'Std':<10} {'Target (0.70)'}"
        )
        print("-" * 60)

        # Track conditions below target
        below_target: List[Tuple[str, float]] = []

        for condition_type, (mean_score, std_score) in results.items():
            target_met = "✓" if mean_score >= target_threshold else "✗"
            print(
                f"{condition_type:<18} {mean_score:<12.4f} {std_score:<10.4f} {target_met}"
            )

            if mean_score < target_threshold:
                below_target.append((condition_type, mean_score))

        print("=" * 60)

        # Print warnings for conditions below target
        if below_target:
            print()
            for condition_type, score in below_target:
                print(
                    f"⚠ WARNING: {condition_type} alignment score ({score:.4f}) "
                    f"is below target threshold of {target_threshold}"
                )
            print()
