"""
Condition Alignment Metrics Module

This module measures how well generated images follow the input condition maps
for spatial conditioning evaluation. It supports depth, pose, and edge condition
types and provides quantitative metrics for condition following accuracy.

Key Features:
- Depth alignment: Compares depth maps extracted from generated images vs input depth maps
- Pose alignment: Compares detected poses in generated images vs input pose skeletons
- Edge alignment: Compares edge maps extracted from generated images vs input edge maps
- Quantitative metrics: SSIM, MSE, Pearson correlation, F1 score for condition adherence
- Batch evaluation with statistical aggregation
- Memory-efficient processing for Colab T4 GPU constraints

Requirements Validated: 5.2
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class AlignmentResult:
    """Result of condition alignment evaluation for a single image pair."""

    condition_type: str
    ssim: float
    mse: float
    pearson_correlation: float
    f1_score: float
    overall_score: float
    processing_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"AlignmentResult(type={self.condition_type}, "
            f"overall={self.overall_score:.4f}, ssim={self.ssim:.4f}, "
            f"mse={self.mse:.6f}, corr={self.pearson_correlation:.4f}, "
            f"f1={self.f1_score:.4f})"
        )


@dataclass
class BatchAlignmentResult:
    """Aggregated alignment results for a batch of image pairs."""

    condition_type: str
    mean_ssim: float
    mean_mse: float
    mean_pearson_correlation: float
    mean_f1_score: float
    mean_overall_score: float
    std_overall_score: float
    num_samples: int
    total_processing_time_seconds: float
    individual_results: List[AlignmentResult] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"BatchAlignmentResult(type={self.condition_type}, "
            f"overall={self.mean_overall_score:.4f} +/- {self.std_overall_score:.4f}, "
            f"n={self.num_samples})"
        )


def _compute_ssim(
    image1: np.ndarray,
    image2: np.ndarray,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
    data_range: float = 1.0,
) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.

    Implements the SSIM formula from Wang et al. (2004):
    SSIM(x,y) = (2*mu_x*mu_y + C1)(2*sigma_xy + C2) /
                ((mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2))

    Args:
        image1: First image as numpy array, values in [0, 1].
        image2: Second image as numpy array, values in [0, 1].
        window_size: Size of the Gaussian window for local statistics.
        k1: Stability constant for luminance.
        k2: Stability constant for contrast.
        data_range: Dynamic range of the images.

    Returns:
        SSIM value in range [-1, 1], where 1 means identical images.
    """
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # Create Gaussian kernel for local statistics
    kernel_size = window_size
    sigma = 1.5
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    window = kernel @ kernel.T

    # Compute local means
    mu1 = cv2.filter2D(image1, -1, window)
    mu2 = cv2.filter2D(image2, -1, window)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute local variances and covariance
    sigma1_sq = cv2.filter2D(image1 ** 2, -1, window) - mu1_sq
    sigma2_sq = cv2.filter2D(image2 ** 2, -1, window) - mu2_sq
    sigma12 = cv2.filter2D(image1 * image2, -1, window) - mu1_mu2

    # Compute SSIM map
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / (denominator + 1e-10)

    return float(np.mean(ssim_map))


def _compute_mse(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute Mean Squared Error between two images.

    Args:
        image1: First image as numpy array.
        image2: Second image as numpy array.

    Returns:
        MSE value (lower is better, 0 means identical).
    """
    return float(np.mean((image1 - image2) ** 2))


def _compute_pearson_correlation(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient between two images.

    Args:
        image1: First image as numpy array.
        image2: Second image as numpy array.

    Returns:
        Pearson correlation in range [-1, 1], where 1 means perfect positive correlation.
    """
    flat1 = image1.flatten()
    flat2 = image2.flatten()

    # Handle constant images (zero variance)
    std1 = np.std(flat1)
    std2 = np.std(flat2)

    if std1 < 1e-10 or std2 < 1e-10:
        # If both are constant and equal, correlation is 1; otherwise 0
        if std1 < 1e-10 and std2 < 1e-10:
            if np.allclose(flat1, flat2):
                return 1.0
            return 0.0
        return 0.0

    correlation = np.corrcoef(flat1, flat2)[0, 1]

    # Handle NaN from numerical issues
    if np.isnan(correlation):
        return 0.0

    return float(correlation)


def _compute_f1_score(
    image1: np.ndarray, image2: np.ndarray, threshold: float = 0.5
) -> float:
    """
    Compute F1 score for binary condition maps (edges, pose skeletons).

    Treats condition maps as binary masks and computes precision, recall, and F1.

    Args:
        image1: Predicted/extracted condition map.
        image2: Ground truth condition map.
        threshold: Threshold for binarization.

    Returns:
        F1 score in range [0, 1].
    """
    # Binarize both images
    binary1 = (image1 > threshold).astype(np.float32)
    binary2 = (image2 > threshold).astype(np.float32)

    # Compute true positives, false positives, false negatives
    tp = np.sum(binary1 * binary2)
    fp = np.sum(binary1 * (1 - binary2))
    fn = np.sum((1 - binary1) * binary2)

    # Compute precision and recall
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)

    # Compute F1
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return float(f1)


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.

    Args:
        image: Input image array.

    Returns:
        Normalized image in [0, 1] range as float32.
    """
    image = image.astype(np.float32)

    if image.max() > 1.0:
        image = image / 255.0

    return np.clip(image, 0.0, 1.0)


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image to single-channel grayscale.

    Args:
        image: Input image (can be 2D or 3D).

    Returns:
        2D grayscale image.
    """
    if len(image.shape) == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    if image.shape[2] == 3:
        # Use standard luminance weights
        return (
            0.2989 * image[:, :, 0]
            + 0.5870 * image[:, :, 1]
            + 0.1140 * image[:, :, 2]
        )
    if image.shape[2] == 4:
        # RGBA - ignore alpha
        return (
            0.2989 * image[:, :, 0]
            + 0.5870 * image[:, :, 1]
            + 0.1140 * image[:, :, 2]
        )
    raise ValueError(f"Unsupported number of channels: {image.shape[2]}")


def _resize_to_match(
    image: np.ndarray, target_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Resize image to match target spatial dimensions.

    Args:
        image: Input image.
        target_shape: Target (height, width).

    Returns:
        Resized image.
    """
    target_h, target_w = target_shape
    current_h, current_w = image.shape[:2]

    if current_h == target_h and current_w == target_w:
        return image

    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


class DepthAlignmentEvaluator:
    """
    Evaluates alignment between input depth condition maps and depth maps
    extracted from generated images.

    Depth alignment measures how well the generated image preserves the spatial
    depth structure specified by the input condition map. Uses SSIM, MSE, and
    Pearson correlation as primary metrics since depth maps are continuous-valued.
    """

    def __init__(self, blur_kernel_size: int = 5):
        """
        Initialize depth alignment evaluator.

        Args:
            blur_kernel_size: Gaussian blur kernel size for noise reduction
                before comparison. Set to 0 to disable.
        """
        self.blur_kernel_size = blur_kernel_size

    def extract_depth_from_generated(self, generated_image: np.ndarray) -> np.ndarray:
        """
        Extract an approximate depth map from a generated image using
        gradient-based estimation.

        This provides a lightweight depth proxy without requiring a full
        depth estimation model. For production use, replace with DPT model
        inference.

        Args:
            generated_image: Generated image as numpy array (H, W, 3) in [0, 1].

        Returns:
            Estimated depth map as (H, W) array in [0, 1].
        """
        gray = _to_grayscale(generated_image)

        # Convert to uint8 for OpenCV Laplacian compatibility
        gray_uint8 = (gray * 255).astype(np.uint8)

        # Use Laplacian as a proxy for depth edges/structure
        # Higher frequency content correlates with closer objects
        laplacian = cv2.Laplacian(gray_uint8, cv2.CV_64F)
        depth_proxy = np.abs(laplacian)

        # Normalize to [0, 1]
        max_val = depth_proxy.max()
        if max_val > 0:
            depth_proxy = depth_proxy / max_val

        # Apply Gaussian blur for smoother depth estimation
        if self.blur_kernel_size > 0:
            k = self.blur_kernel_size
            if k % 2 == 0:
                k += 1
            depth_proxy = cv2.GaussianBlur(depth_proxy, (k, k), 0)
            # Re-normalize after blur
            max_val = depth_proxy.max()
            if max_val > 0:
                depth_proxy = depth_proxy / max_val

        return depth_proxy.astype(np.float32)

    def compute_alignment(
        self,
        generated_image: np.ndarray,
        condition_map: np.ndarray,
        use_extracted_depth: bool = True,
    ) -> AlignmentResult:
        """
        Compute depth alignment between a generated image and its condition map.

        Args:
            generated_image: Generated image (H, W, 3) in [0, 1] or [0, 255].
            condition_map: Input depth condition map (H, W) or (H, W, C) in [0, 1] or [0, 255].
            use_extracted_depth: If True, extract depth from generated image for comparison.
                If False, compare the generated image directly (grayscale) against condition.

        Returns:
            AlignmentResult with depth alignment metrics.
        """
        start_time = time.time()

        # Normalize inputs
        gen_img = _normalize_image(generated_image)
        cond_map = _normalize_image(condition_map)

        # Convert condition map to grayscale
        cond_gray = _to_grayscale(cond_map)

        # Get comparison image
        if use_extracted_depth:
            comparison = self.extract_depth_from_generated(gen_img)
        else:
            comparison = _to_grayscale(gen_img)

        # Resize to match if needed
        comparison = _resize_to_match(comparison, cond_gray.shape[:2])

        # Compute metrics
        ssim = _compute_ssim(comparison, cond_gray)
        mse = _compute_mse(comparison, cond_gray)
        correlation = _compute_pearson_correlation(comparison, cond_gray)
        # F1 is less meaningful for continuous depth maps, but we compute it
        # with a moderate threshold for structural similarity
        f1 = _compute_f1_score(comparison, cond_gray, threshold=0.3)

        # Overall score: weighted combination favoring correlation and SSIM
        # for depth maps (continuous-valued signals)
        overall = 0.35 * max(0, ssim) + 0.35 * max(0, correlation) + 0.15 * f1 + 0.15 * (1.0 - min(mse, 1.0))

        processing_time = (time.time() - start_time) * 1000

        return AlignmentResult(
            condition_type="depth",
            ssim=ssim,
            mse=mse,
            pearson_correlation=correlation,
            f1_score=f1,
            overall_score=overall,
            processing_time_ms=processing_time,
            metadata={"use_extracted_depth": use_extracted_depth},
        )


class PoseAlignmentEvaluator:
    """
    Evaluates alignment between input pose condition maps and poses detected
    in generated images.

    Pose alignment measures how well the generated image preserves the body
    pose structure specified by the input skeleton condition map. Uses F1 score
    and structural metrics since pose maps are sparse binary-like signals.
    """

    def __init__(
        self,
        dilation_kernel_size: int = 3,
        distance_tolerance: int = 5,
    ):
        """
        Initialize pose alignment evaluator.

        Args:
            dilation_kernel_size: Kernel size for dilating pose skeletons
                before comparison (accounts for slight spatial shifts).
            distance_tolerance: Pixel distance tolerance for matching
                pose keypoints/limbs.
        """
        self.dilation_kernel_size = dilation_kernel_size
        self.distance_tolerance = distance_tolerance

    def extract_pose_from_generated(self, generated_image: np.ndarray) -> np.ndarray:
        """
        Extract a pose-like structural map from a generated image.

        Uses edge detection and morphological operations to approximate
        body structure. For production use, replace with DWPose/OpenPose
        inference.

        Args:
            generated_image: Generated image (H, W, 3) in [0, 1].

        Returns:
            Pose-like structural map (H, W) in [0, 1].
        """
        # Convert to uint8 for OpenCV operations
        img_uint8 = (generated_image * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

        # Use Canny edges as structural proxy
        edges = cv2.Canny(gray, 50, 150)

        # Apply morphological operations to get skeleton-like structure
        kernel = cv2.getStructuringElement(
            cv2.MORPH_CROSS, (3, 3)
        )
        # Thin the edges
        skeleton = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        return skeleton.astype(np.float32) / 255.0

    def _dilate_map(self, binary_map: np.ndarray) -> np.ndarray:
        """
        Dilate a binary map to account for spatial tolerance.

        Args:
            binary_map: Binary map to dilate.

        Returns:
            Dilated binary map.
        """
        if self.dilation_kernel_size <= 0:
            return binary_map

        k = self.dilation_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(
            (binary_map * 255).astype(np.uint8), kernel, iterations=1
        )
        return dilated.astype(np.float32) / 255.0

    def compute_alignment(
        self,
        generated_image: np.ndarray,
        condition_map: np.ndarray,
        use_extracted_pose: bool = True,
    ) -> AlignmentResult:
        """
        Compute pose alignment between a generated image and its condition map.

        Args:
            generated_image: Generated image (H, W, 3) in [0, 1] or [0, 255].
            condition_map: Input pose condition map (H, W) or (H, W, C) in [0, 1] or [0, 255].
            use_extracted_pose: If True, extract pose from generated image.
                If False, compare grayscale generated image against condition.

        Returns:
            AlignmentResult with pose alignment metrics.
        """
        start_time = time.time()

        # Normalize inputs
        gen_img = _normalize_image(generated_image)
        cond_map = _normalize_image(condition_map)

        # Convert condition map to grayscale
        cond_gray = _to_grayscale(cond_map)

        # Get comparison map
        if use_extracted_pose:
            comparison = self.extract_pose_from_generated(gen_img)
        else:
            comparison = _to_grayscale(gen_img)

        # Resize to match
        comparison = _resize_to_match(comparison, cond_gray.shape[:2])

        # Dilate both maps for tolerance-aware comparison
        cond_dilated = self._dilate_map(cond_gray)
        comp_dilated = self._dilate_map(comparison)

        # Compute metrics using dilated maps for F1 (tolerance-aware)
        f1 = _compute_f1_score(comp_dilated, cond_dilated, threshold=0.3)

        # Compute standard metrics on original maps
        ssim = _compute_ssim(comparison, cond_gray)
        mse = _compute_mse(comparison, cond_gray)
        correlation = _compute_pearson_correlation(comparison, cond_gray)

        # Overall score: weighted combination favoring F1 and structural
        # metrics for sparse pose maps
        overall = 0.40 * f1 + 0.25 * max(0, ssim) + 0.20 * max(0, correlation) + 0.15 * (1.0 - min(mse, 1.0))

        processing_time = (time.time() - start_time) * 1000

        return AlignmentResult(
            condition_type="pose",
            ssim=ssim,
            mse=mse,
            pearson_correlation=correlation,
            f1_score=f1,
            overall_score=overall,
            processing_time_ms=processing_time,
            metadata={
                "use_extracted_pose": use_extracted_pose,
                "dilation_kernel_size": self.dilation_kernel_size,
                "distance_tolerance": self.distance_tolerance,
            },
        )


class EdgeAlignmentEvaluator:
    """
    Evaluates alignment between input edge condition maps and edges detected
    in generated images.

    Edge alignment measures how well the generated image preserves the edge
    structure specified by the input Canny edge condition map. Uses F1 score
    with spatial tolerance and SSIM for structural comparison.
    """

    def __init__(
        self,
        canny_low: float = 50.0,
        canny_high: float = 150.0,
        dilation_kernel_size: int = 3,
    ):
        """
        Initialize edge alignment evaluator.

        Args:
            canny_low: Low threshold for Canny edge detection on generated images.
            canny_high: High threshold for Canny edge detection on generated images.
            dilation_kernel_size: Kernel size for dilating edge maps before
                comparison (accounts for slight spatial shifts).
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.dilation_kernel_size = dilation_kernel_size

    def extract_edges_from_generated(self, generated_image: np.ndarray) -> np.ndarray:
        """
        Extract edge map from a generated image using Canny edge detection.

        Args:
            generated_image: Generated image (H, W, 3) in [0, 1].

        Returns:
            Edge map (H, W) in [0, 1].
        """
        # Convert to uint8 for OpenCV
        img_uint8 = (generated_image * 255).astype(np.uint8)

        # Convert to grayscale
        if len(img_uint8.shape) == 3:
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_uint8

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        return edges.astype(np.float32) / 255.0

    def _dilate_map(self, binary_map: np.ndarray) -> np.ndarray:
        """
        Dilate a binary map to account for spatial tolerance.

        Args:
            binary_map: Binary map to dilate.

        Returns:
            Dilated binary map.
        """
        if self.dilation_kernel_size <= 0:
            return binary_map

        k = self.dilation_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(
            (binary_map * 255).astype(np.uint8), kernel, iterations=1
        )
        return dilated.astype(np.float32) / 255.0

    def compute_alignment(
        self,
        generated_image: np.ndarray,
        condition_map: np.ndarray,
        use_extracted_edges: bool = True,
    ) -> AlignmentResult:
        """
        Compute edge alignment between a generated image and its condition map.

        Args:
            generated_image: Generated image (H, W, 3) in [0, 1] or [0, 255].
            condition_map: Input edge condition map (H, W) or (H, W, C) in [0, 1] or [0, 255].
            use_extracted_edges: If True, extract edges from generated image.
                If False, compare grayscale generated image against condition.

        Returns:
            AlignmentResult with edge alignment metrics.
        """
        start_time = time.time()

        # Normalize inputs
        gen_img = _normalize_image(generated_image)
        cond_map = _normalize_image(condition_map)

        # Convert condition map to grayscale
        cond_gray = _to_grayscale(cond_map)

        # Get comparison map
        if use_extracted_edges:
            comparison = self.extract_edges_from_generated(gen_img)
        else:
            comparison = _to_grayscale(gen_img)

        # Resize to match
        comparison = _resize_to_match(comparison, cond_gray.shape[:2])

        # Dilate both maps for tolerance-aware F1 computation
        cond_dilated = self._dilate_map(cond_gray)
        comp_dilated = self._dilate_map(comparison)

        # Compute tolerance-aware F1
        f1 = _compute_f1_score(comp_dilated, cond_dilated, threshold=0.3)

        # Compute standard metrics
        ssim = _compute_ssim(comparison, cond_gray)
        mse = _compute_mse(comparison, cond_gray)
        correlation = _compute_pearson_correlation(comparison, cond_gray)

        # Overall score: weighted combination favoring F1 for binary edge maps
        overall = 0.40 * f1 + 0.25 * max(0, ssim) + 0.20 * max(0, correlation) + 0.15 * (1.0 - min(mse, 1.0))

        processing_time = (time.time() - start_time) * 1000

        return AlignmentResult(
            condition_type="edge",
            ssim=ssim,
            mse=mse,
            pearson_correlation=correlation,
            f1_score=f1,
            overall_score=overall,
            processing_time_ms=processing_time,
            metadata={
                "use_extracted_edges": use_extracted_edges,
                "canny_low": self.canny_low,
                "canny_high": self.canny_high,
                "dilation_kernel_size": self.dilation_kernel_size,
            },
        )


class ConditionAlignmentEvaluator:
    """
    Unified condition alignment evaluator that supports all condition types
    (depth, pose, edge) through a single interface.

    This class provides the compute_condition_alignment method specified in the
    design document's EvaluationMetrics interface. It delegates to the appropriate
    specialized evaluator based on the condition type.

    Usage:
        evaluator = ConditionAlignmentEvaluator()

        # Single image evaluation
        result = evaluator.compute_alignment(
            generated_image=gen_img,
            condition_map=cond_map,
            condition_type="depth"
        )

        # Batch evaluation
        batch_result = evaluator.compute_batch_alignment(
            generated_images=[img1, img2, img3],
            condition_maps=[map1, map2, map3],
            condition_type="edge"
        )

        # Simplified interface matching design doc
        score = evaluator.compute_condition_alignment(
            generated_images=[img1, img2],
            condition_maps=[map1, map2],
            condition_type="depth"
        )
    """

    # Supported condition types
    SUPPORTED_TYPES = ("depth", "pose", "edge")

    def __init__(
        self,
        depth_blur_kernel: int = 5,
        pose_dilation_kernel: int = 3,
        pose_distance_tolerance: int = 5,
        edge_canny_low: float = 50.0,
        edge_canny_high: float = 150.0,
        edge_dilation_kernel: int = 3,
    ):
        """
        Initialize the unified condition alignment evaluator.

        Args:
            depth_blur_kernel: Gaussian blur kernel size for depth evaluation.
            pose_dilation_kernel: Dilation kernel for pose tolerance.
            pose_distance_tolerance: Pixel distance tolerance for pose matching.
            edge_canny_low: Low threshold for Canny edge detection.
            edge_canny_high: High threshold for Canny edge detection.
            edge_dilation_kernel: Dilation kernel for edge tolerance.
        """
        self._evaluators = {
            "depth": DepthAlignmentEvaluator(blur_kernel_size=depth_blur_kernel),
            "pose": PoseAlignmentEvaluator(
                dilation_kernel_size=pose_dilation_kernel,
                distance_tolerance=pose_distance_tolerance,
            ),
            "edge": EdgeAlignmentEvaluator(
                canny_low=edge_canny_low,
                canny_high=edge_canny_high,
                dilation_kernel_size=edge_dilation_kernel,
            ),
        }
        logger.info(
            "ConditionAlignmentEvaluator initialized with support for: %s",
            ", ".join(self.SUPPORTED_TYPES),
        )

    def _get_evaluator(self, condition_type: str):
        """
        Get the appropriate evaluator for the given condition type.

        Args:
            condition_type: One of 'depth', 'pose', or 'edge'.

        Returns:
            The specialized evaluator instance.

        Raises:
            ValueError: If condition_type is not supported.
        """
        condition_type = condition_type.lower().strip()
        if condition_type not in self._evaluators:
            raise ValueError(
                f"Unsupported condition type: '{condition_type}'. "
                f"Supported types: {self.SUPPORTED_TYPES}"
            )
        return self._evaluators[condition_type]

    def _prepare_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Convert input to numpy array in expected format.

        Args:
            image: Input image as numpy array or PIL Image.

        Returns:
            Numpy array in float32 format.
        """
        if isinstance(image, Image.Image):
            return np.array(image).astype(np.float32) / 255.0
        return _normalize_image(image)

    def compute_alignment(
        self,
        generated_image: Union[np.ndarray, Image.Image],
        condition_map: Union[np.ndarray, Image.Image],
        condition_type: str,
    ) -> AlignmentResult:
        """
        Compute condition alignment for a single generated image and condition map pair.

        Args:
            generated_image: Generated image as numpy array (H, W, 3) or PIL Image.
            condition_map: Input condition map as numpy array or PIL Image.
            condition_type: Type of condition - 'depth', 'pose', or 'edge'.

        Returns:
            AlignmentResult with all computed metrics.

        Raises:
            ValueError: If condition_type is not supported.
        """
        evaluator = self._get_evaluator(condition_type)
        gen_img = self._prepare_image(generated_image)
        cond_map = self._prepare_image(condition_map)

        return evaluator.compute_alignment(gen_img, cond_map)

    def compute_batch_alignment(
        self,
        generated_images: List[Union[np.ndarray, Image.Image]],
        condition_maps: List[Union[np.ndarray, Image.Image]],
        condition_type: str,
    ) -> BatchAlignmentResult:
        """
        Compute condition alignment for a batch of image-condition pairs.

        Provides aggregated statistics (mean, std) across all pairs in the batch.

        Args:
            generated_images: List of generated images.
            condition_maps: List of corresponding condition maps.
            condition_type: Type of condition - 'depth', 'pose', or 'edge'.

        Returns:
            BatchAlignmentResult with aggregated metrics and individual results.

        Raises:
            ValueError: If lists have different lengths or condition_type is unsupported.
        """
        if len(generated_images) != len(condition_maps):
            raise ValueError(
                f"Number of generated images ({len(generated_images)}) must match "
                f"number of condition maps ({len(condition_maps)})"
            )

        if len(generated_images) == 0:
            raise ValueError("At least one image-condition pair is required")

        start_time = time.time()
        results: List[AlignmentResult] = []

        for i, (gen_img, cond_map) in enumerate(
            zip(generated_images, condition_maps)
        ):
            try:
                result = self.compute_alignment(gen_img, cond_map, condition_type)
                results.append(result)
            except Exception as e:
                logger.warning(
                    "Failed to compute alignment for sample %d: %s", i, str(e)
                )
                continue

        if len(results) == 0:
            raise RuntimeError("All alignment computations failed")

        # Aggregate metrics
        ssim_values = [r.ssim for r in results]
        mse_values = [r.mse for r in results]
        corr_values = [r.pearson_correlation for r in results]
        f1_values = [r.f1_score for r in results]
        overall_values = [r.overall_score for r in results]

        total_time = time.time() - start_time

        return BatchAlignmentResult(
            condition_type=condition_type,
            mean_ssim=float(np.mean(ssim_values)),
            mean_mse=float(np.mean(mse_values)),
            mean_pearson_correlation=float(np.mean(corr_values)),
            mean_f1_score=float(np.mean(f1_values)),
            mean_overall_score=float(np.mean(overall_values)),
            std_overall_score=float(np.std(overall_values)),
            num_samples=len(results),
            total_processing_time_seconds=total_time,
            individual_results=results,
        )

    def compute_condition_alignment(
        self,
        generated_images: List[Union[np.ndarray, Image.Image]],
        condition_maps: List[Union[np.ndarray, Image.Image]],
        condition_type: str = "depth",
    ) -> float:
        """
        Compute overall condition alignment score for a batch of images.

        This is the simplified interface matching the design document's
        EvaluationMetrics.compute_condition_alignment specification.

        Args:
            generated_images: List of generated images (numpy arrays or PIL Images).
            condition_maps: List of corresponding condition maps.
            condition_type: Type of condition - 'depth', 'pose', or 'edge'.
                Defaults to 'depth'.

        Returns:
            Float score representing mean condition alignment (0 to 1, higher is better).

        Raises:
            ValueError: If inputs are invalid or condition_type is unsupported.
        """
        batch_result = self.compute_batch_alignment(
            generated_images, condition_maps, condition_type
        )
        return batch_result.mean_overall_score
