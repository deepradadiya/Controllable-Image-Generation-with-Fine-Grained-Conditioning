"""
FID Score Computation Module

This module implements Fréchet Inception Distance (FID) calculation for evaluating
the quality of generated images against real images. The implementation follows
the original FID paper methodology with optimizations for batch processing and
statistical significance testing.

Mathematical Background:
    The Fréchet Inception Distance (Heusel et al., 2017) measures the distance
    between two multivariate Gaussian distributions fitted to InceptionV3 features
    extracted from real and generated image sets.
    
    Given:
        - μ_r, Σ_r: mean and covariance of real image features
        - μ_g, Σ_g: mean and covariance of generated image features
    
    The FID is defined as:
        FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2·(Σ_r · Σ_g)^(1/2))
    
    where:
        - ||·||² is the squared Euclidean norm (measures mean shift)
        - Tr(·) is the matrix trace (measures covariance mismatch)
        - (·)^(1/2) is the matrix square root
    
    Interpretation:
        - FID = 0: Generated distribution is identical to real distribution
        - Lower FID: Better quality and diversity of generated images
        - Typical values: FID < 50 is good, FID < 10 is excellent
    
    Assumptions and Limitations:
        - Assumes features follow a multivariate Gaussian distribution
        - Sensitive to sample size (need ≥2048 samples for stable estimates)
        - Does not capture all aspects of image quality (e.g., artifacts)
        - InceptionV3 features may not perfectly represent human perception

Statistical Significance:
    We use bootstrap resampling to compute confidence intervals for the FID score.
    This accounts for the sampling variability inherent in estimating distribution
    parameters from finite samples.

Key Features:
- InceptionV3-based feature extraction (2048-dimensional feature space)
- Batch processing for large evaluation sets (memory-efficient)
- Statistical significance testing with bootstrap confidence intervals
- Memory-efficient processing for Colab T4 GPU constraints
- Comprehensive error handling and numerical stability safeguards

Requirements Validated: 5.1, 5.5
"""

import logging
import warnings
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import inception_v3
from PIL import Image
import scipy.stats
from tqdm import tqdm

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

logger = logging.getLogger(__name__)


@dataclass
class FIDResults:
    """Results from FID score computation with statistical analysis"""
    fid_score: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    num_real_samples: int
    num_generated_samples: int
    computation_time_seconds: float
    real_mean: np.ndarray
    real_cov: np.ndarray
    generated_mean: np.ndarray
    generated_cov: np.ndarray
    
    def __str__(self) -> str:
        return (f"FID Score: {self.fid_score:.3f} "
                f"(95% CI: [{self.confidence_interval[0]:.3f}, {self.confidence_interval[1]:.3f}])")


class ImageDataset(Dataset):
    """Dataset wrapper for loading images from various sources"""
    
    def __init__(self, 
                 images: Union[List[str], List[Path], List[Image.Image], List[np.ndarray]],
                 transform: Optional[transforms.Compose] = None):
        """
        Initialize dataset with images from various sources
        
        Args:
            images: List of image paths, PIL Images, or numpy arrays
            transform: Optional torchvision transforms to apply
        """
        self.images = images
        self.transform = transform or self._get_default_transform()
    
    def _get_default_transform(self) -> transforms.Compose:
        """Default preprocessing transform for InceptionV3"""
        return transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and preprocess image at given index"""
        image = self.images[idx]
        
        # Convert to PIL Image if necessary
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                image = Image.fromarray(image)
            else:
                # Assume float array in [0, 1] range
                image = Image.fromarray((image * 255).astype(np.uint8))
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        return self.transform(image)


class InceptionV3FeatureExtractor(nn.Module):
    """
    InceptionV3 model for feature extraction (up to final pooling layer).
    
    Extracts 2048-dimensional feature vectors from images using a pre-trained
    InceptionV3 network with the final classification layer removed. These
    features capture high-level semantic content that correlates with human
    perception of image quality and diversity.
    
    Architecture:
        The InceptionV3 network processes images through a series of inception
        modules that capture features at multiple scales. We extract features
        after the final average pooling layer (before the FC classifier), which
        produces a 2048-dimensional vector per image.
    
    Why InceptionV3:
        - Standard choice for FID computation (ensures comparability with literature)
        - Pre-trained on ImageNet (captures general visual features)
        - 2048-dim features provide sufficient capacity for distribution modeling
        - Well-studied statistical properties for Gaussian assumption
    
    Input Requirements:
        - Images must be resized to 299×299 pixels
        - Normalized with ImageNet mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize InceptionV3 feature extractor
        
        Args:
            device: Device to run the model on
        """
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained InceptionV3
        self.inception = inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = nn.Identity()  # Remove final classification layer
        self.inception.eval()
        self.inception.to(self.device)
        
        # Freeze all parameters
        for param in self.inception.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images
        
        Args:
            x: Input tensor of shape (batch_size, 3, 299, 299)
            
        Returns:
            Feature tensor of shape (batch_size, 2048)
        """
        with torch.no_grad():
            # InceptionV3 forward pass up to final pooling
            x = self.inception.Conv2d_1a_3x3(x)
            x = self.inception.Conv2d_2a_3x3(x)
            x = self.inception.Conv2d_2b_3x3(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            x = self.inception.Conv2d_3b_1x1(x)
            x = self.inception.Conv2d_4a_3x3(x)
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            
            x = self.inception.Mixed_5b(x)
            x = self.inception.Mixed_5c(x)
            x = self.inception.Mixed_5d(x)
            x = self.inception.Mixed_6a(x)
            x = self.inception.Mixed_6b(x)
            x = self.inception.Mixed_6c(x)
            x = self.inception.Mixed_6d(x)
            x = self.inception.Mixed_6e(x)
            x = self.inception.Mixed_7a(x)
            x = self.inception.Mixed_7b(x)
            x = self.inception.Mixed_7c(x)
            
            # Global average pooling
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
            
            return x


class FIDCalculator:
    """
    Fréchet Inception Distance calculator with batch processing and statistical analysis.
    
    This implementation follows the standard FID computation pipeline:
        1. Extract 2048-dim features from InceptionV3's penultimate layer
        2. Fit multivariate Gaussians to real and generated feature distributions
        3. Compute the Fréchet distance between the two Gaussians
        4. Optionally compute bootstrap confidence intervals
    
    Statistical Methodology:
        The FID assumes that InceptionV3 features follow a multivariate Gaussian
        distribution. While this is an approximation, it works well in practice
        because:
        - The Central Limit Theorem applies to averaged pooled features
        - The 2048-dim feature space is sufficiently high-dimensional
        - Empirical validation shows strong correlation with human judgment
    
    Confidence Intervals:
        We use non-parametric bootstrap resampling to estimate the uncertainty
        in the FID score. This involves:
        1. Repeatedly resampling (with replacement) from both feature sets
        2. Computing FID for each bootstrap sample
        3. Taking percentiles of the bootstrap distribution as CI bounds
        
        This approach makes no distributional assumptions about the FID estimator
        itself and accounts for both sampling variability and estimation error.
    
    Memory Considerations:
        - Features are extracted in batches to limit GPU memory usage
        - Covariance matrices are computed in NumPy (CPU) to avoid GPU memory pressure
        - Bootstrap sampling operates on pre-extracted features (no re-extraction)
    """
    
    def __init__(self, 
                 batch_size: int = 32,
                 device: Optional[torch.device] = None,
                 num_workers: int = 4):
        """
        Initialize FID calculator
        
        Args:
            batch_size: Batch size for feature extraction
            device: Device to run computations on
            num_workers: Number of workers for data loading
        """
        self.batch_size = batch_size
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = num_workers
        
        # Initialize feature extractor
        self.feature_extractor = InceptionV3FeatureExtractor(self.device)
        
        logger.info(f"FID Calculator initialized on device: {self.device}")
    
    def extract_features(self, 
                        images: Union[List[str], List[Path], List[Image.Image], List[np.ndarray]],
                        show_progress: bool = True) -> np.ndarray:
        """
        Extract InceptionV3 features from a list of images
        
        Args:
            images: List of images (paths, PIL Images, or numpy arrays)
            show_progress: Whether to show progress bar
            
        Returns:
            Feature array of shape (num_images, 2048)
        """
        if len(images) == 0:
            raise ValueError("No images provided for feature extraction")
        
        # Create dataset and dataloader
        dataset = ImageDataset(images)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        features = []
        
        # Extract features in batches
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting features", disable=not show_progress):
                batch = batch.to(self.device)
                batch_features = self.feature_extractor(batch)
                features.append(batch_features.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def compute_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mean and covariance matrix from features
        
        Args:
            features: Feature array of shape (num_samples, feature_dim)
            
        Returns:
            Tuple of (mean, covariance_matrix)
        """
        if features.shape[0] < 2:
            raise ValueError("Need at least 2 samples to compute covariance")
        
        mean = np.mean(features, axis=0)
        cov = np.cov(features, rowvar=False)
        
        return mean, cov
    
    def calculate_frechet_distance(self, 
                                 mu1: np.ndarray, 
                                 sigma1: np.ndarray,
                                 mu2: np.ndarray, 
                                 sigma2: np.ndarray,
                                 eps: float = 1e-6) -> float:
        """
        Calculate Fréchet distance between two multivariate Gaussians.
        
        The Fréchet distance (also called Wasserstein-2 distance for Gaussians)
        measures the distance between two probability distributions. For
        multivariate Gaussians N(μ₁, Σ₁) and N(μ₂, Σ₂), it has a closed form:
        
            FID = ||μ₁ - μ₂||² + Tr(Σ₁ + Σ₂ - 2·(Σ₁·Σ₂)^(1/2))
        
        The first term measures the shift in means (how different the "average"
        generated image is from the "average" real image in feature space).
        
        The second term measures the mismatch in covariance structure (how
        different the diversity and correlations are between the two sets).
        
        Numerical Stability:
            - We add eps·I to both covariance matrices to ensure positive definiteness
            - Complex values in the matrix square root (from numerical errors) are
              handled by taking the real part
            - A fallback eigenvalue decomposition is used if scipy.linalg.sqrtm fails
        
        Args:
            mu1: Mean of first (real) distribution, shape (2048,).
            sigma1: Covariance of first distribution, shape (2048, 2048).
            mu2: Mean of second (generated) distribution, shape (2048,).
            sigma2: Covariance of second distribution, shape (2048, 2048).
            eps: Small value added to diagonal for numerical stability.
            
        Returns:
            FID score (non-negative float, lower is better).
        """
        # Ensure inputs are numpy arrays
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        
        # Term 1: Squared Euclidean distance between means
        # ||μ₁ - μ₂||² = Σᵢ (μ₁ᵢ - μ₂ᵢ)²
        diff = mu1 - mu2
        mean_diff = np.sum(diff ** 2)
        
        # Term 2: Trace of covariance mismatch
        # Tr(Σ₁ + Σ₂ - 2·(Σ₁·Σ₂)^(1/2))
        # Add epsilon to diagonal for numerical stability (ensures positive definiteness)
        sigma1 = sigma1 + eps * np.eye(sigma1.shape[0])
        sigma2 = sigma2 + eps * np.eye(sigma2.shape[0])
        
        try:
            # Compute (Σ₁·Σ₂)^(1/2) using scipy's matrix square root
            # For positive definite matrices, this should yield a real result
            product = sigma1.dot(sigma2)
            sqrt_product = scipy.linalg.sqrtm(product)
            
            # Handle complex numbers from numerical precision issues
            # For truly positive definite matrices, imaginary parts should be ~0
            if np.iscomplexobj(sqrt_product):
                if not np.allclose(np.diagonal(sqrt_product).imag, 0, atol=1e-3):
                    logger.warning("Imaginary component in matrix square root, taking real part")
                sqrt_product = sqrt_product.real
            
            # Compute trace term: Tr(Σ₁ + Σ₂ - 2·sqrt(Σ₁·Σ₂))
            trace_term = np.trace(sigma1 + sigma2 - 2 * sqrt_product)
            
        except Exception as e:
            logger.error(f"Error in matrix square root computation: {e}")
            # Fallback: use eigenvalue decomposition
            # Tr(sqrt(A)) = Σᵢ sqrt(λᵢ) where λᵢ are eigenvalues of A
            # This is more numerically stable for ill-conditioned matrices
            try:
                eigenvals = scipy.linalg.eigvals(product)
                sqrt_eigenvals = np.sqrt(np.maximum(eigenvals.real, 0))
                trace_term = np.trace(sigma1 + sigma2) - 2 * np.sum(sqrt_eigenvals)
            except Exception as e2:
                logger.error(f"Fallback computation also failed: {e2}")
                raise ValueError("Unable to compute matrix square root for FID calculation")
        
        # Final FID = mean_shift + covariance_mismatch
        # Both terms are non-negative, so FID ≥ 0 with equality iff distributions match
        fid = mean_diff + trace_term
        
        return float(fid)
    
    def bootstrap_fid_confidence_interval(self,
                                        real_features: np.ndarray,
                                        generated_features: np.ndarray,
                                        confidence_level: float = 0.95,
                                        num_bootstrap: int = 1000,
                                        random_seed: Optional[int] = 42) -> Tuple[float, float]:
        """
        Calculate confidence interval for FID score using bootstrap sampling.
        
        Bootstrap resampling provides a non-parametric estimate of the uncertainty
        in the FID score. This is important because:
        1. FID is estimated from finite samples (not the true distribution)
        2. The estimator's distribution is unknown (no closed-form CI)
        3. Small sample sizes can lead to highly variable FID estimates
        
        Method:
            For each of num_bootstrap iterations:
            1. Resample N_real features with replacement from real features
            2. Resample N_gen features with replacement from generated features
            3. Compute FID on the resampled sets
            4. Collect all bootstrap FID values
            5. Take the α/2 and 1-α/2 percentiles as CI bounds
        
        Interpretation:
            A 95% CI of [45.2, 52.8] means: if we repeated the experiment many
            times with different random samples from the same distributions,
            95% of the computed FID scores would fall in this range.
        
        Args:
            real_features: Features from real images, shape (N_real, 2048).
            generated_features: Features from generated images, shape (N_gen, 2048).
            confidence_level: Confidence level (e.g., 0.95 for 95% CI).
            num_bootstrap: Number of bootstrap iterations (1000 is standard).
            random_seed: Random seed for reproducibility.
            
        Returns:
            Tuple of (lower_bound, upper_bound) for the confidence interval.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        n_real = real_features.shape[0]
        n_generated = generated_features.shape[0]
        
        bootstrap_fids = []
        
        for _ in range(num_bootstrap):
            # Bootstrap sample from both distributions
            real_indices = np.random.choice(n_real, size=n_real, replace=True)
            generated_indices = np.random.choice(n_generated, size=n_generated, replace=True)
            
            real_sample = real_features[real_indices]
            generated_sample = generated_features[generated_indices]
            
            # Compute statistics for bootstrap sample
            real_mean, real_cov = self.compute_statistics(real_sample)
            generated_mean, generated_cov = self.compute_statistics(generated_sample)
            
            # Calculate FID for bootstrap sample
            bootstrap_fid = self.calculate_frechet_distance(
                real_mean, real_cov, generated_mean, generated_cov
            )
            bootstrap_fids.append(bootstrap_fid)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_fids, lower_percentile)
        upper_bound = np.percentile(bootstrap_fids, upper_percentile)
        
        return lower_bound, upper_bound
    
    def compute_fid(self,
                   real_images: Union[List[str], List[Path], List[Image.Image], List[np.ndarray]],
                   generated_images: Union[List[str], List[Path], List[Image.Image], List[np.ndarray]],
                   compute_confidence_interval: bool = True,
                   confidence_level: float = 0.95,
                   num_bootstrap: int = 1000,
                   show_progress: bool = True) -> FIDResults:
        """
        Compute FID score between real and generated images
        
        Args:
            real_images: List of real images
            generated_images: List of generated images
            compute_confidence_interval: Whether to compute confidence interval
            confidence_level: Confidence level for interval (default: 0.95)
            num_bootstrap: Number of bootstrap samples for CI
            show_progress: Whether to show progress bars
            
        Returns:
            FIDResults object containing FID score and statistics
        """
        start_time = time.time()
        
        # Validate inputs
        if len(real_images) == 0:
            raise ValueError("No real images provided")
        if len(generated_images) == 0:
            raise ValueError("No generated images provided")
        
        logger.info(f"Computing FID between {len(real_images)} real and {len(generated_images)} generated images")
        
        # Extract features
        if show_progress:
            print("Extracting features from real images...")
        real_features = self.extract_features(real_images, show_progress=show_progress)
        
        if show_progress:
            print("Extracting features from generated images...")
        generated_features = self.extract_features(generated_images, show_progress=show_progress)
        
        # Compute statistics
        real_mean, real_cov = self.compute_statistics(real_features)
        generated_mean, generated_cov = self.compute_statistics(generated_features)
        
        # Calculate FID score
        fid_score = self.calculate_frechet_distance(
            real_mean, real_cov, generated_mean, generated_cov
        )
        
        # Compute confidence interval if requested
        confidence_interval = (0.0, 0.0)
        if compute_confidence_interval:
            if show_progress:
                print("Computing confidence interval...")
            confidence_interval = self.bootstrap_fid_confidence_interval(
                real_features, generated_features, confidence_level, num_bootstrap
            )
        
        computation_time = time.time() - start_time
        
        results = FIDResults(
            fid_score=fid_score,
            confidence_interval=confidence_interval,
            confidence_level=confidence_level,
            num_real_samples=len(real_images),
            num_generated_samples=len(generated_images),
            computation_time_seconds=computation_time,
            real_mean=real_mean,
            real_cov=real_cov,
            generated_mean=generated_mean,
            generated_cov=generated_cov
        )
        
        logger.info(f"FID computation completed in {computation_time:.2f}s: {results}")
        
        return results


def compute_fid_from_paths(real_image_dir: Union[str, Path],
                          generated_image_dir: Union[str, Path],
                          batch_size: int = 32,
                          device: Optional[torch.device] = None,
                          compute_confidence_interval: bool = True,
                          confidence_level: float = 0.95,
                          num_bootstrap: int = 1000,
                          image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) -> FIDResults:
    """
    Convenience function to compute FID from image directories
    
    Args:
        real_image_dir: Directory containing real images
        generated_image_dir: Directory containing generated images
        batch_size: Batch size for processing
        device: Device to run on
        compute_confidence_interval: Whether to compute confidence interval
        confidence_level: Confidence level for interval
        num_bootstrap: Number of bootstrap samples
        image_extensions: Valid image file extensions
        
    Returns:
        FIDResults object
    """
    real_dir = Path(real_image_dir)
    generated_dir = Path(generated_image_dir)
    
    # Collect image paths
    real_images = []
    for ext in image_extensions:
        real_images.extend(real_dir.glob(f'*{ext}'))
        real_images.extend(real_dir.glob(f'*{ext.upper()}'))
    
    generated_images = []
    for ext in image_extensions:
        generated_images.extend(generated_dir.glob(f'*{ext}'))
        generated_images.extend(generated_dir.glob(f'*{ext.upper()}'))
    
    if not real_images:
        raise ValueError(f"No images found in real image directory: {real_dir}")
    if not generated_images:
        raise ValueError(f"No images found in generated image directory: {generated_dir}")
    
    # Convert to strings for compatibility
    real_images = [str(p) for p in real_images]
    generated_images = [str(p) for p in generated_images]
    
    # Initialize calculator and compute FID
    calculator = FIDCalculator(batch_size=batch_size, device=device)
    
    return calculator.compute_fid(
        real_images=real_images,
        generated_images=generated_images,
        compute_confidence_interval=compute_confidence_interval,
        confidence_level=confidence_level,
        num_bootstrap=num_bootstrap
    )


# Example usage and testing functions
def main():
    """Example usage of the FID calculator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute FID score between real and generated images")
    parser.add_argument("--real_dir", type=str, required=True, help="Directory with real images")
    parser.add_argument("--generated_dir", type=str, required=True, help="Directory with generated images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--no_ci", action="store_true", help="Skip confidence interval computation")
    parser.add_argument("--confidence_level", type=float, default=0.95, help="Confidence level")
    parser.add_argument("--num_bootstrap", type=int, default=1000, help="Number of bootstrap samples")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Compute FID
    results = compute_fid_from_paths(
        real_image_dir=args.real_dir,
        generated_image_dir=args.generated_dir,
        batch_size=args.batch_size,
        compute_confidence_interval=not args.no_ci,
        confidence_level=args.confidence_level,
        num_bootstrap=args.num_bootstrap
    )
    
    print(f"\nFID Results:")
    print(f"FID Score: {results.fid_score:.3f}")
    if results.confidence_interval != (0.0, 0.0):
        print(f"95% Confidence Interval: [{results.confidence_interval[0]:.3f}, {results.confidence_interval[1]:.3f}]")
    print(f"Real samples: {results.num_real_samples}")
    print(f"Generated samples: {results.num_generated_samples}")
    print(f"Computation time: {results.computation_time_seconds:.2f}s")


if __name__ == "__main__":
    main()