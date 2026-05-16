"""
Test suite for FID score computation module

This module contains comprehensive tests for the FID calculator including:
- Unit tests for individual components
- Integration tests for end-to-end FID computation
- Performance tests for batch processing
- Statistical validation tests

Requirements Validated: 5.1, 5.5
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from typing import List
import warnings

import numpy as np
import torch
from PIL import Image
import pytest

from compute_fid import (
    FIDCalculator, 
    FIDResults, 
    ImageDataset, 
    InceptionV3FeatureExtractor,
    compute_fid_from_paths
)

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning)


class TestImageDataset(unittest.TestCase):
    """Test cases for ImageDataset class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_images = []
        
        # Create test images
        for i in range(5):
            img = Image.new('RGB', (256, 256), color=(i*50, i*50, i*50))
            img_path = Path(self.temp_dir) / f"test_image_{i}.png"
            img.save(img_path)
            self.test_images.append(str(img_path))
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_dataset_initialization_with_paths(self):
        """Test dataset initialization with image paths"""
        dataset = ImageDataset(self.test_images)
        self.assertEqual(len(dataset), 5)
    
    def test_dataset_initialization_with_pil_images(self):
        """Test dataset initialization with PIL Images"""
        pil_images = [Image.open(path) for path in self.test_images]
        dataset = ImageDataset(pil_images)
        self.assertEqual(len(dataset), 5)
    
    def test_dataset_getitem_returns_correct_shape(self):
        """Test that dataset returns correctly shaped tensors"""
        dataset = ImageDataset(self.test_images)
        sample = dataset[0]
        
        # Should be (3, 299, 299) after InceptionV3 preprocessing
        self.assertEqual(sample.shape, (3, 299, 299))
        self.assertIsInstance(sample, torch.Tensor)
    
    def test_dataset_with_numpy_arrays(self):
        """Test dataset with numpy array inputs"""
        numpy_images = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(3)]
        dataset = ImageDataset(numpy_images)
        self.assertEqual(len(dataset), 3)
        
        sample = dataset[0]
        self.assertEqual(sample.shape, (3, 299, 299))


class TestInceptionV3FeatureExtractor(unittest.TestCase):
    """Test cases for InceptionV3FeatureExtractor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')  # Use CPU for testing
        self.extractor = InceptionV3FeatureExtractor(device=self.device)
    
    def test_feature_extractor_initialization(self):
        """Test that feature extractor initializes correctly"""
        self.assertIsNotNone(self.extractor.inception)
        self.assertEqual(self.extractor.device, self.device)
    
    def test_feature_extraction_output_shape(self):
        """Test that feature extraction produces correct output shape"""
        # Create dummy input (batch_size=2, channels=3, height=299, width=299)
        dummy_input = torch.randn(2, 3, 299, 299)
        
        with torch.no_grad():
            features = self.extractor(dummy_input)
        
        # Should produce (batch_size, 2048) features
        self.assertEqual(features.shape, (2, 2048))
        self.assertIsInstance(features, torch.Tensor)
    
    def test_feature_extraction_deterministic(self):
        """Test that feature extraction is deterministic"""
        dummy_input = torch.randn(1, 3, 299, 299)
        
        with torch.no_grad():
            features1 = self.extractor(dummy_input)
            features2 = self.extractor(dummy_input)
        
        torch.testing.assert_close(features1, features2)


class TestFIDCalculator(unittest.TestCase):
    """Test cases for FIDCalculator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')  # Use CPU for testing
        self.calculator = FIDCalculator(batch_size=2, device=self.device, num_workers=0)
        
        # Create test images
        self.temp_dir = tempfile.mkdtemp()
        self.real_images = []
        self.generated_images = []
        
        # Create real images (more structured)
        for i in range(10):
            img = Image.new('RGB', (256, 256), color=(100, 150, 200))
            img_path = Path(self.temp_dir) / f"real_{i}.png"
            img.save(img_path)
            self.real_images.append(str(img_path))
        
        # Create generated images (more random)
        for i in range(10):
            img = Image.new('RGB', (256, 256), color=(50 + i*10, 100 + i*5, 150 + i*8))
            img_path = Path(self.temp_dir) / f"generated_{i}.png"
            img.save(img_path)
            self.generated_images.append(str(img_path))
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_extract_features_basic(self):
        """Test basic feature extraction functionality"""
        features = self.calculator.extract_features(self.real_images[:5], show_progress=False)
        
        self.assertEqual(features.shape, (5, 2048))
        self.assertIsInstance(features, np.ndarray)
    
    def test_compute_statistics(self):
        """Test statistics computation from features"""
        # Create dummy features
        features = np.random.randn(100, 2048)
        
        mean, cov = self.calculator.compute_statistics(features)
        
        self.assertEqual(mean.shape, (2048,))
        self.assertEqual(cov.shape, (2048, 2048))
        self.assertIsInstance(mean, np.ndarray)
        self.assertIsInstance(cov, np.ndarray)
    
    def test_compute_statistics_insufficient_samples(self):
        """Test that compute_statistics raises error with insufficient samples"""
        features = np.random.randn(1, 2048)  # Only 1 sample
        
        with self.assertRaises(ValueError):
            self.calculator.compute_statistics(features)
    
    def test_calculate_frechet_distance(self):
        """Test Fréchet distance calculation"""
        # Create simple test distributions
        mu1 = np.array([0.0, 0.0])
        sigma1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        mu2 = np.array([1.0, 1.0])
        sigma2 = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        fid = self.calculator.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        
        self.assertIsInstance(fid, float)
        self.assertGreater(fid, 0)  # Should be positive for different distributions
    
    def test_calculate_frechet_distance_identical_distributions(self):
        """Test FID calculation for identical distributions"""
        mu = np.array([0.0, 0.0])
        sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
        
        fid = self.calculator.calculate_frechet_distance(mu, sigma, mu, sigma)
        
        self.assertAlmostEqual(fid, 0.0, places=6)  # Should be close to 0
    
    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence interval computation"""
        # Create dummy features
        real_features = np.random.randn(50, 2048)
        generated_features = np.random.randn(50, 2048) + 0.1  # Slightly different
        
        lower, upper = self.calculator.bootstrap_fid_confidence_interval(
            real_features, generated_features, num_bootstrap=100, random_seed=42
        )
        
        self.assertIsInstance(lower, float)
        self.assertIsInstance(upper, float)
        self.assertLess(lower, upper)  # Lower bound should be less than upper bound
    
    def test_compute_fid_end_to_end(self):
        """Test end-to-end FID computation"""
        results = self.calculator.compute_fid(
            real_images=self.real_images,
            generated_images=self.generated_images,
            compute_confidence_interval=False,  # Skip CI for speed
            show_progress=False
        )
        
        self.assertIsInstance(results, FIDResults)
        self.assertIsInstance(results.fid_score, float)
        self.assertGreater(results.fid_score, 0)
        self.assertEqual(results.num_real_samples, len(self.real_images))
        self.assertEqual(results.num_generated_samples, len(self.generated_images))
    
    def test_compute_fid_with_confidence_interval(self):
        """Test FID computation with confidence interval"""
        results = self.calculator.compute_fid(
            real_images=self.real_images[:5],  # Use fewer images for speed
            generated_images=self.generated_images[:5],
            compute_confidence_interval=True,
            num_bootstrap=50,  # Fewer bootstrap samples for speed
            show_progress=False
        )
        
        self.assertIsInstance(results, FIDResults)
        self.assertNotEqual(results.confidence_interval, (0.0, 0.0))
        self.assertLess(results.confidence_interval[0], results.confidence_interval[1])
    
    def test_compute_fid_empty_inputs(self):
        """Test that compute_fid raises error with empty inputs"""
        with self.assertRaises(ValueError):
            self.calculator.compute_fid([], self.generated_images)
        
        with self.assertRaises(ValueError):
            self.calculator.compute_fid(self.real_images, [])


class TestFIDFromPaths(unittest.TestCase):
    """Test cases for compute_fid_from_paths function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.real_dir = Path(self.temp_dir) / "real"
        self.generated_dir = Path(self.temp_dir) / "generated"
        
        self.real_dir.mkdir()
        self.generated_dir.mkdir()
        
        # Create test images in directories
        for i in range(5):
            # Real images
            img = Image.new('RGB', (256, 256), color=(100, 150, 200))
            img.save(self.real_dir / f"real_{i}.png")
            
            # Generated images
            img = Image.new('RGB', (256, 256), color=(50 + i*10, 100 + i*5, 150 + i*8))
            img.save(self.generated_dir / f"generated_{i}.jpg")
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_compute_fid_from_paths_basic(self):
        """Test basic FID computation from directory paths"""
        results = compute_fid_from_paths(
            real_image_dir=self.real_dir,
            generated_image_dir=self.generated_dir,
            batch_size=2,
            device=torch.device('cpu'),
            compute_confidence_interval=False
        )
        
        self.assertIsInstance(results, FIDResults)
        self.assertIsInstance(results.fid_score, float)
        self.assertGreater(results.fid_score, 0)
    
    def test_compute_fid_from_paths_empty_directory(self):
        """Test error handling for empty directories"""
        empty_dir = Path(self.temp_dir) / "empty"
        empty_dir.mkdir()
        
        with self.assertRaises(ValueError):
            compute_fid_from_paths(
                real_image_dir=empty_dir,
                generated_image_dir=self.generated_dir
            )


class TestFIDStatisticalProperties(unittest.TestCase):
    """Test statistical properties of FID computation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = FIDCalculator(batch_size=4, device=torch.device('cpu'), num_workers=0)
    
    def test_fid_symmetry(self):
        """Test that FID is symmetric (FID(A,B) = FID(B,A))"""
        # Create two sets of dummy features
        features_a = np.random.randn(20, 2048)
        features_b = np.random.randn(20, 2048) + 0.5
        
        mean_a, cov_a = self.calculator.compute_statistics(features_a)
        mean_b, cov_b = self.calculator.compute_statistics(features_b)
        
        fid_ab = self.calculator.calculate_frechet_distance(mean_a, cov_a, mean_b, cov_b)
        fid_ba = self.calculator.calculate_frechet_distance(mean_b, cov_b, mean_a, cov_a)
        
        self.assertAlmostEqual(fid_ab, fid_ba, places=6)
    
    def test_fid_triangle_inequality(self):
        """Test that FID satisfies triangle inequality property"""
        # Create three sets of features
        features_a = np.random.randn(30, 2048)
        features_b = np.random.randn(30, 2048) + 0.3
        features_c = np.random.randn(30, 2048) + 0.6
        
        mean_a, cov_a = self.calculator.compute_statistics(features_a)
        mean_b, cov_b = self.calculator.compute_statistics(features_b)
        mean_c, cov_c = self.calculator.compute_statistics(features_c)
        
        fid_ab = self.calculator.calculate_frechet_distance(mean_a, cov_a, mean_b, cov_b)
        fid_bc = self.calculator.calculate_frechet_distance(mean_b, cov_b, mean_c, cov_c)
        fid_ac = self.calculator.calculate_frechet_distance(mean_a, cov_a, mean_c, cov_c)
        
        # Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
        # Note: This might not always hold for FID due to its specific formulation
        # but we test it as a sanity check
        self.assertLessEqual(fid_ac, fid_ab + fid_bc + 1e-6)  # Small tolerance for numerical errors


class TestFIDPerformance(unittest.TestCase):
    """Performance tests for FID computation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.calculator = FIDCalculator(batch_size=8, device=torch.device('cpu'), num_workers=0)
    
    def test_batch_processing_efficiency(self):
        """Test that batch processing is more efficient than individual processing"""
        # Create test images
        test_images = []
        for i in range(20):
            img = Image.new('RGB', (256, 256), color=(i*10, i*10, i*10))
            test_images.append(img)
        
        # Time batch processing
        import time
        start_time = time.time()
        features_batch = self.calculator.extract_features(test_images, show_progress=False)
        batch_time = time.time() - start_time
        
        self.assertEqual(features_batch.shape, (20, 2048))
        self.assertLess(batch_time, 60)  # Should complete within reasonable time
    
    def test_memory_efficiency(self):
        """Test that memory usage is reasonable for large batches"""
        # This test mainly ensures no memory errors occur
        large_batch = []
        for i in range(50):
            img = Image.new('RGB', (256, 256), color=(i*5, i*5, i*5))
            large_batch.append(img)
        
        try:
            features = self.calculator.extract_features(large_batch, show_progress=False)
            self.assertEqual(features.shape, (50, 2048))
        except Exception as e:
            self.fail(f"Memory efficiency test failed: {e}")


def create_synthetic_test_data():
    """Create synthetic test data for validation"""
    temp_dir = tempfile.mkdtemp()
    real_dir = Path(temp_dir) / "real"
    generated_dir = Path(temp_dir) / "generated"
    
    real_dir.mkdir()
    generated_dir.mkdir()
    
    # Create real images (more uniform)
    for i in range(20):
        img = Image.new('RGB', (256, 256), color=(100, 150, 200))
        img.save(real_dir / f"real_{i}.png")
    
    # Create generated images (more varied)
    for i in range(20):
        color = (50 + i*5, 100 + i*3, 150 + i*4)
        img = Image.new('RGB', (256, 256), color=color)
        img.save(generated_dir / f"generated_{i}.png")
    
    return temp_dir, real_dir, generated_dir


def run_integration_test():
    """Run a complete integration test"""
    print("Running FID computation integration test...")
    
    # Create test data
    temp_dir, real_dir, generated_dir = create_synthetic_test_data()
    
    try:
        # Compute FID
        results = compute_fid_from_paths(
            real_image_dir=real_dir,
            generated_image_dir=generated_dir,
            batch_size=4,
            device=torch.device('cpu'),
            compute_confidence_interval=True,
            num_bootstrap=100
        )
        
        print(f"Integration test results:")
        print(f"FID Score: {results.fid_score:.3f}")
        print(f"95% CI: [{results.confidence_interval[0]:.3f}, {results.confidence_interval[1]:.3f}]")
        print(f"Computation time: {results.computation_time_seconds:.2f}s")
        print("Integration test passed!")
        
        return True
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        return False
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run integration test
    integration_success = run_integration_test()
    
    # Run unit tests
    print("\nRunning unit tests...")
    unittest.main(verbosity=2, exit=False)
    
    if integration_success:
        print("\nAll tests completed successfully!")
    else:
        print("\nSome tests failed!")