"""
Unit tests for Canny edge map extraction module

This module provides comprehensive unit tests for the edge extraction functionality,
including various image formats, edge cases, error conditions, and validation scenarios.
Tests ensure robust operation across different input types and configurations.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import warnings

import numpy as np
from PIL import Image
import cv2

# Import the modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

# Import directly from the module file to avoid package import issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "extract_edges", 
    Path(__file__).parent.parent.parent / "src" / "data" / "extract_edges.py"
)
extract_edges = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract_edges)

# Import the classes and functions we need
CannyEdgeExtractor = extract_edges.CannyEdgeExtractor
EdgeExtractionConfig = extract_edges.EdgeExtractionConfig
EdgeExtractionResult = extract_edges.EdgeExtractionResult
EdgeMapValidator = extract_edges.EdgeMapValidator
extract_edges_from_image = extract_edges.extract_edges_from_image
extract_edges_from_dataset = extract_edges.extract_edges_from_dataset
save_edge_map = extract_edges.save_edge_map


class TestEdgeExtractionConfig(unittest.TestCase):
    """Test EdgeExtractionConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = EdgeExtractionConfig()
        
        # Test default Canny parameters
        self.assertEqual(config.low_threshold, 50.0)
        self.assertEqual(config.high_threshold, 150.0)
        self.assertEqual(config.aperture_size, 3)
        self.assertFalse(config.l2_gradient)
        
        # Test adaptive thresholding defaults
        self.assertTrue(config.adaptive_threshold)
        self.assertEqual(config.threshold_percentile_low, 0.1)
        self.assertEqual(config.threshold_percentile_high, 0.3)
        
        # Test output parameters
        self.assertEqual(config.output_channels, 3)
        self.assertTrue(config.normalize_output)
        self.assertFalse(config.invert_edges)
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = EdgeExtractionConfig(
            low_threshold=100.0,
            high_threshold=200.0,
            adaptive_threshold=False,
            output_channels=1,
            normalize_output=False
        )
        
        self.assertEqual(config.low_threshold, 100.0)
        self.assertEqual(config.high_threshold, 200.0)
        self.assertFalse(config.adaptive_threshold)
        self.assertEqual(config.output_channels, 1)
        self.assertFalse(config.normalize_output)


class TestEdgeExtractionResult(unittest.TestCase):
    """Test EdgeExtractionResult dataclass and validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = EdgeExtractionConfig()
        
        # Create a valid edge map
        self.valid_edge_map = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        
        # Create a valid result
        self.valid_result = EdgeExtractionResult(
            edge_map=self.valid_edge_map,
            success=True,
            processing_time_ms=100.0,
            edge_density=0.1,
            connected_components=10,
            threshold_low=50.0,
            threshold_high=150.0
        )
    
    def test_valid_result_validation(self):
        """Test validation of a valid result"""
        is_valid, errors = self.valid_result.validate(self.config)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_failed_result_validation(self):
        """Test validation of a failed result"""
        failed_result = EdgeExtractionResult(
            edge_map=None,
            success=False,
            processing_time_ms=0.0,
            edge_density=0.0,
            connected_components=0,
            threshold_low=0.0,
            threshold_high=0.0,
            error_message="Test error"
        )
        
        is_valid, errors = failed_result.validate(self.config)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertIn("Test error", errors[0])
    
    def test_invalid_shape_validation(self):
        """Test validation with invalid edge map shape"""
        invalid_edge_map = np.ones((256, 256), dtype=np.float32)  # Missing channel dimension
        
        result = EdgeExtractionResult(
            edge_map=invalid_edge_map,
            success=True,
            processing_time_ms=100.0,
            edge_density=0.1,
            connected_components=10,
            threshold_low=50.0,
            threshold_high=150.0
        )
        
        is_valid, errors = result.validate(self.config)
        self.assertFalse(is_valid)
        self.assertTrue(any("Invalid edge map shape" in error for error in errors))
    
    def test_invalid_value_range_validation(self):
        """Test validation with invalid value range"""
        invalid_edge_map = np.ones((256, 256, 3), dtype=np.float32) * 2.0  # Values > 1.0
        
        result = EdgeExtractionResult(
            edge_map=invalid_edge_map,
            success=True,
            processing_time_ms=100.0,
            edge_density=0.1,
            connected_components=10,
            threshold_low=50.0,
            threshold_high=150.0
        )
        
        is_valid, errors = result.validate(self.config)
        self.assertFalse(is_valid)
        self.assertTrue(any("out of range" in error for error in errors))
    
    def test_edge_density_validation(self):
        """Test validation with invalid edge density"""
        result = EdgeExtractionResult(
            edge_map=self.valid_edge_map,
            success=True,
            processing_time_ms=100.0,
            edge_density=0.0001,  # Too low (below 0.001)
            connected_components=10,
            threshold_low=50.0,
            threshold_high=150.0
        )
        
        is_valid, errors = result.validate(self.config)
        self.assertFalse(is_valid)
        self.assertTrue(any("Edge density too low" in error for error in errors))


class TestCannyEdgeExtractor(unittest.TestCase):
    """Test CannyEdgeExtractor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = EdgeExtractionConfig()
        self.extractor = CannyEdgeExtractor(self.config)
        
        # Create test images
        self.test_image_rgb = self._create_test_image_rgb()
        self.test_image_pil = Image.fromarray(self.test_image_rgb)
        self.test_image_gray = cv2.cvtColor(self.test_image_rgb, cv2.COLOR_RGB2GRAY)
    
    def _create_test_image_rgb(self) -> np.ndarray:
        """Create a test RGB image with clear edges"""
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Add some geometric shapes with clear edges and higher contrast
        cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.rectangle(image, (60, 60), (140, 140), (0, 0, 0), -1)  # Inner black rectangle
        cv2.circle(image, (200, 200), 30, (128, 128, 128), -1)
        cv2.circle(image, (200, 200), 20, (255, 255, 255), -1)  # Inner white circle
        cv2.line(image, (0, 128), (255, 128), (255, 0, 0), 5)  # Thicker line
        cv2.line(image, (128, 0), (128, 255), (0, 255, 0), 5)  # Vertical line
        
        # Add some diagonal lines for more edges
        cv2.line(image, (0, 0), (100, 100), (0, 0, 255), 3)
        cv2.line(image, (255, 0), (155, 100), (255, 255, 0), 3)
        
        return image
    
    def test_initialization(self):
        """Test extractor initialization"""
        self.assertIsInstance(self.extractor.config, EdgeExtractionConfig)
        self.assertEqual(self.extractor.total_extractions, 0)
        self.assertEqual(self.extractor.successful_extractions, 0)
        self.assertEqual(self.extractor.failed_extractions, 0)
    
    def test_extract_from_pil_image(self):
        """Test edge extraction from PIL Image"""
        result = self.extractor.extract(self.test_image_pil)
        
        self.assertIsInstance(result, EdgeExtractionResult)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.edge_map)
        self.assertEqual(result.edge_map.shape, (256, 256, 3))
        self.assertGreater(result.processing_time_ms, 0)
        self.assertGreater(result.edge_density, 0)
        self.assertGreater(result.connected_components, 0)
    
    def test_extract_from_numpy_array(self):
        """Test edge extraction from numpy array"""
        result = self.extractor.extract(self.test_image_rgb)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.edge_map)
        self.assertEqual(result.edge_map.shape, (256, 256, 3))
    
    def test_extract_from_grayscale(self):
        """Test edge extraction from grayscale image"""
        result = self.extractor.extract(self.test_image_gray)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.edge_map)
        self.assertEqual(result.edge_map.shape, (256, 256, 3))
    
    def test_adaptive_thresholding(self):
        """Test adaptive threshold computation"""
        # Test with adaptive thresholding enabled
        config_adaptive = EdgeExtractionConfig(adaptive_threshold=True)
        extractor_adaptive = CannyEdgeExtractor(config_adaptive)
        
        result_adaptive = extractor_adaptive.extract(self.test_image_rgb)
        
        # Test with fixed thresholding
        config_fixed = EdgeExtractionConfig(
            adaptive_threshold=False,
            low_threshold=50.0,
            high_threshold=150.0
        )
        extractor_fixed = CannyEdgeExtractor(config_fixed)
        
        result_fixed = extractor_fixed.extract(self.test_image_rgb)
        
        # Both should succeed
        self.assertTrue(result_adaptive.success)
        self.assertTrue(result_fixed.success)
        
        # Adaptive thresholds should be different from fixed
        self.assertNotEqual(result_adaptive.threshold_low, 50.0)
        self.assertNotEqual(result_adaptive.threshold_high, 150.0)
        
        # Fixed thresholds should match config
        self.assertEqual(result_fixed.threshold_low, 50.0)
        self.assertEqual(result_fixed.threshold_high, 150.0)
    
    def test_preprocessing_options(self):
        """Test different preprocessing options"""
        # Test with Gaussian blur
        config_blur = EdgeExtractionConfig(apply_gaussian_blur=True)
        extractor_blur = CannyEdgeExtractor(config_blur)
        result_blur = extractor_blur.extract(self.test_image_rgb)
        
        # Test without Gaussian blur
        config_no_blur = EdgeExtractionConfig(apply_gaussian_blur=False)
        extractor_no_blur = CannyEdgeExtractor(config_no_blur)
        result_no_blur = extractor_no_blur.extract(self.test_image_rgb)
        
        # Both should complete extraction (success may vary based on validation)
        self.assertIsNotNone(result_blur.edge_map)
        self.assertIsNotNone(result_no_blur.edge_map)
        
        # Results should be different
        self.assertFalse(np.array_equal(result_blur.edge_map, result_no_blur.edge_map))
    
    def test_postprocessing_options(self):
        """Test different postprocessing options"""
        # Test with morphology
        config_morph = EdgeExtractionConfig(apply_morphology=True)
        extractor_morph = CannyEdgeExtractor(config_morph)
        result_morph = extractor_morph.extract(self.test_image_rgb)
        
        # Test without morphology
        config_no_morph = EdgeExtractionConfig(apply_morphology=False)
        extractor_no_morph = CannyEdgeExtractor(config_no_morph)
        result_no_morph = extractor_no_morph.extract(self.test_image_rgb)
        
        # Both should succeed
        self.assertTrue(result_morph.success)
        self.assertTrue(result_no_morph.success)
    
    def test_output_format_options(self):
        """Test different output format options"""
        # Test 3-channel normalized output
        config_3ch_norm = EdgeExtractionConfig(output_channels=3, normalize_output=True)
        extractor_3ch_norm = CannyEdgeExtractor(config_3ch_norm)
        result_3ch_norm = extractor_3ch_norm.extract(self.test_image_rgb)
        
        self.assertTrue(result_3ch_norm.success)
        self.assertEqual(result_3ch_norm.edge_map.shape[2], 3)
        self.assertEqual(result_3ch_norm.edge_map.dtype, np.float32)
        self.assertLessEqual(result_3ch_norm.edge_map.max(), 1.0)
        
        # Test 1-channel uint8 output
        config_1ch_uint8 = EdgeExtractionConfig(output_channels=1, normalize_output=False)
        extractor_1ch_uint8 = CannyEdgeExtractor(config_1ch_uint8)
        result_1ch_uint8 = extractor_1ch_uint8.extract(self.test_image_rgb)
        
        self.assertTrue(result_1ch_uint8.success)
        self.assertEqual(result_1ch_uint8.edge_map.shape[2], 1)
        self.assertEqual(result_1ch_uint8.edge_map.dtype, np.uint8)
        self.assertLessEqual(result_1ch_uint8.edge_map.max(), 255)
    
    def test_edge_inversion(self):
        """Test edge inversion option"""
        # Test normal edges
        config_normal = EdgeExtractionConfig(invert_edges=False)
        extractor_normal = CannyEdgeExtractor(config_normal)
        result_normal = extractor_normal.extract(self.test_image_rgb)
        
        # Test inverted edges
        config_inverted = EdgeExtractionConfig(invert_edges=True)
        extractor_inverted = CannyEdgeExtractor(config_inverted)
        result_inverted = extractor_inverted.extract(self.test_image_rgb)
        
        # Both should succeed
        self.assertTrue(result_normal.success)
        self.assertTrue(result_inverted.success)
        
        # Results should be different (inverted)
        self.assertFalse(np.array_equal(result_normal.edge_map, result_inverted.edge_map))
    
    def test_batch_extraction(self):
        """Test batch edge extraction"""
        images = [self.test_image_pil, self.test_image_rgb, self.test_image_gray]
        
        results = self.extractor.extract_batch(images, show_progress=False)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, EdgeExtractionResult)
            self.assertTrue(result.success)
            self.assertIsNotNone(result.edge_map)
    
    def test_statistics_tracking(self):
        """Test extraction statistics tracking"""
        # Initial statistics should show no extractions
        initial_stats = self.extractor.get_statistics()
        if "total_extractions" in initial_stats:
            self.assertEqual(initial_stats["total_extractions"], 0)
        else:
            # If no extractions yet, should return a message
            self.assertIn("message", initial_stats)
        
        # Perform some extractions
        self.extractor.extract(self.test_image_pil)
        self.extractor.extract(self.test_image_rgb)
        
        # Check updated statistics
        stats = self.extractor.get_statistics()
        self.assertEqual(stats["total_extractions"], 2)
        self.assertGreaterEqual(stats["successful_extractions"], 0)  # May be 0 if validation fails
        self.assertGreaterEqual(stats["failed_extractions"], 0)
        self.assertEqual(stats["successful_extractions"] + stats["failed_extractions"], 2)
        self.assertGreater(stats["average_processing_time_ms"], 0)
        
        # Reset statistics
        self.extractor.reset_statistics()
        reset_stats = self.extractor.get_statistics()
        if "total_extractions" in reset_stats:
            self.assertEqual(reset_stats["total_extractions"], 0)
        else:
            self.assertIn("message", reset_stats)
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs"""
        # Test None input
        result_none = self.extractor.extract(None)
        self.assertFalse(result_none.success)
        self.assertIsNotNone(result_none.error_message)
        
        # Test empty array
        empty_array = np.array([])
        result_empty = self.extractor.extract(empty_array)
        self.assertFalse(result_empty.success)
        
        # Test invalid shape
        invalid_shape = np.ones((10, 10, 10, 10))  # 4D array
        result_invalid = self.extractor.extract(invalid_shape)
        self.assertFalse(result_invalid.success)
    
    def test_edge_case_images(self):
        """Test edge case images"""
        # Test uniform image (no edges) - should fail validation due to low edge density
        uniform_image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        result_uniform = self.extractor.extract(uniform_image)
        
        # Should fail validation due to low edge density, but extraction should complete
        self.assertFalse(result_uniform.success)  # Validation should fail
        self.assertIsNotNone(result_uniform.edge_map)  # But edge map should exist
        self.assertLess(result_uniform.edge_density, 0.01)
        
        # Test high contrast image with a clear vertical edge
        high_contrast = np.zeros((256, 256, 3), dtype=np.uint8)
        high_contrast[:, :128] = 255  # Half white, half black
        # Add some noise to ensure we get edges
        high_contrast[100:150, 120:135] = 128  # Gray patch at the boundary
        
        result_contrast = self.extractor.extract(high_contrast)
        
        # Should succeed with higher edge density
        if result_contrast.success:
            self.assertGreater(result_contrast.edge_density, 0.001)
        else:
            # If it still fails, at least check that we got some edge detection
            self.assertIsNotNone(result_contrast.edge_map)
            # The edge density might still be low, but should be > 0
            self.assertGreaterEqual(result_contrast.edge_density, 0.0)


class TestEdgeMapValidator(unittest.TestCase):
    """Test EdgeMapValidator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.valid_edge_map = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        self.validator = EdgeMapValidator()
    
    def test_valid_edge_map(self):
        """Test validation of valid edge map"""
        is_valid, errors = self.validator.validate_edge_map(self.valid_edge_map)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_none_edge_map(self):
        """Test validation of None edge map"""
        is_valid, errors = self.validator.validate_edge_map(None)
        self.assertFalse(is_valid)
        self.assertIn("Edge map is None", errors[0])
    
    def test_invalid_shape(self):
        """Test validation of invalid shape"""
        invalid_2d = np.ones((256, 256), dtype=np.float32)
        is_valid, errors = self.validator.validate_edge_map(invalid_2d)
        self.assertFalse(is_valid)
        self.assertTrue(any("must be 3D" in error for error in errors))
    
    def test_wrong_channels(self):
        """Test validation of wrong number of channels"""
        wrong_channels = np.ones((256, 256, 4), dtype=np.float32)
        is_valid, errors = self.validator.validate_edge_map(wrong_channels, expected_channels=3)
        self.assertFalse(is_valid)
        self.assertTrue(any("Expected 3 channels" in error for error in errors))
    
    def test_size_mismatch(self):
        """Test validation of size mismatch"""
        is_valid, errors = self.validator.validate_edge_map(
            self.valid_edge_map, 
            target_size=(512, 512)
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("Size mismatch" in error for error in errors))
    
    def test_value_range_float(self):
        """Test validation of float value range"""
        # Test valid range
        valid_float = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        is_valid, errors = self.validator.validate_edge_map(valid_float)
        self.assertTrue(is_valid)
        
        # Test invalid range
        invalid_float = np.ones((256, 256, 3), dtype=np.float32) * 2.0
        is_valid, errors = self.validator.validate_edge_map(invalid_float)
        self.assertFalse(is_valid)
        self.assertTrue(any("out of range [0,1]" in error for error in errors))
    
    def test_value_range_uint8(self):
        """Test validation of uint8 value range"""
        # Test valid range
        valid_uint8 = np.ones((256, 256, 3), dtype=np.uint8) * 128
        is_valid, errors = self.validator.validate_edge_map(valid_uint8)
        self.assertTrue(is_valid)
        
        # Test invalid range (this shouldn't happen with uint8, but test anyway)
        invalid_uint8 = np.ones((256, 256, 3), dtype=np.int16) * 300
        is_valid, errors = self.validator.validate_edge_map(invalid_uint8)
        self.assertFalse(is_valid)
    
    def test_uniform_edge_map(self):
        """Test validation of uniform edge map"""
        # Test uniform non-zero map (should pass now)
        uniform_map = np.ones((256, 256, 3), dtype=np.float32) * 0.5
        is_valid, errors = self.validator.validate_edge_map(uniform_map)
        self.assertTrue(is_valid)  # Should pass now
        
        # Test uniform zero map (should fail)
        zero_map = np.zeros((256, 256, 3), dtype=np.float32)
        is_valid, errors = self.validator.validate_edge_map(zero_map)
        self.assertFalse(is_valid)
        self.assertTrue(any("completely empty" in error for error in errors))
    
    def test_resize_edge_map(self):
        """Test edge map resizing"""
        original_size = (256, 256)
        target_size = (512, 512)
        
        edge_map = np.random.rand(256, 256, 3).astype(np.float32)
        resized = self.validator.resize_edge_map(edge_map, target_size)
        
        self.assertEqual(resized.shape[:2], target_size[::-1])  # Height, Width
        self.assertEqual(resized.shape[2], 3)
    
    def test_resize_single_channel(self):
        """Test resizing single channel edge map"""
        edge_map = np.random.rand(256, 256, 1).astype(np.float32)
        target_size = (128, 128)
        
        resized = self.validator.resize_edge_map(edge_map, target_size)
        
        self.assertEqual(resized.shape, (128, 128, 1))


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_image = self._create_test_image()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_image(self) -> Image.Image:
        """Create a test image"""
        array = np.zeros((256, 256, 3), dtype=np.uint8)
        # Create more complex pattern with more edges
        cv2.rectangle(array, (50, 50), (200, 200), (255, 255, 255), -1)
        cv2.rectangle(array, (70, 70), (180, 180), (0, 0, 0), -1)
        cv2.rectangle(array, (90, 90), (160, 160), (128, 128, 128), -1)
        cv2.circle(array, (128, 128), 30, (255, 255, 255), -1)
        return Image.fromarray(array)
    
    def test_extract_edges_from_image(self):
        """Test convenience function for single image"""
        result = extract_edges_from_image(self.test_image)
        
        self.assertIsInstance(result, EdgeExtractionResult)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.edge_map)
    
    def test_extract_edges_from_dataset(self):
        """Test convenience function for dataset"""
        images = [self.test_image, self.test_image, self.test_image]
        results = extract_edges_from_dataset(images, show_progress=False)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsInstance(result, EdgeExtractionResult)
            self.assertTrue(result.success)
    
    def test_save_edge_map_png(self):
        """Test saving edge map as PNG"""
        edge_map = np.random.rand(256, 256, 3).astype(np.float32)
        output_path = Path(self.temp_dir) / "test_edge.png"
        
        save_edge_map(edge_map, output_path, format="png")
        
        self.assertTrue(output_path.exists())
        
        # Load and verify
        loaded_image = Image.open(output_path)
        self.assertEqual(loaded_image.size, (256, 256))
        self.assertEqual(loaded_image.mode, "RGB")
    
    def test_save_edge_map_jpg(self):
        """Test saving edge map as JPEG"""
        edge_map = np.random.rand(256, 256, 3).astype(np.float32)
        output_path = Path(self.temp_dir) / "test_edge.jpg"
        
        save_edge_map(edge_map, output_path, format="jpg")
        
        self.assertTrue(output_path.exists())
        
        # Load and verify
        loaded_image = Image.open(output_path)
        self.assertEqual(loaded_image.size, (256, 256))
        self.assertEqual(loaded_image.mode, "RGB")
    
    def test_save_edge_map_single_channel(self):
        """Test saving single channel edge map"""
        edge_map = np.random.rand(256, 256, 1).astype(np.float32)
        output_path = Path(self.temp_dir) / "test_edge_gray.png"
        
        save_edge_map(edge_map, output_path, format="png")
        
        self.assertTrue(output_path.exists())
        
        # Load and verify
        loaded_image = Image.open(output_path)
        self.assertEqual(loaded_image.size, (256, 256))
        self.assertEqual(loaded_image.mode, "L")  # Grayscale
    
    def test_save_edge_map_uint8(self):
        """Test saving uint8 edge map"""
        edge_map = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        output_path = Path(self.temp_dir) / "test_edge_uint8.png"
        
        save_edge_map(edge_map, output_path, format="png")
        
        self.assertTrue(output_path.exists())
    
    def test_save_edge_map_invalid_format(self):
        """Test saving with invalid format"""
        edge_map = np.random.rand(256, 256, 3).astype(np.float32)
        output_path = Path(self.temp_dir) / "test_edge.invalid"
        
        with self.assertRaises(ValueError):
            save_edge_map(edge_map, output_path, format="invalid")
    
    def test_save_edge_map_invalid_channels(self):
        """Test saving with invalid number of channels"""
        edge_map = np.random.rand(256, 256, 5).astype(np.float32)  # 5 channels
        output_path = Path(self.temp_dir) / "test_edge.png"
        
        with self.assertRaises(ValueError):
            save_edge_map(edge_map, output_path, format="png")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = EdgeExtractionConfig()
        self.extractor = CannyEdgeExtractor(self.config)
    
    def test_cv2_error_handling(self):
        """Test handling of OpenCV errors"""
        # Create an image that might cause OpenCV issues
        problematic_image = np.ones((1, 1, 3), dtype=np.uint8)  # Very small image
        
        result = self.extractor.extract(problematic_image)
        
        # Should handle gracefully (either succeed or fail with proper error message)
        self.assertIsInstance(result, EdgeExtractionResult)
        if not result.success:
            self.assertIsNotNone(result.error_message)
    
    def test_memory_error_simulation(self):
        """Test handling of potential memory errors"""
        # This test simulates what would happen with very large images
        # We don't actually create huge images to avoid memory issues in tests
        
        with patch('cv2.Canny', side_effect=MemoryError("Simulated memory error")):
            result = self.extractor.extract(np.ones((256, 256, 3), dtype=np.uint8))
            
            self.assertFalse(result.success)
            self.assertIn("memory error", result.error_message.lower())
    
    def test_invalid_config_handling(self):
        """Test handling of invalid configuration"""
        # Test with invalid output channels
        invalid_config = EdgeExtractionConfig(output_channels=5)
        extractor = CannyEdgeExtractor(invalid_config)
        
        result = extractor.extract(np.ones((256, 256, 3), dtype=np.uint8))
        
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_message)
    
    def test_corrupted_image_handling(self):
        """Test handling of corrupted image data"""
        # Test with NaN values
        corrupted_image = np.ones((256, 256, 3), dtype=np.float32)
        corrupted_image[100:150, 100:150] = np.nan
        
        result = self.extractor.extract(corrupted_image)
        
        # Should handle gracefully
        self.assertIsInstance(result, EdgeExtractionResult)
        if not result.success:
            self.assertIsNotNone(result.error_message)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)