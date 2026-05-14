"""
Integration tests for edge extraction module

This module tests the integration of the edge extraction functionality
with the broader ControlNet training pipeline components.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

# Import the edge extraction module directly to avoid torch dependency
import sys
import importlib.util

# Load the extract_edges module directly
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
extract_edges_from_image = extract_edges.extract_edges_from_image
extract_edges_from_dataset = extract_edges.extract_edges_from_dataset
save_edge_map = extract_edges.save_edge_map


class TestEdgeExtractionIntegration(unittest.TestCase):
    """Integration tests for edge extraction pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "edge_outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create test images that simulate real dataset samples
        self.test_images = self._create_test_dataset()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_dataset(self):
        """Create a small dataset of test images"""
        images = []
        
        # Image 1: Simple geometric shapes with more edges
        img1 = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.rectangle(img1, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.rectangle(img1, (60, 60), (140, 140), (0, 0, 0), -1)  # Inner rectangle
        cv2.circle(img1, (200, 200), 30, (128, 128, 128), -1)
        cv2.circle(img1, (200, 200), 20, (255, 255, 255), -1)  # Inner circle
        cv2.line(img1, (0, 128), (255, 128), (255, 0, 0), 3)  # Horizontal line
        cv2.line(img1, (128, 0), (128, 255), (0, 255, 0), 3)  # Vertical line
        images.append(Image.fromarray(img1))
        
        # Image 2: More complex scene with many edges
        img2 = np.ones((256, 256, 3), dtype=np.uint8) * 240  # Light background
        cv2.rectangle(img2, (20, 100), (100, 200), (100, 100, 100), -1)  # Building
        cv2.rectangle(img2, (30, 110), (90, 150), (200, 200, 200), -1)   # Window
        cv2.rectangle(img2, (35, 115), (50, 130), (50, 50, 50), -1)      # Window frame
        cv2.rectangle(img2, (55, 115), (70, 130), (50, 50, 50), -1)      # Window frame
        cv2.circle(img2, (180, 150), 40, (34, 139, 34), -1)              # Tree
        cv2.rectangle(img2, (170, 190), (190, 230), (139, 69, 19), -1)   # Trunk
        # Add more details
        cv2.line(img2, (0, 230), (255, 230), (64, 64, 64), 5)  # Ground line
        cv2.rectangle(img2, (120, 180), (160, 220), (255, 0, 0), -1)     # Red object
        images.append(Image.fromarray(img2))
        
        # Image 3: High contrast edges with patterns
        img3 = np.zeros((256, 256, 3), dtype=np.uint8)
        img3[:, :128] = 255  # Half white, half black
        cv2.line(img3, (0, 128), (255, 128), (128, 128, 128), 5)  # Horizontal line
        cv2.line(img3, (128, 0), (128, 255), (128, 128, 128), 5)  # Vertical line
        # Add diagonal lines
        cv2.line(img3, (0, 0), (128, 128), (255, 0, 0), 3)
        cv2.line(img3, (128, 0), (255, 128), (0, 255, 0), 3)
        cv2.line(img3, (0, 128), (128, 255), (0, 0, 255), 3)
        cv2.line(img3, (128, 128), (255, 255), (255, 255, 0), 3)
        images.append(Image.fromarray(img3))
        
        # Image 4: Moderate contrast with clear structures
        img4 = np.ones((256, 256, 3), dtype=np.uint8) * 128  # Gray background
        cv2.rectangle(img4, (80, 80), (176, 176), (200, 200, 200), -1)  # Light square
        cv2.rectangle(img4, (90, 90), (166, 166), (60, 60, 60), -1)     # Dark inner square
        cv2.circle(img4, (128, 128), 30, (255, 255, 255), -1)           # White circle
        cv2.circle(img4, (128, 128), 20, (0, 0, 0), -1)                 # Black inner circle
        # Add corner details
        cv2.rectangle(img4, (10, 10), (50, 50), (255, 0, 0), -1)
        cv2.rectangle(img4, (206, 10), (246, 50), (0, 255, 0), -1)
        cv2.rectangle(img4, (10, 206), (50, 246), (0, 0, 255), -1)
        cv2.rectangle(img4, (206, 206), (246, 246), (255, 255, 0), -1)
        images.append(Image.fromarray(img4))
        
        return images
        
        return images
    
    def test_pipeline_compatibility(self):
        """Test that edge extraction works with typical pipeline data"""
        # Test with different image formats that might come from dataset processing
        
        # Use a more lenient configuration for integration testing
        config = EdgeExtractionConfig(
            adaptive_threshold=True,
            min_edge_density=0.0001,  # Very low threshold for integration tests
            min_connected_components=1
        )
        
        # PIL Image (most common)
        result_pil = extract_edges_from_image(self.test_images[0], config)
        self.assertIsInstance(result_pil, EdgeExtractionResult)
        self.assertTrue(result_pil.success)
        
        # Numpy array (RGB)
        img_array = np.array(self.test_images[1])
        result_numpy = extract_edges_from_image(img_array, config)
        self.assertIsInstance(result_numpy, EdgeExtractionResult)
        self.assertTrue(result_numpy.success)
        
        # Grayscale conversion
        img_gray = self.test_images[2].convert('L')
        result_gray = extract_edges_from_image(img_gray, config)
        self.assertIsInstance(result_gray, EdgeExtractionResult)
        self.assertTrue(result_gray.success)
    
    def test_batch_processing_pipeline(self):
        """Test batch processing as would be used in dataset preparation"""
        # Configure for production-like settings with relaxed validation
        config = EdgeExtractionConfig(
            adaptive_threshold=True,
            apply_gaussian_blur=True,
            apply_morphology=True,
            output_channels=3,  # RGB for ControlNet compatibility
            normalize_output=True,  # [0,1] range for neural networks
            min_edge_density=0.0001,  # Very relaxed for integration tests
            min_connected_components=1
        )
        
        # Process batch
        results = extract_edges_from_dataset(
            self.test_images, 
            config=config, 
            show_progress=False
        )
        
        # Verify results
        self.assertEqual(len(results), len(self.test_images))
        
        successful_results = [r for r in results if r.success]
        self.assertGreater(len(successful_results), 0, "At least some extractions should succeed")
        
        # Check output format compatibility
        for result in successful_results:
            self.assertEqual(result.edge_map.shape[2], 3)  # RGB channels
            self.assertEqual(result.edge_map.dtype, np.float32)  # Float type
            self.assertGreaterEqual(result.edge_map.min(), 0.0)  # Valid range
            self.assertLessEqual(result.edge_map.max(), 1.0)
    
    def test_controlnet_format_compatibility(self):
        """Test that output format is compatible with ControlNet requirements"""
        # Test different output configurations
        configs = [
            # RGB normalized (standard ControlNet format)
            EdgeExtractionConfig(output_channels=3, normalize_output=True),
            # Single channel normalized
            EdgeExtractionConfig(output_channels=1, normalize_output=True),
            # RGB uint8 (alternative format)
            EdgeExtractionConfig(output_channels=3, normalize_output=False),
        ]
        
        test_image = self.test_images[0]
        
        for i, config in enumerate(configs):
            with self.subTest(config_index=i):
                result = extract_edges_from_image(test_image, config)
                
                if result.success:
                    # Check shape
                    expected_channels = config.output_channels
                    self.assertEqual(result.edge_map.shape[2], expected_channels)
                    
                    # Check data type and range
                    if config.normalize_output:
                        self.assertEqual(result.edge_map.dtype, np.float32)
                        self.assertGreaterEqual(result.edge_map.min(), 0.0)
                        self.assertLessEqual(result.edge_map.max(), 1.0)
                    else:
                        self.assertEqual(result.edge_map.dtype, np.uint8)
                        self.assertGreaterEqual(result.edge_map.min(), 0)
                        self.assertLessEqual(result.edge_map.max(), 255)
    
    def test_file_io_integration(self):
        """Test file I/O operations for dataset storage"""
        # Use lenient config for integration testing
        config = EdgeExtractionConfig(min_edge_density=0.0001, min_connected_components=1)
        
        # Extract edges from test image
        result = extract_edges_from_image(self.test_images[0], config)
        self.assertTrue(result.success)
        
        # Test saving in different formats
        formats = ["png", "jpg", "tiff"]
        
        for fmt in formats:
            with self.subTest(format=fmt):
                output_path = self.output_dir / f"test_edge.{fmt}"
                
                # Save edge map
                save_edge_map(result.edge_map, output_path, format=fmt)
                
                # Verify file was created
                self.assertTrue(output_path.exists())
                
                # Verify file can be loaded back
                loaded_image = Image.open(output_path)
                self.assertIsNotNone(loaded_image)
                
                # Check dimensions match
                original_height, original_width = result.edge_map.shape[:2]
                self.assertEqual(loaded_image.size, (original_width, original_height))
    
    def test_memory_efficiency(self):
        """Test memory usage with larger images"""
        # Create a larger test image (1024x1024)
        large_image = Image.new('RGB', (1024, 1024), color='white')
        
        # Add lots of content with clear edges to ensure edge density is high enough
        img_array = np.array(large_image)
        
        # Add multiple rectangles
        for i in range(0, 1000, 100):
            cv2.rectangle(img_array, (i, i), (i+80, i+80), (0, 0, 0), 3)
            cv2.rectangle(img_array, (i+20, i+20), (i+60, i+60), (128, 128, 128), -1)
        
        # Add grid pattern
        for i in range(0, 1024, 50):
            cv2.line(img_array, (i, 0), (i, 1024), (64, 64, 64), 1)
            cv2.line(img_array, (0, i), (1024, i), (64, 64, 64), 1)
        
        # Add some circles
        for i in range(100, 900, 200):
            for j in range(100, 900, 200):
                cv2.circle(img_array, (i, j), 30, (255, 0, 0), 2)
                cv2.circle(img_array, (i, j), 20, (0, 255, 0), -1)
        
        large_image = Image.fromarray(img_array)
        
        # Configure for memory efficiency with lenient validation
        config = EdgeExtractionConfig(
            apply_gaussian_blur=True,
            gaussian_blur_kernel=3,  # Smaller kernel
            apply_morphology=True,
            morphology_kernel_size=3,  # Smaller kernel
            output_channels=1,  # Single channel to save memory
            normalize_output=True,
            min_edge_density=0.0001,  # Very lenient
            min_connected_components=1
        )
        
        # Process large image
        result = extract_edges_from_image(large_image, config)
        
        # Should succeed without memory issues
        self.assertTrue(result.success)
        self.assertEqual(result.edge_map.shape, (1024, 1024, 1))
        self.assertGreater(result.processing_time_ms, 0)
    
    def test_error_recovery_integration(self):
        """Test error handling in pipeline context"""
        # Test with problematic inputs that might occur in real datasets
        
        # Very small image
        tiny_image = Image.new('RGB', (10, 10), color='gray')
        result_tiny = extract_edges_from_image(tiny_image)
        self.assertIsInstance(result_tiny, EdgeExtractionResult)
        # May succeed or fail, but should handle gracefully
        
        # Very large image (memory test)
        try:
            huge_image = Image.new('RGB', (4096, 4096), color='white')
            result_huge = extract_edges_from_image(huge_image)
            self.assertIsInstance(result_huge, EdgeExtractionResult)
            # Should either succeed or fail gracefully
        except MemoryError:
            # This is acceptable for very large images
            pass
        
        # Corrupted/invalid data
        invalid_array = np.full((256, 256, 3), np.nan, dtype=np.float32)
        result_invalid = extract_edges_from_image(invalid_array)
        self.assertIsInstance(result_invalid, EdgeExtractionResult)
        # Should fail gracefully with error message
        if not result_invalid.success:
            self.assertIsNotNone(result_invalid.error_message)
    
    def test_configuration_robustness(self):
        """Test robustness with various configuration combinations"""
        # Test extreme configurations that might be used in experimentation
        
        extreme_configs = [
            # Very sensitive
            EdgeExtractionConfig(
                adaptive_threshold=False,
                low_threshold=1.0,
                high_threshold=10.0,
                apply_gaussian_blur=False,
                apply_morphology=False
            ),
            # Very insensitive
            EdgeExtractionConfig(
                adaptive_threshold=False,
                low_threshold=200.0,
                high_threshold=250.0,
                apply_gaussian_blur=True,
                gaussian_blur_kernel=15,
                apply_morphology=True,
                morphology_kernel_size=7
            ),
            # Adaptive with extreme percentiles
            EdgeExtractionConfig(
                adaptive_threshold=True,
                threshold_percentile_low=0.01,
                threshold_percentile_high=0.99,
                threshold_multiplier_low=0.1,
                threshold_multiplier_high=5.0
            )
        ]
        
        test_image = self.test_images[1]  # Use complex scene
        
        for i, config in enumerate(extreme_configs):
            with self.subTest(config_index=i):
                result = extract_edges_from_image(test_image, config)
                
                # Should handle gracefully (may succeed or fail)
                self.assertIsInstance(result, EdgeExtractionResult)
                self.assertIsNotNone(result.edge_map)
                self.assertGreater(result.processing_time_ms, 0)
                
                if not result.success:
                    # If failed, should have error message
                    self.assertIsNotNone(result.error_message)


class TestDatasetIntegration(unittest.TestCase):
    """Test integration with dataset processing patterns"""
    
    def test_dataset_sample_format(self):
        """Test processing samples in dataset format"""
        # Simulate dataset samples with metadata
        dataset_samples = [
            {
                'image': Image.new('RGB', (256, 256), 'white'),
                'caption': 'A white square',
                'image_id': 'test_001'
            },
            {
                'image': Image.new('RGB', (256, 256), 'black'),
                'caption': 'A black square', 
                'image_id': 'test_002'
            }
        ]
        
        # Process samples
        edge_results = []
        for sample in dataset_samples:
            result = extract_edges_from_image(sample['image'])
            edge_results.append({
                'image_id': sample['image_id'],
                'caption': sample['caption'],
                'edge_map': result.edge_map if result.success else None,
                'edge_extraction_success': result.success,
                'edge_density': result.edge_density,
                'processing_time_ms': result.processing_time_ms
            })
        
        # Verify results
        self.assertEqual(len(edge_results), 2)
        for result in edge_results:
            self.assertIn('image_id', result)
            self.assertIn('edge_extraction_success', result)
            self.assertIsInstance(result['processing_time_ms'], float)
            self.assertGreaterEqual(result['edge_density'], 0.0)


if __name__ == "__main__":
    # Run integration tests
    unittest.main(verbosity=2)