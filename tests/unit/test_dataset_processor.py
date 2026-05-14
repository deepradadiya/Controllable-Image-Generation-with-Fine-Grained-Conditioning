"""
Unit tests for COCO Dataset Processor

Tests the dataset processing functionality including sample validation,
dataset integrity checking, and train/validation splits.
"""

import unittest
import numpy as np
from PIL import Image
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from data.dataset_processor import DatasetProcessor, ProcessingSample, DatasetReport


class TestDatasetProcessor(unittest.TestCase):
    """Test cases for DatasetProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = DatasetProcessor(cache_dir="./test_cache")
    
    def create_mock_sample(self, image_id: str = "test_001") -> dict:
        """Create a mock sample for testing"""
        # Create a simple test image
        image_array = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        image = Image.fromarray(image_array, 'RGB')
        
        return {
            'image': image,
            'caption': 'A test image with random pixels for testing the dataset processor',
            'image_id': image_id,
            'id': image_id
        }
    
    def test_sample_processing_valid(self):
        """Test processing of valid samples"""
        mock_sample = self.create_mock_sample()
        processed_sample = self.processor.process_sample(mock_sample)
        
        self.assertIsNotNone(processed_sample)
        self.assertIsInstance(processed_sample, ProcessingSample)
        self.assertEqual(processed_sample.image_id, "test_001")
        self.assertTrue(len(processed_sample.caption) > 0)
    
    def test_sample_processing_invalid_image(self):
        """Test processing of samples with invalid images"""
        mock_sample = self.create_mock_sample()
        mock_sample['image'] = None
        
        processed_sample = self.processor.process_sample(mock_sample)
        self.assertIsNone(processed_sample)
    
    def test_sample_processing_missing_caption(self):
        """Test processing of samples with missing captions"""
        mock_sample = self.create_mock_sample()
        mock_sample['caption'] = ""
        
        processed_sample = self.processor.process_sample(mock_sample)
        self.assertIsNone(processed_sample)
    
    def test_sample_validation_valid(self):
        """Test validation of valid samples"""
        mock_sample = self.create_mock_sample()
        processed_sample = self.processor.process_sample(mock_sample)
        
        is_valid, errors = processed_sample.validate()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_sample_validation_small_image(self):
        """Test validation of samples with too small images"""
        # Create small image
        image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image = Image.fromarray(image_array, 'RGB')
        
        sample = ProcessingSample(
            image=image,
            caption="Test caption",
            image_id="test_small"
        )
        
        is_valid, errors = sample.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("too small" in error for error in errors))
    
    def test_sample_validation_short_caption(self):
        """Test validation of samples with too short captions"""
        mock_sample = self.create_mock_sample()
        processed_sample = self.processor.process_sample(mock_sample)
        processed_sample.caption = "Hi"  # Too short
        
        is_valid, errors = processed_sample.validate()
        self.assertFalse(is_valid)
        self.assertTrue(any("too short" in error for error in errors))
    
    def test_dataset_validation_all_valid(self):
        """Test dataset validation with all valid samples"""
        samples = []
        for i in range(5):
            mock_sample = self.create_mock_sample(f"test_{i:03d}")
            processed_sample = self.processor.process_sample(mock_sample)
            samples.append(processed_sample)
        
        report = self.processor.validate_dataset_integrity(samples)
        
        self.assertTrue(report.is_valid)
        self.assertEqual(report.valid_samples, 5)
        self.assertEqual(report.invalid_samples, 0)
        self.assertEqual(report.success_rate, 1.0)
    
    def test_dataset_validation_mixed_validity(self):
        """Test dataset validation with mixed valid/invalid samples"""
        samples = []
        
        # Add valid samples
        for i in range(3):
            mock_sample = self.create_mock_sample(f"valid_{i:03d}")
            processed_sample = self.processor.process_sample(mock_sample)
            samples.append(processed_sample)
        
        # Add invalid samples (small images)
        for i in range(2):
            image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            image = Image.fromarray(image_array, 'RGB')
            invalid_sample = ProcessingSample(
                image=image,
                caption="Test caption",
                image_id=f"invalid_{i:03d}"
            )
            samples.append(invalid_sample)
        
        report = self.processor.validate_dataset_integrity(samples)
        
        self.assertEqual(report.total_samples, 5)
        self.assertEqual(report.valid_samples, 3)
        self.assertEqual(report.invalid_samples, 2)
        self.assertEqual(report.success_rate, 0.6)
    
    def test_train_val_split_correct_ratios(self):
        """Test train/validation split produces correct ratios"""
        samples = []
        for i in range(10):
            mock_sample = self.create_mock_sample(f"test_{i:03d}")
            processed_sample = self.processor.process_sample(mock_sample)
            samples.append(processed_sample)
        
        train_samples, val_samples = self.processor.create_train_val_split(
            samples, val_ratio=0.2
        )
        
        self.assertEqual(len(train_samples), 8)
        self.assertEqual(len(val_samples), 2)
        self.assertEqual(len(train_samples) + len(val_samples), len(samples))
    
    def test_train_val_split_edge_cases(self):
        """Test train/validation split edge cases"""
        # Empty samples
        train_samples, val_samples = self.processor.create_train_val_split([], 0.2)
        self.assertEqual(len(train_samples), 0)
        self.assertEqual(len(val_samples), 0)
        
        # Single sample
        samples = [self.processor.process_sample(self.create_mock_sample())]
        train_samples, val_samples = self.processor.create_train_val_split(samples, 0.2)
        self.assertEqual(len(train_samples) + len(val_samples), 1)
        
        # Invalid ratio
        with self.assertRaises(ValueError):
            self.processor.create_train_val_split(samples, -0.1)
        
        with self.assertRaises(ValueError):
            self.processor.create_train_val_split(samples, 1.1)
    
    def test_dataset_statistics_generation(self):
        """Test dataset statistics generation"""
        samples = []
        
        # Create samples with varying properties
        for i in range(5):
            width = 256 + i * 64
            height = 256 + i * 32
            image_array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            image = Image.fromarray(image_array, 'RGB')
            
            sample = ProcessingSample(
                image=image,
                caption=f"Test caption number {i} with varying length",
                image_id=f"test_{i:03d}"
            )
            samples.append(sample)
        
        stats = self.processor.get_dataset_statistics(samples)
        
        self.assertEqual(stats['total_samples'], 5)
        self.assertIn('image_statistics', stats)
        self.assertIn('caption_statistics', stats)
        
        # Check image statistics
        img_stats = stats['image_statistics']
        self.assertEqual(img_stats['width']['min'], 256)
        self.assertEqual(img_stats['width']['max'], 256 + 4 * 64)
        
        # Check caption statistics
        cap_stats = stats['caption_statistics']
        self.assertGreater(cap_stats['length']['mean'], 0)
    
    def test_dataset_statistics_empty(self):
        """Test dataset statistics with empty sample list"""
        stats = self.processor.get_dataset_statistics([])
        self.assertIn('error', stats)


class TestProcessingSample(unittest.TestCase):
    """Test cases for ProcessingSample class"""
    
    def test_valid_sample_creation(self):
        """Test creation of valid ProcessingSample"""
        image_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image = Image.fromarray(image_array, 'RGB')
        
        sample = ProcessingSample(
            image=image,
            caption="A valid test caption for the sample",
            image_id="test_sample_001"
        )
        
        self.assertEqual(sample.image_id, "test_sample_001")
        self.assertEqual(sample.image.size, (512, 512))
        self.assertTrue(len(sample.caption) > 0)
    
    def test_sample_validation_comprehensive(self):
        """Test comprehensive sample validation"""
        # Valid sample
        image_array = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image = Image.fromarray(image_array, 'RGB')
        
        valid_sample = ProcessingSample(
            image=image,
            caption="A valid test caption for comprehensive validation testing",
            image_id="valid_sample"
        )
        
        is_valid, errors = valid_sample.validate()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)


class TestDatasetReport(unittest.TestCase):
    """Test cases for DatasetReport class"""
    
    def test_report_initialization(self):
        """Test DatasetReport initialization"""
        report = DatasetReport()
        
        self.assertEqual(report.total_samples, 0)
        self.assertEqual(report.valid_samples, 0)
        self.assertEqual(report.success_rate, 0.0)
        self.assertFalse(report.is_valid)
    
    def test_report_finalization(self):
        """Test DatasetReport finalization"""
        report = DatasetReport()
        report.total_samples = 10
        report.valid_samples = 8
        
        report.finalize()
        
        self.assertEqual(report.success_rate, 0.8)
        self.assertTrue(report.is_valid)  # 80% success rate >= 80% threshold
    
    def test_report_error_handling(self):
        """Test DatasetReport error and warning handling"""
        report = DatasetReport()
        
        report.add_error("Test error message")
        report.add_warning("Test warning message")
        
        self.assertEqual(len(report.errors), 1)
        self.assertEqual(len(report.warnings), 1)
        self.assertEqual(report.errors[0], "Test error message")
        self.assertEqual(report.warnings[0], "Test warning message")


if __name__ == '__main__':
    unittest.main()