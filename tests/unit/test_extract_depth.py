"""
Unit tests for depth map extraction module

Tests the DepthExtractor class and related functionality for extracting depth maps
using the Intel DPT model. Includes tests for memory optimization, batch processing,
validation, and error handling.
"""

import pytest
import numpy as np
import torch
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path

# Import the module under test
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from data.extract_depth import (
    DepthExtractor,
    DepthExtractionConfig,
    DepthExtractionResult,
    BatchExtractionReport,
    MemoryMonitor,
    extract_depth_from_images,
    create_depth_dataset
)


class TestDepthExtractionConfig:
    """Test DepthExtractionConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = DepthExtractionConfig()
        
        assert config.model_name == "Intel/dpt-large"
        assert config.device == "auto"
        assert config.batch_size == 1
        assert config.max_batch_size == 4
        assert config.target_size == (512, 512)
        assert config.normalize_range == (0.0, 1.0)
        assert config.memory_threshold_gb == 12.0
        assert config.enable_memory_monitoring is True
        assert config.precision == "fp16"
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = DepthExtractionConfig(
            model_name="custom/model",
            device="cpu",
            batch_size=2,
            target_size=(256, 256),
            normalize_range=(0.1, 0.9)
        )
        
        assert config.model_name == "custom/model"
        assert config.device == "cpu"
        assert config.batch_size == 2
        assert config.target_size == (256, 256)
        assert config.normalize_range == (0.1, 0.9)


class TestDepthExtractionResult:
    """Test DepthExtractionResult dataclass"""
    
    def test_successful_result_validation(self):
        """Test validation of successful depth extraction result"""
        depth_map = np.random.rand(512, 512, 1).astype(np.float32)
        
        result = DepthExtractionResult(
            depth_map=depth_map,
            original_size=(1024, 768),
            processing_time_ms=150.0,
            memory_used_mb=256.0,
            success=True
        )
        
        is_valid, errors = result.validate()
        assert is_valid
        assert len(errors) == 0
    
    def test_failed_result_validation(self):
        """Test validation of failed depth extraction result"""
        result = DepthExtractionResult(
            depth_map=None,
            original_size=(0, 0),
            processing_time_ms=0.0,
            memory_used_mb=0.0,
            success=False,
            error_message="Test error"
        )
        
        is_valid, errors = result.validate()
        assert not is_valid
        assert len(errors) == 1
        assert "Test error" in errors[0]
    
    def test_invalid_depth_map_shape(self):
        """Test validation with invalid depth map shape"""
        # 2D depth map (missing channel dimension)
        depth_map = np.random.rand(512, 512).astype(np.float32)
        
        result = DepthExtractionResult(
            depth_map=depth_map,
            original_size=(512, 512),
            processing_time_ms=100.0,
            memory_used_mb=128.0,
            success=True
        )
        
        is_valid, errors = result.validate()
        assert not is_valid
        assert any("Invalid depth map shape" in error for error in errors)
    
    def test_invalid_value_range(self):
        """Test validation with values outside [0,1] range"""
        depth_map = np.random.rand(512, 512, 1).astype(np.float32) * 2.0  # Values in [0,2]
        
        result = DepthExtractionResult(
            depth_map=depth_map,
            original_size=(512, 512),
            processing_time_ms=100.0,
            memory_used_mb=128.0,
            success=True
        )
        
        is_valid, errors = result.validate()
        assert not is_valid
        assert any("out of range" in error for error in errors)
    
    def test_nan_values(self):
        """Test validation with NaN values"""
        depth_map = np.random.rand(512, 512, 1).astype(np.float32)
        depth_map[0, 0, 0] = np.nan
        
        result = DepthExtractionResult(
            depth_map=depth_map,
            original_size=(512, 512),
            processing_time_ms=100.0,
            memory_used_mb=128.0,
            success=True
        )
        
        is_valid, errors = result.validate()
        assert not is_valid
        assert any("NaN values" in error for error in errors)


class TestMemoryMonitor:
    """Test MemoryMonitor class"""
    
    def test_memory_monitor_disabled(self):
        """Test memory monitor with monitoring disabled"""
        monitor = MemoryMonitor(enable_monitoring=False)
        
        assert monitor.get_gpu_memory_usage() == 0.0
        assert monitor.get_system_memory_usage() == 0.0
        
        # Should not raise any errors
        monitor.update_peak_usage()
        monitor.clear_gpu_cache()
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=1024**3)  # 1GB
    def test_gpu_memory_monitoring(self, mock_memory, mock_cuda):
        """Test GPU memory monitoring"""
        monitor = MemoryMonitor(enable_monitoring=True)
        
        memory_usage = monitor.get_gpu_memory_usage()
        assert memory_usage == 1.0  # 1GB
    
    @patch('psutil.virtual_memory')
    def test_system_memory_monitoring(self, mock_memory):
        """Test system memory monitoring"""
        mock_memory.return_value.used = 2 * 1024**3  # 2GB
        
        monitor = MemoryMonitor(enable_monitoring=True)
        memory_usage = monitor.get_system_memory_usage()
        assert memory_usage == 2.0  # 2GB


class TestDepthExtractor:
    """Test DepthExtractor class"""
    
    @pytest.fixture
    def mock_model_components(self):
        """Mock DPT model and processor"""
        with patch('data.extract_depth.DPTImageProcessor') as mock_processor_class, \
             patch('data.extract_depth.DPTForDepthEstimation') as mock_model_class:
            
            # Mock processor
            mock_processor = Mock()
            mock_processor.from_pretrained.return_value = mock_processor
            mock_processor_class.from_pretrained.return_value = mock_processor
            
            # Mock model
            mock_model = Mock()
            mock_model.eval.return_value = None
            mock_model.parameters.return_value = [Mock(requires_grad=True)]
            mock_model_class.from_pretrained.return_value = mock_model
            
            # Mock model output
            mock_output = Mock()
            mock_output.predicted_depth = torch.randn(1, 384, 384)
            mock_model.return_value = mock_output
            
            # Mock processor output
            mock_processor.return_value = {
                "pixel_values": torch.randn(1, 3, 384, 384)
            }
            
            yield mock_processor, mock_model
    
    def test_extractor_initialization(self, mock_model_components):
        """Test DepthExtractor initialization"""
        config = DepthExtractionConfig(device="cpu")
        extractor = DepthExtractor(config)
        
        assert extractor.config == config
        assert extractor.device.type == "cpu"
        assert extractor.model is not None
        assert extractor.processor is not None
    
    def test_device_setup_auto_cuda(self, mock_model_components):
        """Test automatic device setup with CUDA available"""
        with patch('torch.cuda.is_available', return_value=True):
            config = DepthExtractionConfig(device="auto")
            extractor = DepthExtractor(config)
            assert extractor.device.type == "cuda"
    
    def test_device_setup_auto_cpu(self, mock_model_components):
        """Test automatic device setup with CUDA unavailable"""
        with patch('torch.cuda.is_available', return_value=False):
            config = DepthExtractionConfig(device="auto")
            extractor = DepthExtractor(config)
            assert extractor.device.type == "cpu"
    
    def test_preprocess_image(self, mock_model_components):
        """Test image preprocessing"""
        config = DepthExtractionConfig(device="cpu")
        extractor = DepthExtractor(config)
        
        # Create test image
        image = Image.new('RGB', (512, 512), color='red')
        
        # Preprocess
        tensor = extractor._preprocess_image(image)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.device.type == "cpu"
    
    def test_postprocess_depth(self, mock_model_components):
        """Test depth tensor postprocessing"""
        config = DepthExtractionConfig(device="cpu", target_size=(256, 256))
        extractor = DepthExtractor(config)
        
        # Create test depth tensor
        depth_tensor = torch.randn(1, 384, 384)
        
        # Postprocess
        depth_map = extractor._postprocess_depth(depth_tensor, (256, 256))
        
        assert isinstance(depth_map, np.ndarray)
        assert depth_map.shape == (256, 256, 1)
        assert 0.0 <= depth_map.min() <= depth_map.max() <= 1.0
        assert depth_map.dtype == np.float32
    
    def test_extract_single_image_success(self, mock_model_components):
        """Test successful single image depth extraction"""
        config = DepthExtractionConfig(device="cpu")
        extractor = DepthExtractor(config)
        
        # Create test image
        image = Image.new('RGB', (512, 512), color='blue')
        
        # Extract depth
        result = extractor.extract(image)
        
        assert result.success
        assert result.depth_map is not None
        assert result.depth_map.shape == (512, 512, 1)
        assert result.original_size == (512, 512)
        assert result.processing_time_ms > 0
    
    def test_extract_single_image_none_input(self, mock_model_components):
        """Test depth extraction with None input"""
        config = DepthExtractionConfig(device="cpu")
        extractor = DepthExtractor(config)
        
        result = extractor.extract(None)
        
        assert not result.success
        assert result.depth_map is None
        assert "Input image is None" in result.error_message
    
    def test_extract_batch_success(self, mock_model_components):
        """Test successful batch depth extraction"""
        config = DepthExtractionConfig(device="cpu", batch_size=2)
        extractor = DepthExtractor(config)
        
        # Create test images
        images = [
            Image.new('RGB', (512, 512), color='red'),
            Image.new('RGB', (256, 256), color='green'),
            Image.new('RGB', (768, 768), color='blue')
        ]
        
        # Extract depth maps
        results = extractor.extract_batch(images)
        
        assert len(results) == 3
        assert all(result.success for result in results)
        assert all(result.depth_map is not None for result in results)
        assert all(result.depth_map.shape == (512, 512, 1) for result in results)
    
    def test_extract_batch_empty_list(self, mock_model_components):
        """Test batch extraction with empty list"""
        config = DepthExtractionConfig(device="cpu")
        extractor = DepthExtractor(config)
        
        results = extractor.extract_batch([])
        assert len(results) == 0
    
    def test_validate_output_valid(self, mock_model_components):
        """Test output validation with valid depth map"""
        config = DepthExtractionConfig(device="cpu")
        extractor = DepthExtractor(config)
        
        depth_map = np.random.rand(512, 512, 1).astype(np.float32)
        assert extractor.validate_output(depth_map)
    
    def test_validate_output_invalid(self, mock_model_components):
        """Test output validation with invalid depth map"""
        config = DepthExtractionConfig(device="cpu")
        extractor = DepthExtractor(config)
        
        # Test various invalid cases
        assert not extractor.validate_output(None)
        assert not extractor.validate_output("not an array")
        assert not extractor.validate_output(np.random.rand(512, 512))  # Missing channel dim
        assert not extractor.validate_output(np.random.rand(512, 512, 3))  # Wrong channels
        
        # Values out of range
        invalid_depth = np.random.rand(512, 512, 1) * 2.0  # [0, 2] range
        assert not extractor.validate_output(invalid_depth)
        
        # NaN values
        nan_depth = np.random.rand(512, 512, 1).astype(np.float32)
        nan_depth[0, 0, 0] = np.nan
        assert not extractor.validate_output(nan_depth)
    
    def test_batch_size_adjustment(self, mock_model_components):
        """Test dynamic batch size adjustment"""
        config = DepthExtractionConfig(device="cpu", memory_threshold_gb=1.0, max_batch_size=8)
        extractor = DepthExtractor(config)
        
        # Test reduction due to high memory usage
        new_size = extractor._adjust_batch_size(4, 2.0)  # 2GB > 1GB threshold
        assert new_size == 2  # Should halve
        
        # Test increase due to low memory usage
        new_size = extractor._adjust_batch_size(2, 0.5)  # 0.5GB < 0.7GB threshold
        assert new_size == 3  # Should increase by 1
        
        # Test no change
        new_size = extractor._adjust_batch_size(4, 1.0)  # Exactly at threshold
        assert new_size == 4  # Should stay same
    
    def test_save_depth_map_png(self, mock_model_components):
        """Test saving depth map as PNG"""
        config = DepthExtractionConfig(device="cpu")
        extractor = DepthExtractor(config)
        
        depth_map = np.random.rand(256, 256, 1).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_depth.png"
            extractor.save_depth_map(depth_map, output_path, format="png")
            
            assert output_path.exists()
            
            # Verify saved image can be loaded
            saved_image = Image.open(output_path)
            assert saved_image.mode == 'L'  # Grayscale
            assert saved_image.size == (256, 256)
    
    def test_save_depth_map_npy(self, mock_model_components):
        """Test saving depth map as numpy array"""
        config = DepthExtractionConfig(device="cpu")
        extractor = DepthExtractor(config)
        
        depth_map = np.random.rand(256, 256, 1).astype(np.float32)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_depth.npy"
            extractor.save_depth_map(depth_map, output_path, format="npy")
            
            assert output_path.exists()
            
            # Verify saved array can be loaded
            saved_array = np.load(output_path)
            np.testing.assert_array_equal(saved_array, depth_map)


class TestUtilityFunctions:
    """Test utility functions"""
    
    @patch('data.extract_depth.DepthExtractor')
    def test_extract_depth_from_images(self, mock_extractor_class):
        """Test extract_depth_from_images utility function"""
        # Mock extractor
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        
        # Mock successful results
        mock_results = [
            DepthExtractionResult(
                depth_map=np.random.rand(512, 512, 1),
                original_size=(512, 512),
                processing_time_ms=100.0,
                memory_used_mb=128.0,
                success=True
            )
        ]
        mock_extractor.extract_batch.return_value = mock_results
        mock_extractor.save_depth_map = Mock()
        
        # Create temporary test image
        with tempfile.TemporaryDirectory() as temp_dir:
            test_image_path = Path(temp_dir) / "test.jpg"
            test_image = Image.new('RGB', (512, 512), color='red')
            test_image.save(test_image_path)
            
            output_dir = Path(temp_dir) / "output"
            
            # Test function
            results = extract_depth_from_images(
                image_paths=[test_image_path],
                output_dir=output_dir
            )
            
            assert len(results) == 1
            assert results[0].success
            mock_extractor.extract_batch.assert_called_once()
            mock_extractor.save_depth_map.assert_called_once()
    
    @patch('data.extract_depth.DepthExtractor')
    def test_create_depth_dataset(self, mock_extractor_class):
        """Test create_depth_dataset utility function"""
        # Mock extractor
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        
        # Mock results
        mock_results = [
            DepthExtractionResult(
                depth_map=np.random.rand(512, 512, 1),
                original_size=(512, 512),
                processing_time_ms=100.0,
                memory_used_mb=128.0,
                success=True
            ),
            DepthExtractionResult(
                depth_map=None,
                original_size=(0, 0),
                processing_time_ms=0.0,
                memory_used_mb=0.0,
                success=False,
                error_message="Test error"
            )
        ]
        mock_extractor.extract_batch.return_value = mock_results
        mock_extractor.get_processing_report.return_value = BatchExtractionReport()
        
        # Test images
        images = [
            Image.new('RGB', (512, 512), color='red'),
            Image.new('RGB', (256, 256), color='green')
        ]
        
        # Test function
        depth_maps, report = create_depth_dataset(images)
        
        assert len(depth_maps) == 2
        assert depth_maps[0] is not None  # Successful extraction
        assert depth_maps[1] is None      # Failed extraction
        assert isinstance(report, BatchExtractionReport)


class TestBatchExtractionReport:
    """Test BatchExtractionReport class"""
    
    def test_report_initialization(self):
        """Test report initialization"""
        report = BatchExtractionReport()
        
        assert report.total_images == 0
        assert report.successful_extractions == 0
        assert report.failed_extractions == 0
        assert report.success_rate == 0.0
        assert len(report.errors) == 0
    
    def test_success_rate_calculation(self):
        """Test success rate calculation"""
        report = BatchExtractionReport()
        report.total_images = 10
        report.successful_extractions = 8
        report.failed_extractions = 2
        
        assert report.success_rate == 0.8
    
    def test_success_rate_zero_images(self):
        """Test success rate with zero images"""
        report = BatchExtractionReport()
        assert report.success_rate == 0.0
    
    def test_add_error(self):
        """Test adding errors to report"""
        report = BatchExtractionReport()
        
        report.add_error("Test error 1")
        report.add_error("Test error 2")
        
        assert len(report.errors) == 2
        assert "Test error 1" in report.errors
        assert "Test error 2" in report.errors
    
    def test_finalize_report(self):
        """Test report finalization"""
        report = BatchExtractionReport()
        report.total_images = 5
        report.successful_extractions = 4
        report.failed_extractions = 1
        report.total_processing_time_seconds = 2.0
        
        report.finalize()
        
        assert report.success_rate == 0.8
        assert report.average_processing_time_ms == 500.0  # 2000ms / 4 successful


if __name__ == "__main__":
    pytest.main([__file__, "-v"])