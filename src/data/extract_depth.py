"""
Depth Map Extraction using Intel DPT Model for ControlNet Training Pipeline

This module provides depth map extraction capabilities using the Intel DPT (Dense Prediction Transformer)
model. It includes batch processing with memory optimization for T4 GPU constraints, depth map validation,
and normalization to 0-1 range as required for ControlNet conditioning.

Key Features:
- Intel DPT-Large model integration for high-quality depth estimation
- Memory-optimized batch processing for T4 GPU (15GB VRAM)
- Automatic depth map validation and normalization (0-1 range)
- Robust error handling and fallback mechanisms
- Progress tracking and performance monitoring
- Support for various input image formats and sizes

Requirements Addressed:
- 2.2: Generate depth maps using DPT model
- 9.2: Validate condition map outputs with correct dimensions and value ranges
- 9.3: Log failures and skip corrupted samples
- 4.7: Memory optimization for T4 GPU constraints

Technical Implementation:
- Uses Intel DPT-Large model from transformers library
- Implements gradient-free inference for memory efficiency
- Supports batch processing with dynamic batch size adjustment
- Normalizes depth maps to [0, 1] range for ControlNet compatibility
- Handles various image formats (RGB, RGBA, grayscale)
"""

import gc
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
import hashlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
from tqdm import tqdm
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


@dataclass
class DepthExtractionConfig:
    """Configuration for depth map extraction"""
    model_name: str = "Intel/dpt-large"
    device: str = "auto"  # "auto", "cuda", "cpu"
    batch_size: int = 1  # Optimized for T4 GPU
    max_batch_size: int = 4  # Maximum batch size to attempt
    target_size: Tuple[int, int] = (512, 512)  # Target output size
    normalize_range: Tuple[float, float] = (0.0, 1.0)  # Output normalization range
    memory_threshold_gb: float = 12.0  # Memory threshold for batch size adjustment
    enable_memory_monitoring: bool = True
    cache_dir: Optional[str] = None
    precision: str = "fp16"  # "fp32", "fp16" for memory optimization


@dataclass
class DepthExtractionResult:
    """Result of depth map extraction for a single image"""
    depth_map: np.ndarray
    original_size: Tuple[int, int]
    processing_time_ms: float
    memory_used_mb: float
    success: bool = True
    error_message: Optional[str] = None
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate depth map quality and format"""
        errors = []
        
        if not self.success:
            errors.append(f"Extraction failed: {self.error_message}")
            return False, errors
        
        if self.depth_map is None:
            errors.append("Depth map is None")
            return False, errors
        
        # Check dimensions
        if len(self.depth_map.shape) != 3:
            errors.append(f"Invalid depth map shape: {self.depth_map.shape} (expected 3D)")
        elif self.depth_map.shape[2] != 1:
            errors.append(f"Invalid depth map channels: {self.depth_map.shape[2]} (expected 1)")
        
        # Check value range
        min_val, max_val = self.depth_map.min(), self.depth_map.max()
        if not (0.0 <= min_val <= max_val <= 1.0):
            errors.append(f"Depth values out of range [0,1]: [{min_val:.3f}, {max_val:.3f}]")
        
        # Check for invalid values
        if np.any(np.isnan(self.depth_map)):
            errors.append("Depth map contains NaN values")
        
        if np.any(np.isinf(self.depth_map)):
            errors.append("Depth map contains infinite values")
        
        return len(errors) == 0, errors


@dataclass
class BatchExtractionReport:
    """Report for batch depth extraction processing"""
    total_images: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    total_processing_time_seconds: float = 0.0
    average_processing_time_ms: float = 0.0
    peak_memory_usage_gb: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        return self.successful_extractions / self.total_images if self.total_images > 0 else 0.0
    
    def add_error(self, error: str) -> None:
        """Add error to report"""
        self.errors.append(error)
        logger.error(f"Depth extraction error: {error}")
    
    def finalize(self) -> None:
        """Finalize report statistics"""
        if self.successful_extractions > 0:
            self.average_processing_time_ms = (
                self.total_processing_time_seconds * 1000 / self.successful_extractions
            )
        
        logger.info(f"Depth extraction completed: {self.successful_extractions}/{self.total_images} "
                   f"successful ({self.success_rate:.1%} success rate)")


class MemoryMonitor:
    """GPU and system memory monitoring utility"""
    
    def __init__(self, enable_monitoring: bool = True):
        self.enable_monitoring = enable_monitoring
        self.peak_gpu_memory = 0.0
        self.peak_system_memory = 0.0
    
    def get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in GB"""
        if not self.enable_monitoring or not torch.cuda.is_available():
            return 0.0
        
        return torch.cuda.memory_allocated() / (1024**3)
    
    def get_system_memory_usage(self) -> float:
        """Get current system memory usage in GB"""
        if not self.enable_monitoring:
            return 0.0
        
        return psutil.virtual_memory().used / (1024**3)
    
    def update_peak_usage(self) -> None:
        """Update peak memory usage tracking"""
        if not self.enable_monitoring:
            return
        
        gpu_usage = self.get_gpu_memory_usage()
        system_usage = self.get_system_memory_usage()
        
        self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_usage)
        self.peak_system_memory = max(self.peak_system_memory, system_usage)
    
    def clear_gpu_cache(self) -> None:
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


class DepthExtractor:
    """
    Intel DPT-based depth map extractor with memory optimization
    
    This class provides high-quality depth map extraction using the Intel DPT-Large model
    with optimizations for T4 GPU memory constraints. It supports batch processing,
    automatic memory management, and robust error handling.
    """
    
    def __init__(self, config: Optional[DepthExtractionConfig] = None):
        """
        Initialize depth extractor
        
        Args:
            config: Configuration for depth extraction
        """
        self.config = config or DepthExtractionConfig()
        self.memory_monitor = MemoryMonitor(self.config.enable_memory_monitoring)
        
        # Initialize device
        self.device = self._setup_device()
        logger.info(f"DepthExtractor initialized on device: {self.device}")
        
        # Initialize model and processor
        self.model = None
        self.processor = None
        self._load_model()
        
        # Processing statistics
        self.report = BatchExtractionReport()
    
    def _setup_device(self) -> torch.device:
        """Setup computation device based on configuration"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
                logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                device = torch.device("cpu")
                logger.warning("CUDA not available, using CPU")
        else:
            device = torch.device(self.config.device)
        
        return device
    
    def _load_model(self) -> None:
        """Load DPT model and processor with memory optimization"""
        logger.info(f"Loading DPT model: {self.config.model_name}")
        
        try:
            # Load processor
            self.processor = DPTImageProcessor.from_pretrained(
                self.config.model_name,
                cache_dir=self.config.cache_dir
            )
            
            # Load model with appropriate precision
            if self.config.precision == "fp16" and self.device.type == "cuda":
                self.model = DPTForDepthEstimation.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16,
                    cache_dir=self.config.cache_dir
                ).to(self.device)
                logger.info("Model loaded in FP16 precision for memory efficiency")
            else:
                self.model = DPTForDepthEstimation.from_pretrained(
                    self.config.model_name,
                    cache_dir=self.config.cache_dir
                ).to(self.device)
                logger.info("Model loaded in FP32 precision")
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Disable gradients for inference
            for param in self.model.parameters():
                param.requires_grad = False
            
            logger.info("DPT model loaded successfully")
            
        except Exception as e:
            error_msg = f"Failed to load DPT model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for DPT model input
        
        Args:
            image: PIL Image to preprocess
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process with DPT processor
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        
        # Apply precision conversion if needed
        if self.config.precision == "fp16" and self.device.type == "cuda":
            pixel_values = pixel_values.half()
        
        return pixel_values
    
    def _postprocess_depth(self, 
                          depth_tensor: torch.Tensor, 
                          target_size: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess depth tensor to normalized depth map
        
        Args:
            depth_tensor: Raw depth tensor from model
            target_size: Target output size (width, height)
            
        Returns:
            Normalized depth map as numpy array
        """
        # Move to CPU and convert to float32
        depth = depth_tensor.cpu().float().numpy()
        
        # Remove batch dimension if present
        if depth.ndim == 3:
            depth = depth[0]
        
        # Resize to target size
        if depth.shape != target_size[::-1]:  # OpenCV uses (height, width)
            depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1] range
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            # Handle edge case where depth is constant
            depth = np.zeros_like(depth)
        
        # Apply target normalization range
        min_range, max_range = self.config.normalize_range
        depth = depth * (max_range - min_range) + min_range
        
        # Add channel dimension for consistency
        depth = np.expand_dims(depth, axis=2)
        
        return depth.astype(np.float32)
    
    def _adjust_batch_size(self, current_batch_size: int, memory_usage: float) -> int:
        """
        Dynamically adjust batch size based on memory usage
        
        Args:
            current_batch_size: Current batch size
            memory_usage: Current memory usage in GB
            
        Returns:
            Adjusted batch size
        """
        if memory_usage > self.config.memory_threshold_gb:
            # Reduce batch size
            new_batch_size = max(1, current_batch_size // 2)
            logger.warning(f"Memory usage {memory_usage:.1f}GB exceeds threshold, "
                          f"reducing batch size from {current_batch_size} to {new_batch_size}")
            return new_batch_size
        elif memory_usage < self.config.memory_threshold_gb * 0.7 and current_batch_size < self.config.max_batch_size:
            # Increase batch size if memory allows
            new_batch_size = min(self.config.max_batch_size, current_batch_size + 1)
            logger.info(f"Memory usage {memory_usage:.1f}GB allows increase, "
                       f"increasing batch size from {current_batch_size} to {new_batch_size}")
            return new_batch_size
        
        return current_batch_size
    
    def extract(self, image: Image.Image) -> DepthExtractionResult:
        """
        Extract depth map from a single image
        
        Args:
            image: PIL Image to extract depth from
            
        Returns:
            DepthExtractionResult containing depth map and metadata
        """
        start_time = time.time()
        initial_memory = self.memory_monitor.get_gpu_memory_usage()
        
        try:
            # Validate input
            if image is None:
                return DepthExtractionResult(
                    depth_map=None,
                    original_size=(0, 0),
                    processing_time_ms=0.0,
                    memory_used_mb=0.0,
                    success=False,
                    error_message="Input image is None"
                )
            
            original_size = image.size
            
            # Preprocess image
            pixel_values = self._preprocess_image(image)
            
            # Extract depth with no gradient computation
            with torch.no_grad():
                outputs = self.model(pixel_values)
                depth_tensor = outputs.predicted_depth
            
            # Postprocess depth map
            depth_map = self._postprocess_depth(depth_tensor, self.config.target_size)
            
            # Calculate processing metrics
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            final_memory = self.memory_monitor.get_gpu_memory_usage()
            memory_used = (final_memory - initial_memory) * 1024  # Convert to MB
            
            # Update memory monitoring
            self.memory_monitor.update_peak_usage()
            
            return DepthExtractionResult(
                depth_map=depth_map,
                original_size=original_size,
                processing_time_ms=processing_time,
                memory_used_mb=memory_used,
                success=True
            )
            
        except Exception as e:
            error_msg = f"Depth extraction failed: {str(e)}"
            logger.error(error_msg)
            
            processing_time = (time.time() - start_time) * 1000
            
            return DepthExtractionResult(
                depth_map=None,
                original_size=image.size if image else (0, 0),
                processing_time_ms=processing_time,
                memory_used_mb=0.0,
                success=False,
                error_message=error_msg
            )
        
        finally:
            # Clean up GPU memory
            if self.device.type == "cuda":
                self.memory_monitor.clear_gpu_cache()
    
    def extract_batch(self, images: List[Image.Image]) -> List[DepthExtractionResult]:
        """
        Extract depth maps from a batch of images with memory optimization
        
        Args:
            images: List of PIL Images to process
            
        Returns:
            List of DepthExtractionResult objects
        """
        if not images:
            return []
        
        logger.info(f"Starting batch depth extraction for {len(images)} images")
        start_time = time.time()
        
        results = []
        current_batch_size = self.config.batch_size
        
        # Process images in batches
        progress_bar = tqdm(total=len(images), desc="Extracting depth maps", unit="images")
        
        i = 0
        while i < len(images):
            batch_end = min(i + current_batch_size, len(images))
            batch_images = images[i:batch_end]
            
            try:
                # Monitor memory before batch processing
                memory_before = self.memory_monitor.get_gpu_memory_usage()
                
                # Process batch
                batch_results = []
                for image in batch_images:
                    result = self.extract(image)
                    batch_results.append(result)
                    
                    # Update statistics
                    if result.success:
                        self.report.successful_extractions += 1
                        self.report.total_processing_time_seconds += result.processing_time_ms / 1000
                    else:
                        self.report.failed_extractions += 1
                        self.report.add_error(result.error_message or "Unknown error")
                
                results.extend(batch_results)
                
                # Monitor memory after batch processing
                memory_after = self.memory_monitor.get_gpu_memory_usage()
                self.report.peak_memory_usage_gb = max(
                    self.report.peak_memory_usage_gb, 
                    memory_after
                )
                
                # Adjust batch size based on memory usage
                current_batch_size = self._adjust_batch_size(current_batch_size, memory_after)
                
                # Update progress
                progress_bar.update(len(batch_images))
                progress_bar.set_postfix({
                    'batch_size': current_batch_size,
                    'memory': f"{memory_after:.1f}GB",
                    'success_rate': f"{self.report.successful_extractions/(i+len(batch_images)):.1%}"
                })
                
                i = batch_end
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Handle OOM error
                    logger.warning(f"GPU OOM with batch size {current_batch_size}, reducing to 1")
                    current_batch_size = 1
                    self.memory_monitor.clear_gpu_cache()
                    
                    # Process images one by one
                    for image in batch_images:
                        result = self.extract(image)
                        results.append(result)
                        
                        if result.success:
                            self.report.successful_extractions += 1
                            self.report.total_processing_time_seconds += result.processing_time_ms / 1000
                        else:
                            self.report.failed_extractions += 1
                            self.report.add_error(result.error_message or "Unknown error")
                    
                    progress_bar.update(len(batch_images))
                    i = batch_end
                else:
                    # Other runtime errors
                    error_msg = f"Batch processing failed: {str(e)}"
                    logger.error(error_msg)
                    self.report.add_error(error_msg)
                    
                    # Skip this batch and continue
                    for _ in batch_images:
                        results.append(DepthExtractionResult(
                            depth_map=None,
                            original_size=(0, 0),
                            processing_time_ms=0.0,
                            memory_used_mb=0.0,
                            success=False,
                            error_message=error_msg
                        ))
                        self.report.failed_extractions += 1
                    
                    progress_bar.update(len(batch_images))
                    i = batch_end
        
        progress_bar.close()
        
        # Finalize report
        self.report.total_images = len(images)
        self.report.total_processing_time_seconds = time.time() - start_time
        self.report.finalize()
        
        logger.info(f"Batch depth extraction completed: {self.report.successful_extractions}/{len(images)} "
                   f"successful ({self.report.success_rate:.1%})")
        
        return results
    
    def validate_output(self, depth_map: np.ndarray) -> bool:
        """
        Validate depth map output format and quality
        
        Args:
            depth_map: Depth map to validate
            
        Returns:
            True if valid, False otherwise
        """
        if depth_map is None:
            return False
        
        # Check basic format
        if not isinstance(depth_map, np.ndarray):
            return False
        
        if len(depth_map.shape) != 3 or depth_map.shape[2] != 1:
            return False
        
        # Check value range
        if not (0.0 <= depth_map.min() <= depth_map.max() <= 1.0):
            return False
        
        # Check for invalid values
        if np.any(np.isnan(depth_map)) or np.any(np.isinf(depth_map)):
            return False
        
        return True
    
    def get_processing_report(self) -> BatchExtractionReport:
        """Get comprehensive processing report"""
        return self.report
    
    def save_depth_map(self, 
                      depth_map: np.ndarray, 
                      output_path: Union[str, Path],
                      format: str = "png") -> None:
        """
        Save depth map to file
        
        Args:
            depth_map: Depth map to save
            output_path: Output file path
            format: Output format ("png", "npy", "jpg")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "npy":
            # Save as numpy array (preserves exact values)
            np.save(output_path, depth_map)
        else:
            # Convert to 8-bit image for standard formats
            if depth_map.shape[2] == 1:
                depth_image = (depth_map[:, :, 0] * 255).astype(np.uint8)
            else:
                depth_image = (depth_map * 255).astype(np.uint8)
            
            # Save as image
            Image.fromarray(depth_image, mode='L').save(output_path, format.upper())
        
        logger.info(f"Depth map saved to {output_path}")
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'memory_monitor'):
            self.memory_monitor.clear_gpu_cache()


# Utility functions for common depth extraction operations

def extract_depth_from_images(image_paths: List[Union[str, Path]],
                            output_dir: Optional[Union[str, Path]] = None,
                            config: Optional[DepthExtractionConfig] = None) -> List[DepthExtractionResult]:
    """
    Convenience function to extract depth maps from image files
    
    Args:
        image_paths: List of paths to input images
        output_dir: Optional directory to save depth maps
        config: Optional configuration for extraction
        
    Returns:
        List of DepthExtractionResult objects
    """
    extractor = DepthExtractor(config)
    
    # Load images
    images = []
    valid_paths = []
    
    for path in image_paths:
        try:
            image = Image.open(path).convert('RGB')
            images.append(image)
            valid_paths.append(Path(path))
        except Exception as e:
            logger.error(f"Failed to load image {path}: {str(e)}")
            images.append(None)
            valid_paths.append(None)
    
    # Extract depth maps
    results = extractor.extract_batch(images)
    
    # Save depth maps if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (result, path) in enumerate(zip(results, valid_paths)):
            if result.success and path:
                output_path = output_dir / f"{path.stem}_depth.png"
                extractor.save_depth_map(result.depth_map, output_path)
    
    return results


def create_depth_dataset(images: List[Image.Image],
                        config: Optional[DepthExtractionConfig] = None) -> Tuple[List[np.ndarray], BatchExtractionReport]:
    """
    Create depth map dataset from list of images
    
    Args:
        images: List of PIL Images
        config: Optional configuration for extraction
        
    Returns:
        Tuple of (depth_maps, report)
    """
    extractor = DepthExtractor(config)
    results = extractor.extract_batch(images)
    
    # Extract successful depth maps
    depth_maps = []
    for result in results:
        if result.success:
            depth_maps.append(result.depth_map)
        else:
            depth_maps.append(None)
    
    return depth_maps, extractor.get_processing_report()


if __name__ == "__main__":
    # Example usage and testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Depth Map Extraction using DPT")
    parser.add_argument("--input", type=str, required=True, help="Input image or directory")
    parser.add_argument("--output", type=str, default="./depth_output", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--precision", type=str, default="fp16", help="Model precision (fp32, fp16)")
    
    args = parser.parse_args()
    
    # Setup configuration
    config = DepthExtractionConfig(
        batch_size=args.batch_size,
        device=args.device,
        precision=args.precision
    )
    
    # Process input
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        image_paths = [input_path]
    elif input_path.is_dir():
        # Directory of images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = [
            p for p in input_path.rglob('*') 
            if p.suffix.lower() in image_extensions
        ]
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
    
    logger.info(f"Found {len(image_paths)} images to process")
    
    # Extract depth maps
    results = extract_depth_from_images(
        image_paths=image_paths,
        output_dir=args.output,
        config=config
    )
    
    # Print summary
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    print(f"\nDepth Extraction Summary:")
    print(f"Total images: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(results):.1%}")
    
    if successful > 0:
        avg_time = np.mean([r.processing_time_ms for r in results if r.success])
        print(f"Average processing time: {avg_time:.1f}ms")
    
    print(f"Output saved to: {args.output}")