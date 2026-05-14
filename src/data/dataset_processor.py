"""
COCO Dataset Downloader and Processor for ControlNet Training Pipeline

This module provides comprehensive dataset processing capabilities for the ControlNet
training pipeline, including streaming download, validation, and train/validation splits.
Optimized for Google Colab T4 GPU constraints with robust error handling and retry logic.

Key Features:
- Streaming download with progress tracking
- Automatic retry mechanism for network failures
- Dataset validation and integrity checking
- Memory-efficient processing using generators
- Train/validation split functionality
- Comprehensive error handling and logging

Requirements Addressed:
- 2.1: COCO dataset download from HuggingFace datasets
- 9.1: Dataset integrity verification and format compatibility
- 9.3: Failure logging and sample skipping for corrupted data
- 9.5: Dataset statistics and failure mode analysis
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterator, Any
from urllib.error import URLError
import hashlib
import os

import numpy as np
from PIL import Image
import torch
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetReport:
    """Comprehensive dataset processing report"""
    total_samples: int = 0
    valid_samples: int = 0
    invalid_samples: int = 0
    corrupted_images: int = 0
    missing_captions: int = 0
    format_errors: int = 0
    download_failures: int = 0
    processing_time_seconds: float = 0.0
    success_rate: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Check if dataset processing was successful"""
        return self.success_rate >= 0.8 and self.valid_samples > 0
    
    def add_error(self, error: str) -> None:
        """Add error message to report"""
        self.errors.append(error)
        logger.error(f"Dataset processing error: {error}")
    
    def add_warning(self, warning: str) -> None:
        """Add warning message to report"""
        self.warnings.append(warning)
        logger.warning(f"Dataset processing warning: {warning}")
    
    def finalize(self) -> None:
        """Calculate final statistics"""
        if self.total_samples > 0:
            self.success_rate = self.valid_samples / self.total_samples
        else:
            self.success_rate = 0.0
        
        logger.info(f"Dataset processing completed: {self.valid_samples}/{self.total_samples} "
                   f"samples valid ({self.success_rate:.2%} success rate)")


@dataclass
class ProcessingSample:
    """Single sample from dataset processing"""
    image: Image.Image
    caption: str
    image_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate sample integrity and format"""
        errors = []
        
        # Check image validity
        if self.image is None:
            errors.append("Image is None")
        else:
            try:
                # Verify image can be processed
                width, height = self.image.size
                if width < 256 or height < 256:
                    errors.append(f"Image too small: {width}x{height} (minimum 256x256)")
                if width > 2048 or height > 2048:
                    errors.append(f"Image too large: {width}x{height} (maximum 2048x2048)")
                
                # Check image format
                if self.image.mode not in ['RGB', 'RGBA']:
                    errors.append(f"Unsupported image mode: {self.image.mode}")
                    
            except Exception as e:
                errors.append(f"Image validation error: {str(e)}")
        
        # Check caption validity
        if not self.caption or not isinstance(self.caption, str):
            errors.append("Caption is empty or not a string")
        else:
            caption_length = len(self.caption.strip())
            if caption_length < 5:
                errors.append(f"Caption too short: {caption_length} characters")
            elif caption_length > 500:
                errors.append(f"Caption too long: {caption_length} characters")
        
        # Check image_id validity
        if not self.image_id or not isinstance(self.image_id, str):
            errors.append("Image ID is empty or not a string")
        
        return len(errors) == 0, errors


class NetworkRetrySession:
    """HTTP session with automatic retry logic for robust downloads"""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.0):
        self.session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
            backoff_factor=backoff_factor
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
    
    def get(self, url: str, **kwargs) -> requests.Response:
        """GET request with retry logic"""
        return self.session.get(url, **kwargs)
    
    def close(self):
        """Close the session"""
        self.session.close()


class DatasetProcessor:
    """
    COCO Dataset Processor with streaming download and validation
    
    This class handles downloading, processing, and validating COCO dataset samples
    for ControlNet training. It includes robust error handling, retry logic, and
    memory-efficient streaming processing.
    """
    
    def __init__(self, 
                 cache_dir: Optional[str] = None,
                 max_retries: int = 3,
                 timeout: int = 30,
                 chunk_size: int = 1000):
        """
        Initialize dataset processor
        
        Args:
            cache_dir: Directory for caching downloaded data
            max_retries: Maximum number of retry attempts for failed downloads
            timeout: Timeout in seconds for network requests
            chunk_size: Number of samples to process in each chunk
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "controlnet"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_retries = max_retries
        self.timeout = timeout
        self.chunk_size = chunk_size
        
        # Initialize network session with retry logic
        self.session = NetworkRetrySession(max_retries=max_retries)
        
        # Processing statistics
        self.report = DatasetReport()
        
        logger.info(f"DatasetProcessor initialized with cache_dir: {self.cache_dir}")
    
    def download_coco_subset(self, 
                           subset_size: int = 10000,
                           split: str = "train",
                           streaming: bool = True) -> Dataset:
        """
        Download COCO dataset subset from HuggingFace datasets
        
        Args:
            subset_size: Number of samples to download (0 for full dataset)
            split: Dataset split to download ('train', 'validation')
            streaming: Whether to use streaming mode for memory efficiency
            
        Returns:
            Dataset object containing the downloaded samples
            
        Raises:
            DatasetDownloadError: If download fails after all retries
        """
        logger.info(f"Starting COCO dataset download: {subset_size} samples from {split} split")
        start_time = time.time()
        
        try:
            # Load dataset with streaming for memory efficiency
            if streaming:
                dataset = load_dataset(
                    "detection-datasets/coco",
                    split=split,
                    streaming=True,
                    cache_dir=str(self.cache_dir)
                )
                
                # Take subset if specified
                if subset_size > 0:
                    dataset = dataset.take(subset_size)
                    
            else:
                # Load full dataset into memory (not recommended for large datasets)
                dataset = load_dataset(
                    "detection-datasets/coco",
                    split=split,
                    cache_dir=str(self.cache_dir)
                )
                
                if subset_size > 0:
                    dataset = dataset.select(range(min(subset_size, len(dataset))))
            
            download_time = time.time() - start_time
            logger.info(f"Dataset download completed in {download_time:.2f} seconds")
            
            return dataset
            
        except Exception as e:
            error_msg = f"Failed to download COCO dataset: {str(e)}"
            self.report.add_error(error_msg)
            self.report.download_failures += 1
            raise DatasetDownloadError(error_msg, 0, e)
    
    def process_sample(self, sample: Dict[str, Any]) -> Optional[ProcessingSample]:
        """
        Process a single dataset sample with validation
        
        Args:
            sample: Raw sample from HuggingFace dataset
            
        Returns:
            ProcessingSample if valid, None if invalid
        """
        try:
            # Extract image
            image = sample.get('image')
            if image is None:
                self.report.add_error(f"Missing image in sample")
                return None
            
            # Convert to PIL Image if needed
            if not isinstance(image, Image.Image):
                try:
                    image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
                except Exception as e:
                    self.report.add_error(f"Failed to convert image: {str(e)}")
                    return None
            
            # Ensure RGB format
            if image.mode != 'RGB':
                try:
                    image = image.convert('RGB')
                except Exception as e:
                    self.report.add_error(f"Failed to convert image to RGB: {str(e)}")
                    return None
            
            # Extract caption/text
            caption = ""
            if 'caption' in sample:
                caption = sample['caption']
            elif 'text' in sample:
                caption = sample['text']
            elif 'annotations' in sample and sample['annotations']:
                # Try to extract from annotations if available
                annotations = sample['annotations']
                if isinstance(annotations, list) and len(annotations) > 0:
                    caption = annotations[0].get('caption', '')
            
            if not caption:
                self.report.add_error("Missing caption/text in sample")
                return None
            
            # Extract image ID
            image_id = sample.get('image_id', sample.get('id', f"unknown_{hash(str(sample))%10000}"))
            
            # Create processing sample
            processing_sample = ProcessingSample(
                image=image,
                caption=str(caption).strip(),
                image_id=str(image_id),
                metadata=sample
            )
            
            # Validate sample
            is_valid, errors = processing_sample.validate()
            if not is_valid:
                for error in errors:
                    self.report.add_error(f"Sample {image_id} validation failed: {error}")
                return None
            
            return processing_sample
            
        except Exception as e:
            self.report.add_error(f"Failed to process sample: {str(e)}")
            return None
    
    def process_dataset_stream(self, 
                             dataset: Dataset,
                             max_samples: Optional[int] = None) -> Iterator[ProcessingSample]:
        """
        Process dataset samples as a stream with progress tracking
        
        Args:
            dataset: HuggingFace dataset to process
            max_samples: Maximum number of samples to process
            
        Yields:
            ProcessingSample objects for valid samples
        """
        logger.info("Starting dataset stream processing")
        
        processed_count = 0
        valid_count = 0
        
        # Create progress bar
        progress_bar = tqdm(
            desc="Processing samples",
            unit="samples",
            total=max_samples if max_samples else None
        )
        
        try:
            for sample in dataset:
                if max_samples and processed_count >= max_samples:
                    break
                
                processed_count += 1
                self.report.total_samples += 1
                
                # Process sample
                processed_sample = self.process_sample(sample)
                
                if processed_sample is not None:
                    valid_count += 1
                    self.report.valid_samples += 1
                    yield processed_sample
                else:
                    self.report.invalid_samples += 1
                
                # Update progress
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'valid': valid_count,
                    'invalid': processed_count - valid_count,
                    'success_rate': f"{valid_count/processed_count:.1%}" if processed_count > 0 else "0%"
                })
                
                # Periodic logging
                if processed_count % 1000 == 0:
                    logger.info(f"Processed {processed_count} samples, "
                              f"{valid_count} valid ({valid_count/processed_count:.1%})")
        
        finally:
            progress_bar.close()
            logger.info(f"Stream processing completed: {valid_count}/{processed_count} samples valid")
    
    def validate_dataset_integrity(self, samples: List[ProcessingSample]) -> DatasetReport:
        """
        Validate dataset integrity and generate comprehensive report
        
        Args:
            samples: List of processed samples to validate
            
        Returns:
            DatasetReport with validation results and statistics
        """
        logger.info(f"Validating dataset integrity for {len(samples)} samples")
        start_time = time.time()
        
        report = DatasetReport()
        report.total_samples = len(samples)
        
        for i, sample in enumerate(samples):
            try:
                is_valid, errors = sample.validate()
                
                if is_valid:
                    report.valid_samples += 1
                else:
                    report.invalid_samples += 1
                    for error in errors:
                        report.add_error(f"Sample {i} ({sample.image_id}): {error}")
                        
                        # Categorize errors
                        if "image" in error.lower():
                            report.corrupted_images += 1
                        elif "caption" in error.lower():
                            report.missing_captions += 1
                        else:
                            report.format_errors += 1
                            
            except Exception as e:
                report.invalid_samples += 1
                report.add_error(f"Validation error for sample {i}: {str(e)}")
                report.format_errors += 1
        
        report.processing_time_seconds = time.time() - start_time
        report.finalize()
        
        return report
    
    def create_train_val_split(self, 
                             samples: List[ProcessingSample],
                             val_ratio: float = 0.1,
                             random_seed: int = 42) -> Tuple[List[ProcessingSample], List[ProcessingSample]]:
        """
        Create train/validation split from processed samples
        
        Args:
            samples: List of processed samples
            val_ratio: Ratio of samples to use for validation (0.0 to 1.0)
            random_seed: Random seed for reproducible splits
            
        Returns:
            Tuple of (train_samples, val_samples)
        """
        if not 0.0 <= val_ratio <= 1.0:
            raise ValueError(f"val_ratio must be between 0.0 and 1.0, got {val_ratio}")
        
        if len(samples) == 0:
            logger.warning("No samples provided for train/val split")
            return [], []
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        
        # Shuffle samples
        shuffled_samples = samples.copy()
        np.random.shuffle(shuffled_samples)
        
        # Calculate split point
        val_size = int(len(shuffled_samples) * val_ratio)
        train_size = len(shuffled_samples) - val_size
        
        # Split samples
        train_samples = shuffled_samples[:train_size]
        val_samples = shuffled_samples[train_size:]
        
        logger.info(f"Created train/val split: {len(train_samples)} train, "
                   f"{len(val_samples)} validation ({val_ratio:.1%} split)")
        
        return train_samples, val_samples
    
    def save_processed_dataset(self, 
                             samples: List[ProcessingSample],
                             output_path: Union[str, Path],
                             format: str = "json") -> None:
        """
        Save processed dataset to disk
        
        Args:
            samples: List of processed samples to save
            output_path: Path to save the dataset
            format: Output format ('json', 'parquet')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {len(samples)} samples to {output_path}")
        
        if format == "json":
            # Convert samples to serializable format
            data = []
            for sample in samples:
                # Save image separately and store path
                image_filename = f"{sample.image_id}.jpg"
                image_path = output_path.parent / "images" / image_filename
                image_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save image
                sample.image.save(image_path, "JPEG", quality=95)
                
                # Create data entry
                data.append({
                    "image_id": sample.image_id,
                    "image_path": str(image_path),
                    "caption": sample.caption,
                    "metadata": sample.metadata
                })
            
            # Save JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Dataset saved successfully to {output_path}")
    
    def get_dataset_statistics(self, samples: List[ProcessingSample]) -> Dict[str, Any]:
        """
        Generate comprehensive dataset statistics
        
        Args:
            samples: List of processed samples
            
        Returns:
            Dictionary containing dataset statistics
        """
        if not samples:
            return {"error": "No samples provided"}
        
        # Image statistics
        widths = []
        heights = []
        aspect_ratios = []
        
        # Caption statistics
        caption_lengths = []
        
        for sample in samples:
            width, height = sample.image.size
            widths.append(width)
            heights.append(height)
            aspect_ratios.append(width / height)
            caption_lengths.append(len(sample.caption))
        
        stats = {
            "total_samples": len(samples),
            "image_statistics": {
                "width": {
                    "min": min(widths),
                    "max": max(widths),
                    "mean": np.mean(widths),
                    "std": np.std(widths)
                },
                "height": {
                    "min": min(heights),
                    "max": max(heights),
                    "mean": np.mean(heights),
                    "std": np.std(heights)
                },
                "aspect_ratio": {
                    "min": min(aspect_ratios),
                    "max": max(aspect_ratios),
                    "mean": np.mean(aspect_ratios),
                    "std": np.std(aspect_ratios)
                }
            },
            "caption_statistics": {
                "length": {
                    "min": min(caption_lengths),
                    "max": max(caption_lengths),
                    "mean": np.mean(caption_lengths),
                    "std": np.std(caption_lengths)
                }
            }
        }
        
        return stats
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'session'):
            self.session.close()


class DatasetDownloadError(Exception):
    """Raised when dataset download fails"""
    
    def __init__(self, message: str, retry_count: int, last_error: Exception):
        self.retry_count = retry_count
        self.last_error = last_error
        super().__init__(f"Dataset download failed after {retry_count} retries: {message}")


# Utility functions for common dataset operations

def create_coco_dataset_for_training(subset_size: int = 10000,
                                   val_ratio: float = 0.1,
                                   cache_dir: Optional[str] = None) -> Tuple[List[ProcessingSample], List[ProcessingSample], DatasetReport]:
    """
    Convenience function to create train/val datasets from COCO
    
    Args:
        subset_size: Number of samples to download
        val_ratio: Validation split ratio
        cache_dir: Cache directory for downloads
        
    Returns:
        Tuple of (train_samples, val_samples, report)
    """
    processor = DatasetProcessor(cache_dir=cache_dir)
    
    try:
        # Download dataset
        dataset = processor.download_coco_subset(subset_size=subset_size)
        
        # Process samples
        samples = list(processor.process_dataset_stream(dataset, max_samples=subset_size))
        
        # Validate dataset
        report = processor.validate_dataset_integrity(samples)
        
        if not report.is_valid:
            logger.warning(f"Dataset validation failed: {report.success_rate:.1%} success rate")
        
        # Create train/val split
        train_samples, val_samples = processor.create_train_val_split(samples, val_ratio=val_ratio)
        
        return train_samples, val_samples, report
        
    except Exception as e:
        logger.error(f"Failed to create COCO dataset: {str(e)}")
        raise


if __name__ == "__main__":
    # Example usage and testing
    import argparse
    
    parser = argparse.ArgumentParser(description="COCO Dataset Processor")
    parser.add_argument("--subset-size", type=int, default=1000, help="Number of samples to download")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--output-dir", type=str, default="./data/processed", help="Output directory")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory")
    
    args = parser.parse_args()
    
    # Create dataset
    train_samples, val_samples, report = create_coco_dataset_for_training(
        subset_size=args.subset_size,
        val_ratio=args.val_ratio,
        cache_dir=args.cache_dir
    )
    
    # Print report
    print(f"\nDataset Processing Report:")
    print(f"Total samples: {report.total_samples}")
    print(f"Valid samples: {report.valid_samples}")
    print(f"Success rate: {report.success_rate:.1%}")
    print(f"Processing time: {report.processing_time_seconds:.2f} seconds")
    
    if report.errors:
        print(f"\nErrors ({len(report.errors)}):")
        for error in report.errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(report.errors) > 5:
            print(f"  ... and {len(report.errors) - 5} more errors")
    
    # Save datasets
    output_dir = Path(args.output_dir)
    if train_samples:
        processor = DatasetProcessor()
        processor.save_processed_dataset(train_samples, output_dir / "train.json")
        processor.save_processed_dataset(val_samples, output_dir / "val.json")
        
        # Generate statistics
        stats = processor.get_dataset_statistics(train_samples + val_samples)
        with open(output_dir / "statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nDatasets saved to {output_dir}")
        print(f"Train samples: {len(train_samples)}")
        print(f"Validation samples: {len(val_samples)}")