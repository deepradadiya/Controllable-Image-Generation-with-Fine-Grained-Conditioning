"""
Dataset Verification and Quality Assurance for ControlNet Training Pipeline

This module provides comprehensive dataset verification and quality assurance capabilities
for the ControlNet training pipeline. It validates image-prompt-condition triplets,
generates dataset statistics, and performs failure mode analysis to ensure data quality.

Key Features:
- Comprehensive validation of image-prompt-condition triplets
- Dataset completeness and integrity checking
- Statistical analysis and quality metrics
- Failure mode detection and categorization
- Performance monitoring and bottleneck identification
- Detailed reporting with actionable insights

Requirements Addressed:
- 2.6: Check image-prompt-condition triplet completeness and validity
- 9.3: Validate that text prompts are non-empty and within reasonable length limits
- 9.4: Generate dataset report showing success rates and common failure modes
- 9.5: Dataset statistics and failure mode analysis
"""

import json
import logging
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import hashlib
import statistics

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import existing data processing components
from .dataset_processor import ProcessingSample, DatasetReport
try:
    from .extract_depth import DepthExtractor, DepthExtractionResult
    from .extract_pose import PoseExtractor, PoseExtractionResult  
    from .extract_edges import EdgeExtractor, EdgeExtractionResult
    EXTRACTORS_AVAILABLE = True
except ImportError:
    # Graceful fallback if extractors are not available
    EXTRACTORS_AVAILABLE = False
    logger.warning("Condition extractors not available. Some functionality will be limited.")


@dataclass
class ValidationConfig:
    """Configuration for dataset validation"""
    min_image_size: Tuple[int, int] = (256, 256)
    max_image_size: Tuple[int, int] = (2048, 2048)
    min_caption_length: int = 5
    max_caption_length: int = 500
    required_condition_types: List[str] = field(default_factory=lambda: ["depth", "pose", "edge"])
    condition_map_tolerance: float = 0.01  # Tolerance for condition map validation
    enable_visual_validation: bool = True
    sample_validation_count: int = 100  # Number of samples for visual validation
    parallel_processing: bool = True
    max_workers: int = 4


@dataclass
class TripletValidationResult:
    """Result of validating a single image-prompt-condition triplet"""
    sample_id: str
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Component validation results
    image_valid: bool = True
    prompt_valid: bool = True
    condition_maps_valid: Dict[str, bool] = field(default_factory=dict)
    
    # Quality metrics
    image_quality_score: float = 0.0
    prompt_quality_score: float = 0.0
    condition_alignment_scores: Dict[str, float] = field(default_factory=dict)
    
    # Processing metadata
    validation_time_ms: float = 0.0
    
    def add_error(self, component: str, error: str) -> None:
        """Add validation error"""
        self.errors.append(f"{component}: {error}")
        self.is_valid = False
        logger.debug(f"Validation error for {self.sample_id} - {component}: {error}")
    
    def add_warning(self, component: str, warning: str) -> None:
        """Add validation warning"""
        self.warnings.append(f"{component}: {warning}")
        logger.debug(f"Validation warning for {self.sample_id} - {component}: {warning}")


@dataclass
class DatasetValidationReport:
    """Comprehensive dataset validation report"""
    total_samples: int = 0
    valid_samples: int = 0
    invalid_samples: int = 0
    
    # Component-specific validation results
    image_validation_results: Dict[str, int] = field(default_factory=dict)
    prompt_validation_results: Dict[str, int] = field(default_factory=dict)
    condition_validation_results: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Quality statistics
    quality_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Failure analysis
    failure_modes: Dict[str, int] = field(default_factory=dict)
    error_categories: Dict[str, List[str]] = field(default_factory=dict)
    
    # Performance metrics
    validation_time_seconds: float = 0.0
    samples_per_second: float = 0.0
    
    # Dataset statistics
    dataset_statistics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate"""
        return self.valid_samples / self.total_samples if self.total_samples > 0 else 0.0
    
    @property
    def is_acceptable(self) -> bool:
        """Check if dataset quality is acceptable for training"""
        return self.success_rate >= 0.8 and self.valid_samples >= 100
    
    def add_failure_mode(self, mode: str, count: int = 1) -> None:
        """Add or increment failure mode count"""
        self.failure_modes[mode] = self.failure_modes.get(mode, 0) + count
    
    def finalize(self) -> None:
        """Calculate final statistics and metrics"""
        if self.total_samples > 0:
            self.samples_per_second = self.total_samples / max(self.validation_time_seconds, 0.001)
        
        logger.info(f"Dataset validation completed: {self.valid_samples}/{self.total_samples} "
                   f"samples valid ({self.success_rate:.2%} success rate)")


class TripletValidator:
    """Validates individual image-prompt-condition triplets"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
        # Initialize condition extractors for validation
        self.depth_extractor = None
        self.pose_extractor = None
        self.edge_extractor = None
        
        if EXTRACTORS_AVAILABLE:
            self.depth_extractor = DepthExtractor() if "depth" in config.required_condition_types else None
            self.pose_extractor = PoseExtractor() if "pose" in config.required_condition_types else None
            self.edge_extractor = EdgeExtractor() if "edge" in config.required_condition_types else None
    
    def validate_image(self, image: Image.Image, sample_id: str) -> Tuple[bool, List[str], float]:
        """
        Validate image component of triplet
        
        Args:
            image: PIL Image to validate
            sample_id: Sample identifier for error reporting
            
        Returns:
            Tuple of (is_valid, errors, quality_score)
        """
        errors = []
        quality_score = 0.0
        
        try:
            # Check image exists and is valid
            if image is None:
                errors.append("Image is None")
                return False, errors, 0.0
            
            # Check image dimensions
            width, height = image.size
            min_w, min_h = self.config.min_image_size
            max_w, max_h = self.config.max_image_size
            
            if width < min_w or height < min_h:
                errors.append(f"Image too small: {width}x{height} (minimum {min_w}x{min_h})")
            
            if width > max_w or height > max_h:
                errors.append(f"Image too large: {width}x{height} (maximum {max_w}x{max_h})")
            
            # Check image format and mode
            if image.mode not in ['RGB', 'RGBA', 'L']:
                errors.append(f"Unsupported image mode: {image.mode}")
            
            # Calculate quality score based on resolution and aspect ratio
            resolution_score = min(1.0, (width * height) / (512 * 512))
            aspect_ratio = width / height
            aspect_score = 1.0 - abs(aspect_ratio - 1.0) * 0.2  # Prefer square-ish images
            quality_score = (resolution_score + max(0, aspect_score)) / 2
            
            # Check for potential corruption
            try:
                # Try to access pixel data
                image.load()
                # Convert to array to check for valid data
                img_array = np.array(image)
                if img_array.size == 0:
                    errors.append("Image contains no pixel data")
                elif np.all(img_array == 0):
                    errors.append("Image appears to be completely black")
                elif len(np.unique(img_array)) < 10:
                    errors.append("Image has very low color diversity (possible corruption)")
            except Exception as e:
                errors.append(f"Image data corruption detected: {str(e)}")
            
        except Exception as e:
            errors.append(f"Image validation error: {str(e)}")
        
        return len(errors) == 0, errors, quality_score
    
    def validate_prompt(self, prompt: str, sample_id: str) -> Tuple[bool, List[str], float]:
        """
        Validate prompt/caption component of triplet
        
        Args:
            prompt: Text prompt to validate
            sample_id: Sample identifier for error reporting
            
        Returns:
            Tuple of (is_valid, errors, quality_score)
        """
        errors = []
        quality_score = 0.0
        
        try:
            # Check prompt exists and is string
            if prompt is None:
                errors.append("Prompt is None")
                return False, errors, 0.0
            
            if not isinstance(prompt, str):
                errors.append(f"Prompt is not a string: {type(prompt)}")
                return False, errors, 0.0
            
            # Clean and check prompt
            cleaned_prompt = prompt.strip()
            
            if not cleaned_prompt:
                errors.append("Prompt is empty after stripping whitespace")
                return False, errors, 0.0
            
            # Check length constraints
            prompt_length = len(cleaned_prompt)
            
            if prompt_length < self.config.min_caption_length:
                errors.append(f"Prompt too short: {prompt_length} characters "
                            f"(minimum {self.config.min_caption_length})")
            
            if prompt_length > self.config.max_caption_length:
                errors.append(f"Prompt too long: {prompt_length} characters "
                            f"(maximum {self.config.max_caption_length})")
            
            # Calculate quality score based on length and content diversity
            length_score = min(1.0, prompt_length / 50)  # Optimal around 50 characters
            
            # Check for content diversity (word count, unique words)
            words = cleaned_prompt.lower().split()
            word_count = len(words)
            unique_words = len(set(words))
            
            if word_count == 0:
                diversity_score = 0.0
            else:
                diversity_score = min(1.0, unique_words / word_count)
            
            # Check for descriptive content (presence of adjectives, nouns)
            descriptive_words = ['a', 'an', 'the', 'with', 'of', 'in', 'on', 'at', 'by']
            descriptive_count = sum(1 for word in words if word not in descriptive_words)
            descriptive_score = min(1.0, descriptive_count / max(1, word_count))
            
            quality_score = (length_score + diversity_score + descriptive_score) / 3
            
        except Exception as e:
            errors.append(f"Prompt validation error: {str(e)}")
        
        return len(errors) == 0, errors, quality_score
    
    def validate_condition_maps(self, 
                              image: Image.Image,
                              condition_maps: Dict[str, np.ndarray],
                              sample_id: str) -> Tuple[Dict[str, bool], Dict[str, List[str]], Dict[str, float]]:
        """
        Validate condition maps for the triplet
        
        Args:
            image: Original image for condition map validation
            condition_maps: Dictionary of condition type -> condition map
            sample_id: Sample identifier for error reporting
            
        Returns:
            Tuple of (validity_dict, errors_dict, quality_scores_dict)
        """
        validity = {}
        errors = {}
        quality_scores = {}
        
        image_width, image_height = image.size
        
        for condition_type in self.config.required_condition_types:
            condition_errors = []
            is_valid = True
            quality_score = 0.0
            
            try:
                if condition_type not in condition_maps:
                    condition_errors.append(f"Missing {condition_type} condition map")
                    is_valid = False
                else:
                    condition_map = condition_maps[condition_type]
                    
                    # Validate condition map format
                    if condition_map is None:
                        condition_errors.append(f"{condition_type} condition map is None")
                        is_valid = False
                    elif not isinstance(condition_map, np.ndarray):
                        condition_errors.append(f"{condition_type} condition map is not numpy array")
                        is_valid = False
                    else:
                        # Check dimensions
                        if len(condition_map.shape) != 3:
                            condition_errors.append(f"{condition_type} condition map has invalid shape: "
                                                  f"{condition_map.shape} (expected 3D)")
                            is_valid = False
                        else:
                            map_height, map_width, channels = condition_map.shape
                            
                            # Check if dimensions match image (with some tolerance)
                            if abs(map_width - image_width) > 10 or abs(map_height - image_height) > 10:
                                condition_errors.append(f"{condition_type} condition map size mismatch: "
                                                      f"{map_width}x{map_height} vs image {image_width}x{image_height}")
                            
                            # Check channel count
                            expected_channels = 1 if condition_type == "depth" else 3
                            if channels != expected_channels:
                                condition_errors.append(f"{condition_type} condition map has {channels} channels "
                                                      f"(expected {expected_channels})")
                            
                            # Check value range
                            min_val, max_val = condition_map.min(), condition_map.max()
                            if condition_type == "depth":
                                if min_val < -0.1 or max_val > 1.1:
                                    condition_errors.append(f"{condition_type} condition map values out of range: "
                                                          f"[{min_val:.3f}, {max_val:.3f}] (expected [0, 1])")
                            else:
                                if min_val < -10 or max_val > 265:
                                    condition_errors.append(f"{condition_type} condition map values out of range: "
                                                          f"[{min_val:.1f}, {max_val:.1f}] (expected [0, 255])")
                            
                            # Calculate quality score based on information content
                            if len(condition_errors) == 0:
                                # Check for information content (not all zeros/uniform)
                                if condition_type == "depth":
                                    unique_values = len(np.unique(condition_map))
                                    info_score = min(1.0, unique_values / 1000)  # More unique values = better
                                else:
                                    # For pose/edge maps, check for non-zero content
                                    non_zero_ratio = np.count_nonzero(condition_map) / condition_map.size
                                    info_score = min(1.0, non_zero_ratio * 2)  # Some content expected
                                
                                # Check for reasonable distribution
                                std_dev = np.std(condition_map)
                                distribution_score = min(1.0, std_dev / (max_val - min_val + 1e-6))
                                
                                quality_score = (info_score + distribution_score) / 2
            
            except Exception as e:
                condition_errors.append(f"{condition_type} condition map validation error: {str(e)}")
                is_valid = False
            
            validity[condition_type] = is_valid
            errors[condition_type] = condition_errors
            quality_scores[condition_type] = quality_score
        
        return validity, errors, quality_scores
    
    def validate_triplet(self, 
                        image: Image.Image,
                        prompt: str,
                        condition_maps: Dict[str, np.ndarray],
                        sample_id: str) -> TripletValidationResult:
        """
        Validate complete image-prompt-condition triplet
        
        Args:
            image: PIL Image
            prompt: Text prompt/caption
            condition_maps: Dictionary of condition maps
            sample_id: Sample identifier
            
        Returns:
            TripletValidationResult with comprehensive validation results
        """
        start_time = time.time()
        result = TripletValidationResult(sample_id=sample_id)
        
        try:
            # Validate image component
            image_valid, image_errors, image_quality = self.validate_image(image, sample_id)
            result.image_valid = image_valid
            result.image_quality_score = image_quality
            
            for error in image_errors:
                result.add_error("image", error)
            
            # Validate prompt component
            prompt_valid, prompt_errors, prompt_quality = self.validate_prompt(prompt, sample_id)
            result.prompt_valid = prompt_valid
            result.prompt_quality_score = prompt_quality
            
            for error in prompt_errors:
                result.add_error("prompt", error)
            
            # Validate condition maps
            if image_valid:  # Only validate condition maps if image is valid
                condition_validity, condition_errors, condition_quality = self.validate_condition_maps(
                    image, condition_maps, sample_id
                )
                
                result.condition_maps_valid = condition_validity
                result.condition_alignment_scores = condition_quality
                
                for condition_type, errors in condition_errors.items():
                    for error in errors:
                        result.add_error(f"condition_{condition_type}", error)
            
            # Overall validation result
            result.is_valid = (result.image_valid and 
                             result.prompt_valid and 
                             all(result.condition_maps_valid.values()))
            
        except Exception as e:
            result.add_error("validation", f"Unexpected error during validation: {str(e)}")
        
        result.validation_time_ms = (time.time() - start_time) * 1000
        return result


class DatasetVerifier:
    """
    Comprehensive dataset verification and quality assurance system
    
    This class provides end-to-end validation of ControlNet training datasets,
    including triplet validation, statistical analysis, and failure mode detection.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize dataset verifier
        
        Args:
            config: Validation configuration (uses defaults if None)
        """
        self.config = config or ValidationConfig()
        self.triplet_validator = TripletValidator(self.config)
        
        # Initialize condition extractors for re-extraction if needed
        self.extractors = {}
        if EXTRACTORS_AVAILABLE:
            if "depth" in self.config.required_condition_types:
                self.extractors["depth"] = DepthExtractor()
            if "pose" in self.config.required_condition_types:
                self.extractors["pose"] = PoseExtractor()
            if "edge" in self.config.required_condition_types:
                self.extractors["edge"] = EdgeExtractor()
        
        logger.info(f"DatasetVerifier initialized with {len(self.config.required_condition_types)} condition types")
    
    def verify_dataset_completeness(self, 
                                  samples: List[ProcessingSample],
                                  condition_maps_dir: Optional[Path] = None) -> DatasetValidationReport:
        """
        Verify dataset completeness and generate comprehensive report
        
        Args:
            samples: List of processing samples to verify
            condition_maps_dir: Directory containing pre-extracted condition maps
            
        Returns:
            DatasetValidationReport with detailed validation results
        """
        logger.info(f"Starting dataset verification for {len(samples)} samples")
        start_time = time.time()
        
        report = DatasetValidationReport()
        report.total_samples = len(samples)
        
        # Initialize counters
        validation_results = []
        failure_mode_counter = Counter()
        error_category_counter = defaultdict(list)
        
        # Progress tracking
        progress_bar = tqdm(samples, desc="Verifying dataset", unit="samples")
        
        for sample in progress_bar:
            try:
                # Load or extract condition maps
                condition_maps = self._load_or_extract_condition_maps(
                    sample, condition_maps_dir
                )
                
                # Validate triplet
                validation_result = self.triplet_validator.validate_triplet(
                    image=sample.image,
                    prompt=sample.caption,
                    condition_maps=condition_maps,
                    sample_id=sample.image_id
                )
                
                validation_results.append(validation_result)
                
                # Update report statistics
                if validation_result.is_valid:
                    report.valid_samples += 1
                else:
                    report.invalid_samples += 1
                    
                    # Categorize failure modes
                    for error in validation_result.errors:
                        component = error.split(':')[0] if ':' in error else 'unknown'
                        failure_mode_counter[component] += 1
                        error_category_counter[component].append(error)
                
                # Update progress
                current_success_rate = report.valid_samples / (report.valid_samples + report.invalid_samples)
                progress_bar.set_postfix({
                    'valid': report.valid_samples,
                    'success_rate': f"{current_success_rate:.1%}"
                })
                
            except Exception as e:
                logger.error(f"Error validating sample {sample.image_id}: {str(e)}")
                report.invalid_samples += 1
                failure_mode_counter['validation_error'] += 1
                error_category_counter['validation_error'].append(str(e))
        
        progress_bar.close()
        
        # Finalize report
        report.validation_time_seconds = time.time() - start_time
        report.failure_modes = dict(failure_mode_counter)
        report.error_categories = dict(error_category_counter)
        
        # Generate quality statistics
        report.quality_statistics = self._calculate_quality_statistics(validation_results)
        
        # Generate dataset statistics
        report.dataset_statistics = self._generate_dataset_statistics(samples, validation_results)
        
        report.finalize()
        
        logger.info(f"Dataset verification completed in {report.validation_time_seconds:.2f} seconds")
        logger.info(f"Success rate: {report.success_rate:.2%}")
        
        return report
    
    def _load_or_extract_condition_maps(self, 
                                      sample: ProcessingSample,
                                      condition_maps_dir: Optional[Path]) -> Dict[str, np.ndarray]:
        """
        Load existing condition maps or extract them on-the-fly
        
        Args:
            sample: Processing sample
            condition_maps_dir: Directory with pre-extracted condition maps
            
        Returns:
            Dictionary of condition type -> condition map
        """
        condition_maps = {}
        
        for condition_type in self.config.required_condition_types:
            try:
                # Try to load existing condition map
                if condition_maps_dir:
                    condition_map_path = condition_maps_dir / condition_type / f"{sample.image_id}.npy"
                    if condition_map_path.exists():
                        condition_maps[condition_type] = np.load(condition_map_path)
                        continue
                
                # Extract condition map on-the-fly
                if condition_type in self.extractors:
                    extractor = self.extractors[condition_type]
                    
                    if condition_type == "depth":
                        result = extractor.extract_depth(sample.image)
                        if result.success:
                            condition_maps[condition_type] = result.depth_map
                    elif condition_type == "pose":
                        result = extractor.extract_pose(sample.image)
                        if result.success:
                            condition_maps[condition_type] = result.pose_map
                    elif condition_type == "edge":
                        result = extractor.extract_edges(sample.image)
                        if result.success:
                            condition_maps[condition_type] = result.edge_map
                
            except Exception as e:
                logger.warning(f"Failed to load/extract {condition_type} condition map for {sample.image_id}: {str(e)}")
        
        return condition_maps
    
    def _calculate_quality_statistics(self, validation_results: List[TripletValidationResult]) -> Dict[str, Dict[str, float]]:
        """Calculate quality statistics from validation results"""
        if not validation_results:
            return {}
        
        # Extract quality scores
        image_scores = [r.image_quality_score for r in validation_results if r.image_quality_score > 0]
        prompt_scores = [r.prompt_quality_score for r in validation_results if r.prompt_quality_score > 0]
        
        condition_scores = defaultdict(list)
        for result in validation_results:
            for condition_type, score in result.condition_alignment_scores.items():
                if score > 0:
                    condition_scores[condition_type].append(score)
        
        statistics_dict = {}
        
        # Image quality statistics
        if image_scores:
            statistics_dict['image_quality'] = {
                'mean': statistics.mean(image_scores),
                'median': statistics.median(image_scores),
                'std': statistics.stdev(image_scores) if len(image_scores) > 1 else 0.0,
                'min': min(image_scores),
                'max': max(image_scores)
            }
        
        # Prompt quality statistics
        if prompt_scores:
            statistics_dict['prompt_quality'] = {
                'mean': statistics.mean(prompt_scores),
                'median': statistics.median(prompt_scores),
                'std': statistics.stdev(prompt_scores) if len(prompt_scores) > 1 else 0.0,
                'min': min(prompt_scores),
                'max': max(prompt_scores)
            }
        
        # Condition quality statistics
        for condition_type, scores in condition_scores.items():
            if scores:
                statistics_dict[f'{condition_type}_quality'] = {
                    'mean': statistics.mean(scores),
                    'median': statistics.median(scores),
                    'std': statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    'min': min(scores),
                    'max': max(scores)
                }
        
        return statistics_dict
    
    def _generate_dataset_statistics(self, 
                                   samples: List[ProcessingSample],
                                   validation_results: List[TripletValidationResult]) -> Dict[str, Any]:
        """Generate comprehensive dataset statistics"""
        if not samples:
            return {}
        
        # Image statistics
        image_widths = []
        image_heights = []
        image_aspects = []
        
        # Caption statistics
        caption_lengths = []
        word_counts = []
        
        for sample in samples:
            # Image stats
            width, height = sample.image.size
            image_widths.append(width)
            image_heights.append(height)
            image_aspects.append(width / height)
            
            # Caption stats
            caption_lengths.append(len(sample.caption))
            word_counts.append(len(sample.caption.split()))
        
        # Validation statistics
        validation_times = [r.validation_time_ms for r in validation_results]
        
        return {
            'sample_count': len(samples),
            'image_dimensions': {
                'width': {
                    'min': min(image_widths),
                    'max': max(image_widths),
                    'mean': statistics.mean(image_widths),
                    'median': statistics.median(image_widths)
                },
                'height': {
                    'min': min(image_heights),
                    'max': max(image_heights),
                    'mean': statistics.mean(image_heights),
                    'median': statistics.median(image_heights)
                },
                'aspect_ratio': {
                    'min': min(image_aspects),
                    'max': max(image_aspects),
                    'mean': statistics.mean(image_aspects),
                    'median': statistics.median(image_aspects)
                }
            },
            'caption_statistics': {
                'character_length': {
                    'min': min(caption_lengths),
                    'max': max(caption_lengths),
                    'mean': statistics.mean(caption_lengths),
                    'median': statistics.median(caption_lengths)
                },
                'word_count': {
                    'min': min(word_counts),
                    'max': max(word_counts),
                    'mean': statistics.mean(word_counts),
                    'median': statistics.median(word_counts)
                }
            },
            'validation_performance': {
                'avg_validation_time_ms': statistics.mean(validation_times) if validation_times else 0,
                'total_validation_time_seconds': sum(validation_times) / 1000 if validation_times else 0
            }
        }
    
    def generate_failure_analysis_report(self, report: DatasetValidationReport) -> Dict[str, Any]:
        """
        Generate detailed failure mode analysis
        
        Args:
            report: Dataset validation report
            
        Returns:
            Dictionary containing failure analysis
        """
        analysis = {
            'summary': {
                'total_failures': report.invalid_samples,
                'failure_rate': 1.0 - report.success_rate,
                'most_common_failures': []
            },
            'failure_breakdown': {},
            'recommendations': []
        }
        
        # Analyze failure modes
        if report.failure_modes:
            # Sort failure modes by frequency
            sorted_failures = sorted(report.failure_modes.items(), key=lambda x: x[1], reverse=True)
            analysis['summary']['most_common_failures'] = sorted_failures[:5]
            
            # Detailed breakdown
            for failure_mode, count in sorted_failures:
                percentage = (count / report.total_samples) * 100
                analysis['failure_breakdown'][failure_mode] = {
                    'count': count,
                    'percentage': percentage,
                    'sample_errors': report.error_categories.get(failure_mode, [])[:10]  # First 10 examples
                }
        
        # Generate recommendations based on failure patterns
        recommendations = []
        
        if 'image' in report.failure_modes:
            image_failures = report.failure_modes['image']
            if image_failures > report.total_samples * 0.1:
                recommendations.append(
                    f"High image failure rate ({image_failures} samples). "
                    "Consider improving image preprocessing or filtering criteria."
                )
        
        if 'prompt' in report.failure_modes:
            prompt_failures = report.failure_modes['prompt']
            if prompt_failures > report.total_samples * 0.05:
                recommendations.append(
                    f"Significant prompt issues ({prompt_failures} samples). "
                    "Review caption quality and length requirements."
                )
        
        for condition_type in ['depth', 'pose', 'edge']:
            condition_key = f'condition_{condition_type}'
            if condition_key in report.failure_modes:
                condition_failures = report.failure_modes[condition_key]
                if condition_failures > report.total_samples * 0.15:
                    recommendations.append(
                        f"High {condition_type} condition map failure rate ({condition_failures} samples). "
                        f"Check {condition_type} extraction pipeline and model performance."
                    )
        
        if report.success_rate < 0.8:
            recommendations.append(
                "Overall success rate below 80%. Consider reviewing data collection "
                "and preprocessing pipelines before training."
            )
        
        analysis['recommendations'] = recommendations
        
        return analysis
    
    def save_validation_report(self, 
                             report: DatasetValidationReport,
                             output_path: Union[str, Path],
                             include_visualizations: bool = True) -> None:
        """
        Save comprehensive validation report to disk
        
        Args:
            report: Dataset validation report
            output_path: Path to save the report
            include_visualizations: Whether to generate and save visualizations
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate failure analysis
        failure_analysis = self.generate_failure_analysis_report(report)
        
        # Create comprehensive report
        full_report = {
            'validation_summary': {
                'total_samples': report.total_samples,
                'valid_samples': report.valid_samples,
                'invalid_samples': report.invalid_samples,
                'success_rate': report.success_rate,
                'is_acceptable': report.is_acceptable,
                'validation_time_seconds': report.validation_time_seconds,
                'samples_per_second': report.samples_per_second
            },
            'quality_statistics': report.quality_statistics,
            'dataset_statistics': report.dataset_statistics,
            'failure_analysis': failure_analysis,
            'detailed_failure_modes': report.failure_modes,
            'error_categories': {k: v[:20] for k, v in report.error_categories.items()}  # Limit examples
        }
        
        # Save JSON report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Validation report saved to {output_path}")
        
        # Generate visualizations if requested
        if include_visualizations:
            self._generate_validation_visualizations(report, output_path.parent)
    
    def _generate_validation_visualizations(self, 
                                          report: DatasetValidationReport,
                                          output_dir: Path) -> None:
        """Generate visualization plots for validation report"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create visualizations directory
            viz_dir = output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # 1. Success rate pie chart
            fig, ax = plt.subplots(figsize=(8, 6))
            labels = ['Valid Samples', 'Invalid Samples']
            sizes = [report.valid_samples, report.invalid_samples]
            colors = ['#2ecc71', '#e74c3c']
            
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'Dataset Validation Results\n({report.total_samples} total samples)')
            plt.tight_layout()
            plt.savefig(viz_dir / "validation_summary.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Failure modes bar chart
            if report.failure_modes:
                fig, ax = plt.subplots(figsize=(12, 6))
                failure_items = list(report.failure_modes.items())
                failure_items.sort(key=lambda x: x[1], reverse=True)
                
                modes, counts = zip(*failure_items[:10])  # Top 10 failure modes
                
                bars = ax.bar(range(len(modes)), counts)
                ax.set_xlabel('Failure Mode')
                ax.set_ylabel('Number of Failures')
                ax.set_title('Top Failure Modes')
                ax.set_xticks(range(len(modes)))
                ax.set_xticklabels(modes, rotation=45, ha='right')
                
                # Add value labels on bars
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{count}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(viz_dir / "failure_modes.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # 3. Quality score distributions
            if report.quality_statistics:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.flatten()
                
                quality_types = ['image_quality', 'prompt_quality']
                for condition_type in ['depth', 'pose', 'edge']:
                    condition_key = f'{condition_type}_quality'
                    if condition_key in report.quality_statistics:
                        quality_types.append(condition_key)
                
                for i, quality_type in enumerate(quality_types[:4]):
                    if quality_type in report.quality_statistics:
                        stats = report.quality_statistics[quality_type]
                        
                        # Create histogram (simulated from statistics)
                        mean_val = stats['mean']
                        std_val = stats['std']
                        
                        # Generate sample data for visualization
                        sample_data = np.random.normal(mean_val, std_val, 1000)
                        sample_data = np.clip(sample_data, 0, 1)  # Clip to valid range
                        
                        axes[i].hist(sample_data, bins=30, alpha=0.7, edgecolor='black')
                        axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
                        axes[i].set_xlabel('Quality Score')
                        axes[i].set_ylabel('Frequency')
                        axes[i].set_title(f'{quality_type.replace("_", " ").title()} Distribution')
                        axes[i].legend()
                
                # Hide unused subplots
                for i in range(len(quality_types), 4):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                plt.savefig(viz_dir / "quality_distributions.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            logger.info(f"Validation visualizations saved to {viz_dir}")
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available. Skipping visualizations.")
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")


# Utility functions for dataset verification

def verify_controlnet_dataset(samples: List[ProcessingSample],
                            condition_maps_dir: Optional[Path] = None,
                            config: Optional[ValidationConfig] = None,
                            output_dir: Optional[Path] = None) -> DatasetValidationReport:
    """
    Convenience function to verify ControlNet training dataset
    
    Args:
        samples: List of processing samples to verify
        condition_maps_dir: Directory containing pre-extracted condition maps
        config: Validation configuration (uses defaults if None)
        output_dir: Directory to save validation report (optional)
        
    Returns:
        DatasetValidationReport with comprehensive validation results
    """
    verifier = DatasetVerifier(config)
    
    # Perform verification
    report = verifier.verify_dataset_completeness(samples, condition_maps_dir)
    
    # Save report if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = output_dir / "dataset_validation_report.json"
        verifier.save_validation_report(report, report_path, include_visualizations=True)
        
        # Also save a summary text report
        summary_path = output_dir / "validation_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"ControlNet Dataset Validation Summary\n")
            f.write(f"=====================================\n\n")
            f.write(f"Total Samples: {report.total_samples}\n")
            f.write(f"Valid Samples: {report.valid_samples}\n")
            f.write(f"Invalid Samples: {report.invalid_samples}\n")
            f.write(f"Success Rate: {report.success_rate:.2%}\n")
            f.write(f"Dataset Acceptable: {'Yes' if report.is_acceptable else 'No'}\n")
            f.write(f"Validation Time: {report.validation_time_seconds:.2f} seconds\n")
            f.write(f"Processing Speed: {report.samples_per_second:.1f} samples/second\n\n")
            
            if report.failure_modes:
                f.write("Top Failure Modes:\n")
                sorted_failures = sorted(report.failure_modes.items(), key=lambda x: x[1], reverse=True)
                for mode, count in sorted_failures[:5]:
                    percentage = (count / report.total_samples) * 100
                    f.write(f"  - {mode}: {count} samples ({percentage:.1f}%)\n")
        
        logger.info(f"Validation summary saved to {summary_path}")
    
    return report


if __name__ == "__main__":
    # Example usage and testing
    import argparse
    from .dataset_processor import create_coco_dataset_for_training
    
    parser = argparse.ArgumentParser(description="Dataset Verification and Quality Assurance")
    parser.add_argument("--dataset-path", type=str, help="Path to processed dataset JSON file")
    parser.add_argument("--condition-maps-dir", type=str, help="Directory containing condition maps")
    parser.add_argument("--output-dir", type=str, default="./validation_results", help="Output directory for reports")
    parser.add_argument("--subset-size", type=int, default=1000, help="Number of samples to verify (for testing)")
    parser.add_argument("--condition-types", nargs='+', default=["depth", "pose", "edge"], 
                       help="Condition types to validate")
    
    args = parser.parse_args()
    
    # Load or create dataset
    if args.dataset_path:
        # Load existing dataset (implementation would depend on saved format)
        logger.info(f"Loading dataset from {args.dataset_path}")
        # samples = load_processed_dataset(args.dataset_path)
        samples = []  # Placeholder
    else:
        # Create test dataset
        logger.info(f"Creating test dataset with {args.subset_size} samples")
        train_samples, val_samples, _ = create_coco_dataset_for_training(
            subset_size=args.subset_size,
            val_ratio=0.1
        )
        samples = train_samples + val_samples
    
    if samples:
        # Configure validation
        config = ValidationConfig(
            required_condition_types=args.condition_types,
            enable_visual_validation=True
        )
        
        # Verify dataset
        condition_maps_dir = Path(args.condition_maps_dir) if args.condition_maps_dir else None
        output_dir = Path(args.output_dir)
        
        report = verify_controlnet_dataset(
            samples=samples,
            condition_maps_dir=condition_maps_dir,
            config=config,
            output_dir=output_dir
        )
        
        # Print summary
        print(f"\nDataset Verification Results:")
        print(f"============================")
        print(f"Total samples: {report.total_samples}")
        print(f"Valid samples: {report.valid_samples}")
        print(f"Success rate: {report.success_rate:.2%}")
        print(f"Dataset acceptable: {'Yes' if report.is_acceptable else 'No'}")
        
        if report.failure_modes:
            print(f"\nTop failure modes:")
            sorted_failures = sorted(report.failure_modes.items(), key=lambda x: x[1], reverse=True)
            for mode, count in sorted_failures[:3]:
                percentage = (count / report.total_samples) * 100
                print(f"  - {mode}: {count} samples ({percentage:.1f}%)")
        
        print(f"\nDetailed report saved to: {output_dir}")
    else:
        print("No samples to verify. Please provide a dataset path or check dataset creation.")