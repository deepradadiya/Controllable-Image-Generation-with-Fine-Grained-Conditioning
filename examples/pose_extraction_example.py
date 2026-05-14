#!/usr/bin/env python3
"""
Pose Extraction Example and Testing Script

This script demonstrates the pose extraction functionality and provides
comprehensive testing of both DWPose and MediaPipe extractors.

Usage:
    python examples/pose_extraction_example.py --input path/to/image.jpg
    python examples/pose_extraction_example.py --input path/to/image.jpg --speed-critical
    python examples/pose_extraction_example.py --batch path/to/images/
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Union
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from PIL import Image
    import numpy as np
    from src.data.extract_pose import (
        PoseExtractor, 
        create_pose_extractor,
        extract_pose_from_image,
        save_pose_visualization
    )
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install required packages: pip install -r requirements.txt")
    DEPENDENCIES_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_image(width: int = 512, height: int = 512) -> Image.Image:
    """Create a simple test image with a stick figure"""
    # Create a white background
    image = Image.new('RGB', (width, height), 'white')
    
    # For a real test, you would load an actual image with a person
    # This is just a placeholder for testing the pipeline
    return image


def test_single_image(image_path: Union[str, Path], 
                     output_dir: Union[str, Path],
                     speed_critical: bool = False) -> bool:
    """
    Test pose extraction on a single image
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save results
        speed_critical: Whether to use speed-critical mode
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load image
        if isinstance(image_path, str) and image_path == "test":
            # Create test image
            image = create_test_image()
            image_name = "test_image"
        else:
            image = Image.open(image_path)
            image_name = Path(image_path).stem
        
        logger.info(f"Processing image: {image_name} ({image.size})")
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract pose
        start_time = time.time()
        pose_map = extract_pose_from_image(image, speed_critical=speed_critical)
        extraction_time = time.time() - start_time
        
        logger.info(f"Pose extraction completed in {extraction_time:.2f} seconds")
        logger.info(f"Output shape: {pose_map.shape}, dtype: {pose_map.dtype}")
        
        # Save results
        output_path = output_dir / f"{image_name}_pose.png"
        pose_image = Image.fromarray(pose_map)
        pose_image.save(output_path)
        
        # Save side-by-side comparison
        comparison_path = output_dir / f"{image_name}_comparison.png"
        create_comparison_image(image, pose_image, comparison_path)
        
        logger.info(f"Results saved to {output_dir}")
        
        # Validate output
        extractor = create_pose_extractor(speed_critical=speed_critical)
        is_valid = extractor.validate_output(pose_map)
        logger.info(f"Output validation: {'PASSED' if is_valid else 'FAILED'}")
        
        return is_valid
        
    except Exception as e:
        logger.error(f"Failed to process image {image_path}: {e}")
        return False


def create_comparison_image(original: Image.Image, 
                          pose: Image.Image, 
                          output_path: Path) -> None:
    """Create side-by-side comparison image"""
    # Resize images to same height
    height = min(original.height, pose.height, 512)
    aspect_ratio_orig = original.width / original.height
    aspect_ratio_pose = pose.width / pose.height
    
    orig_width = int(height * aspect_ratio_orig)
    pose_width = int(height * aspect_ratio_pose)
    
    original_resized = original.resize((orig_width, height), Image.Resampling.LANCZOS)
    pose_resized = pose.resize((pose_width, height), Image.Resampling.LANCZOS)
    
    # Create comparison image
    total_width = orig_width + pose_width + 10  # 10px gap
    comparison = Image.new('RGB', (total_width, height), 'white')
    
    # Paste images
    comparison.paste(original_resized, (0, 0))
    comparison.paste(pose_resized, (orig_width + 10, 0))
    
    # Save
    comparison.save(output_path)
    logger.info(f"Comparison image saved to {output_path}")


def test_batch_processing(input_dir: Union[str, Path],
                         output_dir: Union[str, Path],
                         speed_critical: bool = False) -> None:
    """
    Test batch processing of multiple images
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save results
        speed_critical: Whether to use speed-critical mode
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [
        f for f in input_dir.iterdir() 
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        logger.warning(f"No image files found in {input_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Load images
    images = []
    for image_file in image_files:
        try:
            image = Image.open(image_file)
            images.append((image, image_file.stem))
        except Exception as e:
            logger.error(f"Failed to load {image_file}: {e}")
    
    if not images:
        logger.error("No valid images loaded")
        return
    
    # Create extractor
    extractor = create_pose_extractor(speed_critical=speed_critical)
    
    # Process batch
    start_time = time.time()
    image_arrays = [img for img, _ in images]
    pose_maps = extractor.batch_extract(image_arrays, show_progress=True)
    total_time = time.time() - start_time
    
    logger.info(f"Batch processing completed in {total_time:.2f} seconds")
    logger.info(f"Average time per image: {total_time/len(images):.2f} seconds")
    
    # Save results
    for (original_image, name), pose_map in zip(images, pose_maps):
        try:
            # Save pose map
            pose_path = output_dir / f"{name}_pose.png"
            pose_image = Image.fromarray(pose_map)
            pose_image.save(pose_path)
            
            # Save comparison
            comparison_path = output_dir / f"{name}_comparison.png"
            create_comparison_image(original_image, pose_image, comparison_path)
            
        except Exception as e:
            logger.error(f"Failed to save results for {name}: {e}")
    
    logger.info(f"Batch results saved to {output_dir}")


def test_extractor_availability() -> None:
    """Test which extractors are available"""
    logger.info("Testing extractor availability...")
    
    try:
        extractor = create_pose_extractor(speed_critical=False)
        logger.info(f"DWPose available: {extractor.dwpose_extractor is not None}")
        logger.info(f"MediaPipe available: {extractor.mediapipe_extractor is not None}")
        
        # Test with a simple image
        test_image = create_test_image(256, 256)
        
        # Test DWPose if available
        if extractor.dwpose_extractor is not None:
            try:
                start_time = time.time()
                poses = extractor.dwpose_extractor.extract_pose(test_image)
                dwpose_time = time.time() - start_time
                logger.info(f"DWPose test: {len(poses)} poses detected in {dwpose_time:.2f}s")
            except Exception as e:
                logger.error(f"DWPose test failed: {e}")
        
        # Test MediaPipe if available
        if extractor.mediapipe_extractor is not None:
            try:
                start_time = time.time()
                poses = extractor.mediapipe_extractor.extract_pose(test_image)
                mediapipe_time = time.time() - start_time
                logger.info(f"MediaPipe test: {len(poses)} poses detected in {mediapipe_time:.2f}s")
            except Exception as e:
                logger.error(f"MediaPipe test failed: {e}")
        
    except Exception as e:
        logger.error(f"Extractor availability test failed: {e}")


def main():
    """Main function"""
    if not DEPENDENCIES_AVAILABLE:
        return 1
    
    parser = argparse.ArgumentParser(description="Pose Extraction Example and Testing")
    parser.add_argument("--input", type=str, help="Input image path or 'test' for test image")
    parser.add_argument("--batch", type=str, help="Input directory for batch processing")
    parser.add_argument("--output", type=str, default="outputs/pose_demo", 
                       help="Output directory (default: outputs/pose_demo)")
    parser.add_argument("--speed-critical", action="store_true", 
                       help="Use MediaPipe for speed-critical scenarios")
    parser.add_argument("--test-availability", action="store_true",
                       help="Test which extractors are available")
    
    args = parser.parse_args()
    
    # Test extractor availability
    if args.test_availability:
        test_extractor_availability()
        return 0
    
    # Single image processing
    if args.input:
        success = test_single_image(
            args.input, 
            args.output, 
            speed_critical=args.speed_critical
        )
        return 0 if success else 1
    
    # Batch processing
    if args.batch:
        test_batch_processing(
            args.batch,
            args.output,
            speed_critical=args.speed_critical
        )
        return 0
    
    # Default: test with generated image
    logger.info("No input specified, running test with generated image")
    success = test_single_image(
        "test",
        args.output,
        speed_critical=args.speed_critical
    )
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())