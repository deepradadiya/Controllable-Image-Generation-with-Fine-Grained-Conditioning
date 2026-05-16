"""
FID Score Computation Example

This example demonstrates how to use the FID calculator to evaluate
the quality of generated images against real images.

Usage:
    python examples/fid_computation_example.py
"""

import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluation.compute_fid import FIDCalculator, compute_fid_from_paths


def create_sample_images():
    """Create sample real and generated images for demonstration"""
    temp_dir = tempfile.mkdtemp()
    real_dir = Path(temp_dir) / "real_images"
    generated_dir = Path(temp_dir) / "generated_images"
    
    real_dir.mkdir()
    generated_dir.mkdir()
    
    print(f"Creating sample images in: {temp_dir}")
    
    # Create "real" images with more consistent patterns
    print("Creating real images...")
    for i in range(20):
        # Create images with natural-looking patterns
        img_array = np.random.randint(80, 120, (256, 256, 3), dtype=np.uint8)
        # Add some structure
        img_array[100:156, 100:156] = [150, 180, 200]  # Add a consistent square
        img = Image.fromarray(img_array)
        img.save(real_dir / f"real_{i:03d}.png")
    
    # Create "generated" images with different characteristics
    print("Creating generated images...")
    for i in range(20):
        # Create images with different patterns (simulating generated images)
        img_array = np.random.randint(60, 140, (256, 256, 3), dtype=np.uint8)
        # Add different structure
        img_array[80:176, 80:176] = [100, 150, 180]  # Different square
        img = Image.fromarray(img_array)
        img.save(generated_dir / f"generated_{i:03d}.png")
    
    return temp_dir, real_dir, generated_dir


def example_basic_fid_computation():
    """Example of basic FID computation"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic FID Computation")
    print("="*60)
    
    # Create sample data
    temp_dir, real_dir, generated_dir = create_sample_images()
    
    try:
        # Method 1: Using the convenience function
        print("\nMethod 1: Using compute_fid_from_paths()")
        results = compute_fid_from_paths(
            real_image_dir=real_dir,
            generated_image_dir=generated_dir,
            batch_size=4,
            compute_confidence_interval=False  # Skip CI for speed
        )
        
        print(f"FID Score: {results.fid_score:.3f}")
        print(f"Real samples: {results.num_real_samples}")
        print(f"Generated samples: {results.num_generated_samples}")
        print(f"Computation time: {results.computation_time_seconds:.2f}s")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def example_fid_with_confidence_interval():
    """Example of FID computation with statistical analysis"""
    print("\n" + "="*60)
    print("EXAMPLE 2: FID with Confidence Interval")
    print("="*60)
    
    # Create sample data
    temp_dir, real_dir, generated_dir = create_sample_images()
    
    try:
        # Method 2: Using FIDCalculator class directly
        print("\nMethod 2: Using FIDCalculator class with confidence interval")
        calculator = FIDCalculator(batch_size=4)
        
        # Get image paths
        real_images = list(real_dir.glob("*.png"))
        generated_images = list(generated_dir.glob("*.png"))
        
        # Convert to strings
        real_images = [str(p) for p in real_images]
        generated_images = [str(p) for p in generated_images]
        
        # Compute FID with confidence interval
        results = calculator.compute_fid(
            real_images=real_images,
            generated_images=generated_images,
            compute_confidence_interval=True,
            confidence_level=0.95,
            num_bootstrap=500  # Reduced for faster computation
        )
        
        print(f"FID Score: {results.fid_score:.3f}")
        print(f"95% Confidence Interval: [{results.confidence_interval[0]:.3f}, {results.confidence_interval[1]:.3f}]")
        print(f"Real samples: {results.num_real_samples}")
        print(f"Generated samples: {results.num_generated_samples}")
        print(f"Computation time: {results.computation_time_seconds:.2f}s")
        
        # Statistical interpretation
        ci_width = results.confidence_interval[1] - results.confidence_interval[0]
        print(f"\nStatistical Analysis:")
        print(f"Confidence interval width: {ci_width:.3f}")
        if ci_width < 1.0:
            print("✓ Narrow confidence interval indicates reliable estimate")
        else:
            print("⚠ Wide confidence interval suggests need for more samples")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def example_batch_processing():
    """Example demonstrating batch processing capabilities"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Processing Demonstration")
    print("="*60)
    
    # Create larger dataset
    temp_dir = tempfile.mkdtemp()
    real_dir = Path(temp_dir) / "real_images"
    generated_dir = Path(temp_dir) / "generated_images"
    
    real_dir.mkdir()
    generated_dir.mkdir()
    
    print("Creating larger dataset for batch processing demo...")
    
    # Create more images to demonstrate batch processing
    for i in range(50):
        # Real images
        img_array = np.random.randint(70, 130, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(real_dir / f"real_{i:03d}.png")
        
        # Generated images
        img_array = np.random.randint(50, 150, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(generated_dir / f"generated_{i:03d}.png")
    
    try:
        # Test different batch sizes
        batch_sizes = [4, 8, 16]
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            
            import time
            start_time = time.time()
            
            results = compute_fid_from_paths(
                real_image_dir=real_dir,
                generated_image_dir=generated_dir,
                batch_size=batch_size,
                compute_confidence_interval=False
            )
            
            elapsed_time = time.time() - start_time
            
            print(f"  FID Score: {results.fid_score:.3f}")
            print(f"  Processing time: {elapsed_time:.2f}s")
            print(f"  Images per second: {(results.num_real_samples + results.num_generated_samples) / elapsed_time:.1f}")
    
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


def example_memory_efficient_processing():
    """Example of memory-efficient processing for large datasets"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Memory-Efficient Processing")
    print("="*60)
    
    print("This example demonstrates memory-efficient processing techniques:")
    print("1. Batch processing to control memory usage")
    print("2. Feature extraction without storing all images in memory")
    print("3. Statistical computation on extracted features")
    
    # Create sample images in memory (simulating large dataset)
    print("\nCreating sample images in memory...")
    real_images = []
    generated_images = []
    
    for i in range(30):
        # Create PIL images directly in memory
        real_img = Image.new('RGB', (256, 256), color=(100 + i, 150 + i, 200 + i))
        generated_img = Image.new('RGB', (256, 256), color=(80 + i*2, 120 + i*2, 180 + i*2))
        
        real_images.append(real_img)
        generated_images.append(generated_img)
    
    # Initialize calculator with memory-efficient settings
    calculator = FIDCalculator(
        batch_size=8,  # Smaller batch size for memory efficiency
        num_workers=2   # Limit parallel processing
    )
    
    print(f"Processing {len(real_images)} real and {len(generated_images)} generated images...")
    
    # Compute FID
    results = calculator.compute_fid(
        real_images=real_images,
        generated_images=generated_images,
        compute_confidence_interval=True,
        num_bootstrap=200  # Reduced for faster computation
    )
    
    print(f"\nResults:")
    print(f"FID Score: {results.fid_score:.3f}")
    print(f"95% CI: [{results.confidence_interval[0]:.3f}, {results.confidence_interval[1]:.3f}]")
    print(f"Processing time: {results.computation_time_seconds:.2f}s")
    
    # Memory usage tips
    print(f"\nMemory Efficiency Tips:")
    print(f"✓ Used batch_size={calculator.batch_size} to control GPU memory")
    print(f"✓ Processed images directly from memory without disk I/O")
    print(f"✓ Used reduced bootstrap samples for faster CI computation")


def main():
    """Run all examples"""
    print("FID Score Computation Examples")
    print("=" * 60)
    print("This script demonstrates various ways to compute FID scores")
    print("for evaluating generated image quality.")
    
    try:
        # Run examples
        example_basic_fid_computation()
        example_fid_with_confidence_interval()
        example_batch_processing()
        example_memory_efficient_processing()
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60)
        
        print("\nKey Takeaways:")
        print("1. FID scores measure distribution similarity between real and generated images")
        print("2. Lower FID scores indicate better quality (more similar to real images)")
        print("3. Confidence intervals provide statistical reliability estimates")
        print("4. Batch processing enables efficient computation on large datasets")
        print("5. Memory-efficient techniques allow processing within GPU constraints")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()