"""
Quick FID Score Computation Demo

A fast demonstration of the FID calculator functionality.
"""

import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluation.compute_fid import FIDCalculator


def create_quick_demo():
    """Quick demo of FID computation"""
    print("Quick FID Computation Demo")
    print("=" * 40)
    
    # Create small set of test images in memory
    print("Creating test images...")
    
    real_images = []
    generated_images = []
    
    # Create 10 "real" images with consistent patterns
    for i in range(10):
        img = Image.new('RGB', (128, 128), color=(100, 150, 200))
        real_images.append(img)
    
    # Create 10 "generated" images with different patterns
    for i in range(10):
        img = Image.new('RGB', (128, 128), color=(80 + i*5, 120 + i*3, 180 + i*4))
        generated_images.append(img)
    
    # Initialize FID calculator
    calculator = FIDCalculator(batch_size=4, num_workers=0)
    
    print("Computing FID score...")
    
    # Compute FID without confidence interval for speed
    results = calculator.compute_fid(
        real_images=real_images,
        generated_images=generated_images,
        compute_confidence_interval=False,
        show_progress=True
    )
    
    print(f"\nResults:")
    print(f"FID Score: {results.fid_score:.3f}")
    print(f"Real samples: {results.num_real_samples}")
    print(f"Generated samples: {results.num_generated_samples}")
    print(f"Computation time: {results.computation_time_seconds:.2f}s")
    
    print(f"\nInterpretation:")
    if results.fid_score < 10:
        print("✓ Excellent quality (FID < 10)")
    elif results.fid_score < 50:
        print("✓ Good quality (FID < 50)")
    elif results.fid_score < 100:
        print("⚠ Moderate quality (FID < 100)")
    else:
        print("⚠ Poor quality (FID > 100)")
    
    print(f"\nDemo completed successfully!")


if __name__ == "__main__":
    create_quick_demo()