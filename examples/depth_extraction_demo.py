#!/usr/bin/env python3
"""
Depth Extraction Demo Script

This script demonstrates the depth map extraction functionality using the Intel DPT model.
It shows how to use the DepthExtractor class for both single image and batch processing.

Usage:
    python examples/depth_extraction_demo.py --input path/to/image.jpg --output ./outputs/depth_demo/

Requirements:
    - All dependencies from requirements.txt installed
    - Input image(s) in supported formats (jpg, png, etc.)
"""

import argparse
import sys
from pathlib import Path
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from PIL import Image
    import numpy as np
    from data.extract_depth import (
        DepthExtractor, 
        DepthExtractionConfig, 
        extract_depth_from_images
    )
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)


def create_demo_images(output_dir: Path) -> list:
    """Create simple demo images for testing"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    demo_images = []
    
    # Create a simple gradient image
    gradient_img = Image.new('RGB', (512, 512))
    pixels = []
    for y in range(512):
        for x in range(512):
            # Create a radial gradient
            center_x, center_y = 256, 256
            distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
            intensity = max(0, min(255, int(255 - distance / 2)))
            pixels.append((intensity, intensity, intensity))
    
    gradient_img.putdata(pixels)
    gradient_path = output_dir / "gradient_demo.png"
    gradient_img.save(gradient_path)
    demo_images.append(gradient_path)
    
    # Create a simple geometric pattern
    pattern_img = Image.new('RGB', (512, 512), color='white')
    # Add some simple shapes for depth variation
    from PIL import ImageDraw
    draw = ImageDraw.Draw(pattern_img)
    
    # Draw concentric circles
    for i in range(5):
        radius = 50 + i * 40
        color = 255 - i * 40
        draw.ellipse([256-radius, 256-radius, 256+radius, 256+radius], 
                    fill=(color, color, color))
    
    pattern_path = output_dir / "pattern_demo.png"
    pattern_img.save(pattern_path)
    demo_images.append(pattern_path)
    
    print(f"Created {len(demo_images)} demo images in {output_dir}")
    return demo_images


def run_single_image_demo(image_path: Path, output_dir: Path):
    """Demonstrate single image depth extraction"""
    print(f"\n=== Single Image Demo ===")
    print(f"Input: {image_path}")
    
    # Configure for CPU to avoid GPU requirements in demo
    config = DepthExtractionConfig(
        device="cpu",  # Use CPU for demo compatibility
        batch_size=1,
        target_size=(512, 512),
        precision="fp32"  # Use FP32 for CPU
    )
    
    # Initialize extractor
    print("Initializing DepthExtractor...")
    extractor = DepthExtractor(config)
    
    # Load and process image
    print("Loading image...")
    image = Image.open(image_path).convert('RGB')
    print(f"Image size: {image.size}")
    
    # Extract depth
    print("Extracting depth map...")
    start_time = time.time()
    result = extractor.extract(image)
    end_time = time.time()
    
    # Display results
    if result.success:
        print(f"✓ Depth extraction successful!")
        print(f"  - Processing time: {result.processing_time_ms:.1f}ms")
        print(f"  - Depth map shape: {result.depth_map.shape}")
        print(f"  - Value range: [{result.depth_map.min():.3f}, {result.depth_map.max():.3f}]")
        print(f"  - Memory used: {result.memory_used_mb:.1f}MB")
        
        # Validate output
        is_valid, errors = result.validate()
        print(f"  - Validation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
        if errors:
            for error in errors:
                print(f"    - {error}")
        
        # Save depth map
        output_path = output_dir / f"{image_path.stem}_depth.png"
        extractor.save_depth_map(result.depth_map, output_path)
        print(f"  - Saved to: {output_path}")
        
    else:
        print(f"✗ Depth extraction failed: {result.error_message}")


def run_batch_demo(image_paths: list, output_dir: Path):
    """Demonstrate batch depth extraction"""
    print(f"\n=== Batch Processing Demo ===")
    print(f"Processing {len(image_paths)} images")
    
    # Configure for batch processing
    config = DepthExtractionConfig(
        device="cpu",  # Use CPU for demo compatibility
        batch_size=2,  # Small batch for demo
        target_size=(256, 256),  # Smaller size for faster processing
        precision="fp32"
    )
    
    # Use convenience function for batch processing
    print("Starting batch extraction...")
    start_time = time.time()
    
    results = extract_depth_from_images(
        image_paths=image_paths,
        output_dir=output_dir / "batch_results",
        config=config
    )
    
    end_time = time.time()
    
    # Display batch results
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    print(f"\n=== Batch Results ===")
    print(f"Total images: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(results):.1%}")
    print(f"Total time: {end_time - start_time:.1f}s")
    
    if successful > 0:
        avg_time = np.mean([r.processing_time_ms for r in results if r.success])
        print(f"Average processing time: {avg_time:.1f}ms per image")
    
    # Show individual results
    for i, (path, result) in enumerate(zip(image_paths, results)):
        status = "✓" if result.success else "✗"
        print(f"  {status} {path.name}: {result.processing_time_ms:.1f}ms")


def main():
    parser = argparse.ArgumentParser(description="Depth Extraction Demo")
    parser.add_argument("--input", type=str, help="Input image or directory")
    parser.add_argument("--output", type=str, default="./outputs/depth_demo", 
                       help="Output directory")
    parser.add_argument("--create-demo", action="store_true", 
                       help="Create demo images if no input provided")
    parser.add_argument("--batch-demo", action="store_true",
                       help="Run batch processing demo")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine input images
    if args.input:
        input_path = Path(args.input)
        if input_path.is_file():
            image_paths = [input_path]
        elif input_path.is_dir():
            # Find all image files
            extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            image_paths = [
                p for p in input_path.rglob('*') 
                if p.suffix.lower() in extensions
            ]
        else:
            print(f"Error: Input path does not exist: {input_path}")
            return
    elif args.create_demo:
        # Create demo images
        image_paths = create_demo_images(output_dir)
    else:
        print("Error: Please provide --input or use --create-demo")
        return
    
    if not image_paths:
        print("No images found to process")
        return
    
    print(f"Found {len(image_paths)} images to process")
    
    # Run demos
    if len(image_paths) == 1 and not args.batch_demo:
        # Single image demo
        run_single_image_demo(image_paths[0], output_dir)
    else:
        # Batch demo
        run_batch_demo(image_paths, output_dir)
    
    print(f"\nDemo completed! Check outputs in: {output_dir}")


if __name__ == "__main__":
    main()