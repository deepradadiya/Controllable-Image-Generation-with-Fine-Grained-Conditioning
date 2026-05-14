#!/usr/bin/env python3
"""
Edge Extraction Demo

This script demonstrates the Canny edge extraction functionality implemented
for the ControlNet training pipeline. It shows how to extract edge maps from
images with different configurations and save the results.

Usage:
    python examples/edge_extraction_demo.py
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import cv2

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import directly from the module file to avoid package import issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "extract_edges", 
    Path(__file__).parent.parent / "src" / "data" / "extract_edges.py"
)
extract_edges = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract_edges)

# Import the classes and functions we need
CannyEdgeExtractor = extract_edges.CannyEdgeExtractor
EdgeExtractionConfig = extract_edges.EdgeExtractionConfig
extract_edges_from_image = extract_edges.extract_edges_from_image
save_edge_map = extract_edges.save_edge_map


def create_demo_image(size=(512, 512)) -> Image.Image:
    """Create a demo image with various shapes and edges"""
    # Create a white background
    image = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(image)
    
    # Draw various shapes with different colors
    # Rectangle
    draw.rectangle([50, 50, 200, 150], fill='red', outline='black', width=3)
    
    # Circle
    draw.ellipse([250, 50, 400, 200], fill='blue', outline='black', width=3)
    
    # Triangle (using polygon)
    triangle_points = [(100, 250), (200, 250), (150, 350)]
    draw.polygon(triangle_points, fill='green', outline='black', width=3)
    
    # Lines
    draw.line([(0, 400), (size[0], 400)], fill='purple', width=5)
    draw.line([(300, 0), (300, size[1])], fill='orange', width=5)
    
    # Some text
    try:
        draw.text((350, 300), "DEMO", fill='black')
    except:
        # If font loading fails, just skip text
        pass
    
    return image


def demonstrate_edge_extraction():
    """Demonstrate different edge extraction configurations"""
    print("🎨 ControlNet Edge Extraction Demo")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("outputs/edge_demo")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create demo image
    print("📸 Creating demo image...")
    demo_image = create_demo_image()
    demo_image.save(output_dir / "original_image.png")
    print(f"   Saved original image to {output_dir / 'original_image.png'}")
    
    # Configuration 1: Default adaptive thresholding
    print("\n🔧 Configuration 1: Default Adaptive Thresholding")
    config1 = EdgeExtractionConfig(
        adaptive_threshold=True,
        apply_gaussian_blur=True,
        apply_morphology=True,
        output_channels=3,
        normalize_output=True
    )
    
    extractor1 = CannyEdgeExtractor(config1)
    result1 = extractor1.extract(demo_image)
    
    print(f"   Success: {result1.success}")
    print(f"   Processing time: {result1.processing_time_ms:.1f} ms")
    print(f"   Edge density: {result1.edge_density:.3f}")
    print(f"   Connected components: {result1.connected_components}")
    print(f"   Thresholds: {result1.threshold_low:.1f} / {result1.threshold_high:.1f}")
    
    if result1.success:
        save_edge_map(result1.edge_map, output_dir / "edges_adaptive.png")
        print(f"   Saved to {output_dir / 'edges_adaptive.png'}")
    
    # Configuration 2: Fixed thresholding
    print("\n🔧 Configuration 2: Fixed Thresholding")
    config2 = EdgeExtractionConfig(
        adaptive_threshold=False,
        low_threshold=50.0,
        high_threshold=150.0,
        apply_gaussian_blur=True,
        apply_morphology=True,
        output_channels=3,
        normalize_output=True
    )
    
    extractor2 = CannyEdgeExtractor(config2)
    result2 = extractor2.extract(demo_image)
    
    print(f"   Success: {result2.success}")
    print(f"   Processing time: {result2.processing_time_ms:.1f} ms")
    print(f"   Edge density: {result2.edge_density:.3f}")
    print(f"   Connected components: {result2.connected_components}")
    print(f"   Thresholds: {result2.threshold_low:.1f} / {result2.threshold_high:.1f}")
    
    if result2.success:
        save_edge_map(result2.edge_map, output_dir / "edges_fixed.png")
        print(f"   Saved to {output_dir / 'edges_fixed.png'}")
    
    # Configuration 3: High sensitivity
    print("\n🔧 Configuration 3: High Sensitivity")
    config3 = EdgeExtractionConfig(
        adaptive_threshold=False,
        low_threshold=20.0,
        high_threshold=60.0,
        apply_gaussian_blur=False,  # No blur for more detail
        apply_morphology=False,     # No morphology for finer edges
        output_channels=1,          # Single channel output
        normalize_output=False      # Keep as uint8
    )
    
    extractor3 = CannyEdgeExtractor(config3)
    result3 = extractor3.extract(demo_image)
    
    print(f"   Success: {result3.success}")
    print(f"   Processing time: {result3.processing_time_ms:.1f} ms")
    print(f"   Edge density: {result3.edge_density:.3f}")
    print(f"   Connected components: {result3.connected_components}")
    print(f"   Thresholds: {result3.threshold_low:.1f} / {result3.threshold_high:.1f}")
    
    if result3.success:
        save_edge_map(result3.edge_map, output_dir / "edges_high_sensitivity.png")
        print(f"   Saved to {output_dir / 'edges_high_sensitivity.png'}")
    
    # Configuration 4: Low sensitivity (thick edges only)
    print("\n🔧 Configuration 4: Low Sensitivity")
    config4 = EdgeExtractionConfig(
        adaptive_threshold=False,
        low_threshold=100.0,
        high_threshold=200.0,
        apply_gaussian_blur=True,
        gaussian_blur_kernel=7,     # Larger blur kernel
        apply_morphology=True,
        morphology_kernel_size=5,   # Larger morphology kernel
        output_channels=3,
        normalize_output=True,
        invert_edges=True          # White edges on black background
    )
    
    extractor4 = CannyEdgeExtractor(config4)
    result4 = extractor4.extract(demo_image)
    
    print(f"   Success: {result4.success}")
    print(f"   Processing time: {result4.processing_time_ms:.1f} ms")
    print(f"   Edge density: {result4.edge_density:.3f}")
    print(f"   Connected components: {result4.connected_components}")
    print(f"   Thresholds: {result4.threshold_low:.1f} / {result4.threshold_high:.1f}")
    
    if result4.success:
        save_edge_map(result4.edge_map, output_dir / "edges_low_sensitivity_inverted.png")
        print(f"   Saved to {output_dir / 'edges_low_sensitivity_inverted.png'}")
    
    # Batch processing demo
    print("\n📦 Batch Processing Demo")
    
    # Create multiple test images
    test_images = []
    for i in range(3):
        # Create variations of the demo image
        img = create_demo_image()
        # Add some variation
        img_array = np.array(img)
        noise = np.random.randint(-20, 20, img_array.shape, dtype=np.int16)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        test_images.append(Image.fromarray(img_array))
    
    # Process batch
    batch_results = extractor1.extract_batch(test_images, show_progress=True)
    
    print(f"   Processed {len(batch_results)} images")
    successful = sum(1 for r in batch_results if r.success)
    print(f"   Successful extractions: {successful}/{len(batch_results)}")
    
    # Save batch results
    for i, result in enumerate(batch_results):
        if result.success:
            save_edge_map(result.edge_map, output_dir / f"batch_result_{i}.png")
    
    # Statistics
    print("\n📊 Extraction Statistics")
    stats = extractor1.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    print(f"\n✅ Demo completed! Check the results in {output_dir}")
    print("\nFiles created:")
    for file_path in sorted(output_dir.glob("*.png")):
        print(f"   - {file_path.name}")


def demonstrate_real_image_processing():
    """Demonstrate edge extraction on a real-world style image"""
    print("\n🌍 Real Image Processing Demo")
    print("=" * 50)
    
    # Create a more realistic test image
    size = (512, 512)
    image = Image.new('RGB', size, color=(240, 240, 240))  # Light gray background
    
    # Convert to numpy for OpenCV operations
    img_array = np.array(image)
    
    # Add some realistic features
    # Building-like rectangles
    cv2.rectangle(img_array, (50, 100), (200, 400), (100, 100, 100), -1)  # Dark building
    cv2.rectangle(img_array, (60, 110), (190, 200), (200, 200, 200), -1)  # Windows
    cv2.rectangle(img_array, (70, 120), (90, 140), (50, 50, 50), -1)      # Window frame
    cv2.rectangle(img_array, (100, 120), (120, 140), (50, 50, 50), -1)    # Window frame
    
    # Tree-like structure
    cv2.circle(img_array, (350, 200), 60, (34, 139, 34), -1)  # Tree crown
    cv2.rectangle(img_array, (340, 260), (360, 350), (139, 69, 19), -1)   # Tree trunk
    
    # Road
    cv2.rectangle(img_array, (0, 400), (512, 450), (64, 64, 64), -1)      # Road
    cv2.line(img_array, (0, 425), (512, 425), (255, 255, 255), 2)         # Road marking
    
    # Sky gradient (simple)
    for y in range(100):
        color = int(200 + y * 0.5)  # Gradient from light to darker
        cv2.line(img_array, (0, y), (512, y), (color, color, 255), 1)
    
    realistic_image = Image.fromarray(img_array)
    
    # Save the realistic image
    output_dir = Path("outputs/edge_demo")
    realistic_image.save(output_dir / "realistic_image.png")
    
    # Extract edges with optimal settings for realistic images
    config = EdgeExtractionConfig(
        adaptive_threshold=True,
        apply_gaussian_blur=True,
        gaussian_blur_kernel=3,
        apply_morphology=True,
        morphology_kernel_size=3,
        output_channels=3,
        normalize_output=True,
        min_edge_density=0.001,  # Lower threshold for realistic images
        min_connected_components=1
    )
    
    result = extract_edges_from_image(realistic_image, config)
    
    print(f"   Success: {result.success}")
    print(f"   Processing time: {result.processing_time_ms:.1f} ms")
    print(f"   Edge density: {result.edge_density:.3f}")
    print(f"   Connected components: {result.connected_components}")
    print(f"   Thresholds: {result.threshold_low:.1f} / {result.threshold_high:.1f}")
    
    if result.success:
        save_edge_map(result.edge_map, output_dir / "realistic_edges.png")
        print(f"   Saved to {output_dir / 'realistic_edges.png'}")
    else:
        print(f"   Error: {result.error_message}")


if __name__ == "__main__":
    try:
        demonstrate_edge_extraction()
        demonstrate_real_image_processing()
        
        print("\n🎉 All demos completed successfully!")
        print("\nThe edge extraction module is ready for use in the ControlNet training pipeline.")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)