"""
COCO Dataset Processing Example

This example demonstrates how to use the DatasetProcessor to download and process
COCO dataset samples for ControlNet training.

Usage:
    python examples/dataset_processing_example.py --subset-size 100 --val-ratio 0.1
"""

import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.dataset_processor import create_coco_dataset_for_training, DatasetProcessor


def main():
    parser = argparse.ArgumentParser(description="COCO Dataset Processing Example")
    parser.add_argument("--subset-size", type=int, default=100, 
                       help="Number of samples to download (default: 100)")
    parser.add_argument("--val-ratio", type=float, default=0.1, 
                       help="Validation split ratio (default: 0.1)")
    parser.add_argument("--cache-dir", type=str, default="./data/cache", 
                       help="Cache directory for downloads")
    parser.add_argument("--output-dir", type=str, default="./data/processed", 
                       help="Output directory for processed dataset")
    
    args = parser.parse_args()
    
    print(f"COCO Dataset Processing Example")
    print(f"Subset size: {args.subset_size}")
    print(f"Validation ratio: {args.val_ratio}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Output directory: {args.output_dir}")
    print("-" * 50)
    
    try:
        # Create dataset using the convenience function
        print("Starting dataset creation...")
        train_samples, val_samples, report = create_coco_dataset_for_training(
            subset_size=args.subset_size,
            val_ratio=args.val_ratio,
            cache_dir=args.cache_dir
        )
        
        print(f"\nDataset Processing Results:")
        print(f"Total samples processed: {report.total_samples}")
        print(f"Valid samples: {report.valid_samples}")
        print(f"Invalid samples: {report.invalid_samples}")
        print(f"Success rate: {report.success_rate:.1%}")
        print(f"Processing time: {report.processing_time_seconds:.2f} seconds")
        
        print(f"\nDataset Split:")
        print(f"Training samples: {len(train_samples)}")
        print(f"Validation samples: {len(val_samples)}")
        
        if report.errors:
            print(f"\nErrors encountered ({len(report.errors)}):")
            for i, error in enumerate(report.errors[:5]):  # Show first 5 errors
                print(f"  {i+1}. {error}")
            if len(report.errors) > 5:
                print(f"  ... and {len(report.errors) - 5} more errors")
        
        # Save processed dataset
        if train_samples:
            print(f"\nSaving processed dataset to {args.output_dir}...")
            processor = DatasetProcessor()
            
            output_dir = Path(args.output_dir)
            processor.save_processed_dataset(train_samples, output_dir / "train.json")
            processor.save_processed_dataset(val_samples, output_dir / "val.json")
            
            # Generate and save statistics
            stats = processor.get_dataset_statistics(train_samples + val_samples)
            
            import json
            with open(output_dir / "statistics.json", 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            print(f"Dataset saved successfully!")
            print(f"Files created:")
            print(f"  - {output_dir / 'train.json'} ({len(train_samples)} samples)")
            print(f"  - {output_dir / 'val.json'} ({len(val_samples)} samples)")
            print(f"  - {output_dir / 'statistics.json'}")
            print(f"  - {output_dir / 'images/'} (image files)")
            
            # Display sample statistics
            print(f"\nDataset Statistics:")
            print(f"Image dimensions:")
            img_stats = stats['image_statistics']
            print(f"  Width: {img_stats['width']['min']}-{img_stats['width']['max']} "
                  f"(avg: {img_stats['width']['mean']:.0f})")
            print(f"  Height: {img_stats['height']['min']}-{img_stats['height']['max']} "
                  f"(avg: {img_stats['height']['mean']:.0f})")
            
            cap_stats = stats['caption_statistics']
            print(f"Caption lengths:")
            print(f"  {cap_stats['length']['min']}-{cap_stats['length']['max']} characters "
                  f"(avg: {cap_stats['length']['mean']:.0f})")
            
            # Show a sample
            if train_samples:
                sample = train_samples[0]
                print(f"\nSample from dataset:")
                print(f"  ID: {sample.image_id}")
                print(f"  Image size: {sample.image.size}")
                print(f"  Caption: {sample.caption[:100]}{'...' if len(sample.caption) > 100 else ''}")
        
        print(f"\nDataset processing completed successfully!")
        
    except Exception as e:
        print(f"Error during dataset processing: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()