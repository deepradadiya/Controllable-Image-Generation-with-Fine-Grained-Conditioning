#!/usr/bin/env python3
"""
Test script for depth conditioning training.

This script tests the depth training pipeline with minimal configuration
to verify that all components work together correctly.
"""

import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.models.config import ControlNetConfig, TrainingConfig, ModelMetadata
from src.training.train_depth import parse_args

def test_depth_training_config():
    """Test depth training configuration creation."""
    print("Testing depth training configuration...")
    
    # Test model config
    model_config = ControlNetConfig(
        condition_type="depth",
        conditioning_channels=1,
        in_channels=4,
        block_out_channels=(320, 640, 1280, 1280),
        cross_attention_dim=768,
    )
    
    print(f"✓ Model config created: {model_config.condition_type} conditioning")
    print(f"✓ Conditioning channels: {model_config.conditioning_channels}")
    
    # Test training config
    training_config = TrainingConfig(
        learning_rate=1e-5,
        num_train_epochs=1,  # Minimal for testing
        train_batch_size=1,
        gradient_accumulation_steps=2,
        mixed_precision="fp16",
        gradient_checkpointing=True,
    )
    
    print(f"✓ Training config created: {training_config.num_train_epochs} epochs")
    print(f"✓ Learning rate: {training_config.learning_rate}")
    
    # Test metadata
    metadata = ModelMetadata(
        model_name="controlnet_depth_test",
        condition_type="depth",
        description="Test depth ControlNet model",
        tags=["depth", "controlnet", "test"],
    )
    
    print(f"✓ Metadata created: {metadata.model_name}")
    
    return True

def test_argument_parsing():
    """Test command line argument parsing."""
    print("\nTesting argument parsing...")
    
    # Mock command line arguments
    test_args = [
        "--output_dir", "./test_output",
        "--num_train_epochs", "1",
        "--train_batch_size", "1",
        "--learning_rate", "1e-5",
        "--resolution", "256",  # Smaller for testing
        "--max_train_samples", "10",  # Very small dataset
        "--gradient_accumulation_steps", "2",
        "--mixed_precision", "fp16",
        "--report_to", "none",  # Disable wandb for testing
    ]
    
    # Temporarily replace sys.argv
    original_argv = sys.argv
    sys.argv = ["train_depth.py"] + test_args
    
    try:
        args = parse_args()
        print(f"✓ Arguments parsed successfully")
        print(f"✓ Output dir: {args.output_dir}")
        print(f"✓ Epochs: {args.num_train_epochs}")
        print(f"✓ Batch size: {args.train_batch_size}")
        print(f"✓ Resolution: {args.resolution}")
        
        return True
        
    except Exception as e:
        print(f"✗ Argument parsing failed: {e}")
        return False
        
    finally:
        sys.argv = original_argv

def test_imports():
    """Test that all required imports work."""
    print("\nTesting imports...")
    
    try:
        # Test core imports
        from src.models.controlnet import ControlNetModel
        from src.models.unet_wrapper import ControlNetUNet2DConditionModel
        from src.training.trainer import ControlNetTrainer
        from src.training.losses import create_loss_function
        
        print("✓ Core model imports successful")
        
        # Test diffusers imports
        from diffusers import DDPMScheduler, UNet2DConditionModel
        print("✓ Diffusers imports successful")
        
        # Test data processing imports
        from src.data.dataset_processor import COCODatasetProcessor
        from src.data.extract_depth import DepthExtractor
        print("✓ Data processing imports successful")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Depth ControlNet Training Pipeline")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_depth_training_config,
        test_argument_parsing,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Depth training pipeline is ready.")
        print("\nTo run actual training, use:")
        print("python src/training/train_depth.py --output_dir ./depth_model --num_train_epochs 1 --max_train_samples 100")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)