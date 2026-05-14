"""
ControlNet Training Pipeline

A production-grade training pipeline for ControlNet models that enables spatial
conditioning of Stable Diffusion 1.5 image generation using depth maps, pose
skeletons, and edge maps.

This package implements the architecture from "Adding Conditional Control to
Text-to-Image Diffusion Models" (Zhang et al., 2023) and is optimized for
Google Colab T4 GPU constraints.

Key Features:
- Memory-efficient training for T4 GPU (15GB VRAM)
- Support for depth, pose, and edge conditioning
- Comprehensive evaluation metrics (FID, condition alignment)
- HuggingFace Space compatible web demo
- Robust error handling and recovery
- Extensive documentation and examples

Quick Start:
    from controlnet_pipeline import BaseConfig, get_config
    from controlnet_pipeline.models import ControlNet
    from controlnet_pipeline.training import ControlNetTrainer
    
    # Load configuration
    config = get_config()
    
    # Initialize model and trainer
    model = ControlNet(config.model)
    trainer = ControlNetTrainer(config)
    
    # Start training
    trainer.train()

For detailed usage examples, see the documentation and notebooks.
"""

__version__ = "0.1.0"
__author__ = "ControlNet Pipeline Team"
__email__ = "contact@controlnet-pipeline.com"
__license__ = "Apache License 2.0"

# Core imports for easy access
try:
    from configs.base_config import (
        BaseConfig,
        get_config,
        get_depth_config,
        get_pose_config,
        get_edge_config,
        get_colab_config,
        get_local_config,
        default_config,
    )
except ImportError:
    # Handle case where configs is not in Python path
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from configs.base_config import (
        BaseConfig,
        get_config,
        get_depth_config,
        get_pose_config,
        get_edge_config,
        get_colab_config,
        get_local_config,
        default_config,
    )

# Version and metadata
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "BaseConfig",
    "get_config",
    "get_depth_config", 
    "get_pose_config",
    "get_edge_config",
    "get_colab_config",
    "get_local_config",
    "default_config",
]

# Environment detection
def get_environment_info():
    """Get information about the current execution environment"""
    import sys
    import platform
    
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    
    # Try to import torch
    try:
        import torch
        info.update({
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        })
        
        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            })
    except ImportError:
        info.update({
            "pytorch_version": "Not installed",
            "cuda_available": False,
        })
    
    # Detect execution environment
    try:
        import google.colab
        info["environment"] = "Google Colab"
    except ImportError:
        try:
            import kaggle
            info["environment"] = "Kaggle"
        except ImportError:
            info["environment"] = "Local"
    
    return info


def print_environment_info():
    """Print formatted environment information"""
    info = get_environment_info()
    
    print("🚀 ControlNet Training Pipeline")
    print(f"Version: {__version__}")
    print(f"Environment: {info['environment']}")
    print(f"Python: {info['python_version'].split()[0]}")
    print(f"PyTorch: {info['pytorch_version']}")
    
    if info.get("cuda_available", False):
        print(f"CUDA: {info.get('cuda_version', 'Unknown')}")
        if 'gpu_name' in info:
            print(f"GPU: {info['gpu_name']} ({info.get('gpu_memory_gb', 0):.1f}GB)")
    else:
        print("CUDA: Not available")
    
    print()


# Initialize package
def _initialize_package():
    """Initialize package on import"""
    # Create default configuration to ensure directories exist
    try:
        config = BaseConfig()
        # Directories are created in __post_init__
    except Exception as e:
        # Silently handle initialization errors
        pass


# Run initialization
_initialize_package()

# Optional: Print environment info on import (can be disabled)
import os
if os.getenv("CONTROLNET_SHOW_INFO", "false").lower() == "true":
    print_environment_info()