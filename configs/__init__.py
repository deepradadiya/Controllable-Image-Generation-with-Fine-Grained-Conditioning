"""
Configuration Module

This module contains configuration files and settings:
- Base configuration with centralized hyperparameters
- Environment-specific configurations (Colab, local, production)
- Model-specific configurations for different conditioning types
"""

# Import main configuration classes with error handling
try:
    from .base_config import (
        BaseConfig,
        ModelConfig,
        TrainingConfig,
        DatasetConfig,
        EvaluationConfig,
        ColabConfig,
        InferenceConfig,
        WebDemoConfig,
        LoggingConfig,
        PathConfig,
        get_config,
        get_depth_config,
        get_pose_config,
        get_edge_config,
        get_colab_config,
        get_local_config,
        default_config,
    )
    
    __all__ = [
        "BaseConfig",
        "ModelConfig", 
        "TrainingConfig",
        "DatasetConfig",
        "EvaluationConfig",
        "ColabConfig",
        "InferenceConfig",
        "WebDemoConfig",
        "LoggingConfig",
        "PathConfig",
        "get_config",
        "get_depth_config",
        "get_pose_config", 
        "get_edge_config",
        "get_colab_config",
        "get_local_config",
        "default_config",
    ]
    
except ImportError as e:
    # Handle missing dependencies gracefully
    print(f"Warning: Could not import all configuration classes: {e}")
    print("Some dependencies may be missing. Please install requirements.txt")
    
    # Provide minimal fallback
    __all__ = []