"""
Deployment Optimization Module for ControlNet Inference Pipeline

This package provides production deployment optimizations for the ControlNet
inference pipeline, designed for HuggingFace Spaces deployment context.

Key Components:
- ModelOptimizer: Model quantization and graph optimization for faster inference
- InferenceCache: LRU caching for text embeddings, condition maps, and latents
- ConcurrencyManager: Request queuing and memory-aware scheduling for multi-user serving
- ModelHealthChecker: Health checking and monitoring for deployed models

Requirements satisfied: 7.1, 8.4
"""

from src.deployment.health_check import ModelHealthChecker, SystemStatus
from src.deployment.optimization import (
    ModelOptimizer,
    InferenceCache,
    ConcurrencyManager,
    OptimizationConfig,
    CacheConfig,
    ConcurrencyConfig,
    QuantizationType,
    RequestPriority,
    RequestStatus,
)

__all__ = [
    "ModelHealthChecker",
    "SystemStatus",
    "ModelOptimizer",
    "InferenceCache",
    "ConcurrencyManager",
    "OptimizationConfig",
    "CacheConfig",
    "ConcurrencyConfig",
    "QuantizationType",
    "RequestPriority",
    "RequestStatus",
]
