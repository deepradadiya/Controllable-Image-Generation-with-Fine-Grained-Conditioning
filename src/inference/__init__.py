"""
Inference Module

This module contains inference pipeline components:
- End-to-end inference pipeline combining SD1.5 with ControlNet
- DDIM sampling with ControlNet guidance
- Model loading and compatibility verification
- Batch inference and parameter controls
- Conditioning strength and scheduler management
"""

from .pipeline import (
    ControlNetInferencePipeline,
    InferenceConfig,
    GenerationParams,
    GenerationResult,
    DDIMScheduler,
    ConditionProcessor,
    ConditionType,
)

__all__ = [
    "ControlNetInferencePipeline",
    "InferenceConfig",
    "GenerationParams",
    "GenerationResult",
    "DDIMScheduler",
    "ConditionProcessor",
    "ConditionType",
]

# Optional imports from sub-modules (tasks 9.2 and 9.3)
try:
    from .controls import (
        GenerationParameters,
        ConditioningStrengthSchedule,
        SchedulerManager,
        SchedulerType,
        BatchInferenceManager,
        create_generator,
        prepare_latents,
        apply_conditioning_scale,
        apply_scheduled_conditioning,
    )
    __all__.extend([
        "GenerationParameters",
        "ConditioningStrengthSchedule",
        "SchedulerManager",
        "SchedulerType",
        "BatchInferenceManager",
        "create_generator",
        "prepare_latents",
        "apply_conditioning_scale",
        "apply_scheduled_conditioning",
    ])
except ImportError:
    pass

try:
    from .model_loader import (
        ModelLoader,
        ModelLoadResult,
        CompatibilityReport,
        load_models_for_inference,
    )
    __all__.extend([
        "ModelLoader",
        "ModelLoadResult",
        "CompatibilityReport",
        "load_models_for_inference",
    ])
except ImportError:
    pass
