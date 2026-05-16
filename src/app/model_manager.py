"""
Model Management for HuggingFace Spaces Demo Application

This module provides a ModelManager class that wraps the ModelLoader for the demo
context. It handles pre-trained model loading from HuggingFace Hub, model caching,
lazy loading, error handling with user-friendly messages, model switching between
condition types, and memory management for GPU-constrained environments.

Designed for HuggingFace Spaces deployment with limited resources (T4 GPU, 16GB RAM).

Requirements satisfied: 8.4, 10.5
"""

import gc
import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.inference.model_loader import (
    ModelLoader,
    ModelLoadResult,
    DEFAULT_CONTROLNET_MODELS,
    DEFAULT_SD15_MODEL_ID,
)

logger = logging.getLogger(__name__)


class ModelStatus(str, Enum):
    """Status of a model in the manager."""

    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    UNLOADED = "unloaded"


@dataclass
class ModelInfo:
    """Information about a managed model."""

    condition_type: str
    model_id: str
    status: ModelStatus = ModelStatus.NOT_LOADED
    error_message: str = ""
    load_time_seconds: float = 0.0
    memory_mb: float = 0.0
    last_used: float = 0.0


@dataclass
class ManagerConfig:
    """Configuration for the ModelManager.

    Args:
        cache_dir: Directory for caching downloaded models.
        max_loaded_models: Maximum number of models to keep in memory simultaneously.
        gpu_memory_limit_gb: Maximum GPU memory budget for models.
        torch_dtype: Data type for model loading (float16 for T4 efficiency).
        device: Target device (None for auto-detect).
        lazy_load: If True, models are only loaded on first generation request.
        preload_condition_types: Condition types to preload at startup (empty for lazy).
        base_model_id: HuggingFace model ID for the base SD1.5 model.
        controlnet_model_ids: Mapping of condition type to HuggingFace model ID.
    """

    cache_dir: str = "./cache/models"
    max_loaded_models: int = 1
    gpu_memory_limit_gb: float = 12.0
    torch_dtype: torch.dtype = torch.float16
    device: Optional[str] = None
    lazy_load: bool = True
    preload_condition_types: List[str] = field(default_factory=list)
    base_model_id: str = DEFAULT_SD15_MODEL_ID
    controlnet_model_ids: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_CONTROLNET_MODELS)
    )


class ModelLoadError(Exception):
    """Raised when model loading fails with a user-friendly message."""

    def __init__(self, message: str, condition_type: str = "", technical_detail: str = ""):
        self.condition_type = condition_type
        self.technical_detail = technical_detail
        self.user_message = message
        super().__init__(message)


class ModelManager:
    """
    Manages model lifecycle for the HuggingFace Spaces demo application.

    Wraps the ModelLoader to provide:
    - Lazy loading: models are loaded only when first generation is requested
    - Model caching: avoids reloading on each generation request
    - Model switching: switch between condition types without full reload
    - Memory management: unloads unused models to stay within GPU limits
    - Error handling: user-friendly messages for model loading failures
    - Status reporting: check if models are loaded and ready

    Designed for HuggingFace Spaces deployment with limited resources.

    Usage:
        config = ManagerConfig(lazy_load=True)
        manager = ModelManager(config)

        # Get a pipeline ready for generation (loads on first call)
        pipeline = manager.get_pipeline("depth")

        # Check status
        status = manager.get_status()

        # Switch condition type
        pipeline = manager.get_pipeline("pose")

        # Cleanup
        manager.unload_all()
    """

    def __init__(self, config: Optional[ManagerConfig] = None):
        """
        Initialize the ModelManager.

        Args:
            config: Manager configuration. Uses defaults if None.
        """
        self.config = config or ManagerConfig()
        self._lock = threading.Lock()

        # Initialize the underlying model loader
        self._loader = ModelLoader(
            cache_dir=self.config.cache_dir,
            torch_dtype=self.config.torch_dtype,
            device=self.config.device,
        )

        # Track model states
        self._models: Dict[str, ModelInfo] = {}
        self._pipelines: Dict[str, Any] = {}
        self._controlnets: Dict[str, Any] = {}
        self._active_condition_type: Optional[str] = None

        # Initialize model info for all supported condition types
        for ctype, model_id in self.config.controlnet_model_ids.items():
            self._models[ctype] = ModelInfo(
                condition_type=ctype,
                model_id=model_id,
                status=ModelStatus.NOT_LOADED,
            )

        logger.info(
            f"ModelManager initialized: device={self._loader.device}, "
            f"lazy_load={self.config.lazy_load}, "
            f"max_loaded={self.config.max_loaded_models}, "
            f"supported_types={list(self._models.keys())}"
        )

        # Preload models if configured (non-lazy mode)
        if not self.config.lazy_load and self.config.preload_condition_types:
            for ctype in self.config.preload_condition_types:
                try:
                    self._load_controlnet(ctype)
                except ModelLoadError as e:
                    logger.warning(f"Preload failed for {ctype}: {e.user_message}")

    def get_pipeline(self, condition_type: str) -> Any:
        """
        Get a ready-to-use inference pipeline for the given condition type.

        This is the main entry point for the demo. It handles lazy loading,
        model switching, and memory management transparently.

        Args:
            condition_type: Type of conditioning (depth, pose, edge).

        Returns:
            A loaded StableDiffusionControlNetPipeline ready for inference.

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        condition_type = condition_type.lower().strip()
        self._validate_condition_type(condition_type)

        with self._lock:
            # Check if pipeline is already loaded for this condition type
            if condition_type in self._pipelines and self._pipelines[condition_type] is not None:
                self._models[condition_type].last_used = time.time()
                self._active_condition_type = condition_type
                logger.info(f"Returning cached pipeline for '{condition_type}'")
                return self._pipelines[condition_type]

            # Ensure we have room in memory
            self._enforce_memory_limits(exclude=condition_type)

            # Load the ControlNet model
            controlnet = self._load_controlnet(condition_type)

            # Load the base pipeline with this ControlNet
            pipeline = self._load_pipeline(condition_type, controlnet)

            # Cache the pipeline
            self._pipelines[condition_type] = pipeline
            self._models[condition_type].status = ModelStatus.READY
            self._models[condition_type].last_used = time.time()
            self._active_condition_type = condition_type

            return pipeline

    def get_controlnet(self, condition_type: str) -> Any:
        """
        Get a loaded ControlNet model for the given condition type.

        Args:
            condition_type: Type of conditioning (depth, pose, edge).

        Returns:
            A loaded ControlNet model.

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        condition_type = condition_type.lower().strip()
        self._validate_condition_type(condition_type)

        with self._lock:
            if condition_type in self._controlnets and self._controlnets[condition_type] is not None:
                self._models[condition_type].last_used = time.time()
                return self._controlnets[condition_type]

            return self._load_controlnet(condition_type)

    def is_ready(self, condition_type: Optional[str] = None) -> bool:
        """
        Check if models are loaded and ready for generation.

        Args:
            condition_type: Specific condition type to check.
                           If None, checks if any model is ready.

        Returns:
            True if the specified (or any) model is ready.
        """
        if condition_type:
            condition_type = condition_type.lower().strip()
            info = self._models.get(condition_type)
            return info is not None and info.status == ModelStatus.READY
        return any(info.status == ModelStatus.READY for info in self._models.values())

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the status of all managed models.

        Returns:
            Dictionary mapping condition type to status information.
        """
        status = {}
        for ctype, info in self._models.items():
            status[ctype] = {
                "status": info.status.value,
                "model_id": info.model_id,
                "error_message": info.error_message,
                "load_time_seconds": info.load_time_seconds,
                "memory_mb": info.memory_mb,
                "last_used": info.last_used,
            }
        return status

    def get_active_condition_type(self) -> Optional[str]:
        """Get the currently active condition type."""
        return self._active_condition_type

    def get_supported_condition_types(self) -> List[str]:
        """Get list of supported condition types."""
        return list(self._models.keys())

    def switch_condition_type(self, condition_type: str) -> Any:
        """
        Switch to a different condition type.

        If the target model is already cached, this is instant.
        Otherwise, it loads the new model (potentially unloading others
        to stay within memory limits).

        Args:
            condition_type: Target condition type (depth, pose, edge).

        Returns:
            A loaded pipeline for the new condition type.

        Raises:
            ModelLoadError: If the model cannot be loaded.
        """
        return self.get_pipeline(condition_type)

    def unload(self, condition_type: str) -> None:
        """
        Unload a specific model to free memory.

        Args:
            condition_type: Condition type to unload.
        """
        condition_type = condition_type.lower().strip()

        with self._lock:
            self._unload_model(condition_type)

    def unload_all(self) -> None:
        """Unload all models and free GPU memory."""
        with self._lock:
            for ctype in list(self._models.keys()):
                self._unload_model(ctype)

            # Clear the underlying loader cache
            self._loader.clear_cache()

            # Force garbage collection and GPU cache clearing
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._active_condition_type = None
            logger.info("All models unloaded and memory freed")

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage information.

        Returns:
            Dictionary with memory usage details in MB.
        """
        usage = {
            "total_model_memory_mb": sum(
                info.memory_mb for info in self._models.values()
                if info.status == ModelStatus.READY
            ),
            "loaded_models": sum(
                1 for info in self._models.values()
                if info.status == ModelStatus.READY
            ),
        }

        if torch.cuda.is_available():
            usage["gpu_allocated_mb"] = torch.cuda.memory_allocated() / (1024 ** 2)
            usage["gpu_reserved_mb"] = torch.cuda.memory_reserved() / (1024 ** 2)
            usage["gpu_total_mb"] = torch.cuda.get_device_properties(0).total_mem / (1024 ** 2)
        else:
            usage["gpu_allocated_mb"] = 0.0
            usage["gpu_reserved_mb"] = 0.0
            usage["gpu_total_mb"] = 0.0

        return usage

    # -------------------------------------------------------------------------
    # Private methods
    # -------------------------------------------------------------------------

    def _validate_condition_type(self, condition_type: str) -> None:
        """Validate that the condition type is supported."""
        if condition_type not in self._models:
            supported = ", ".join(self._models.keys())
            raise ModelLoadError(
                message=(
                    f"Unsupported condition type: '{condition_type}'. "
                    f"Supported types are: {supported}."
                ),
                condition_type=condition_type,
            )

    def _load_controlnet(self, condition_type: str) -> Any:
        """
        Load a ControlNet model for the given condition type.

        Args:
            condition_type: Type of conditioning.

        Returns:
            Loaded ControlNet model.

        Raises:
            ModelLoadError: If loading fails.
        """
        # Return cached ControlNet if available
        if condition_type in self._controlnets and self._controlnets[condition_type] is not None:
            return self._controlnets[condition_type]

        info = self._models[condition_type]
        info.status = ModelStatus.LOADING
        start_time = time.time()

        logger.info(f"Loading ControlNet for '{condition_type}' from '{info.model_id}'")

        try:
            result = self._loader.load_and_verify_controlnet(
                model_id_or_path=info.model_id,
                condition_type=condition_type,
            )

            if not result.success:
                raise ModelLoadError(
                    message=self._format_user_error(condition_type, result.error_message),
                    condition_type=condition_type,
                    technical_detail=result.error_message,
                )

            # Track load time and memory
            load_time = time.time() - start_time
            info.load_time_seconds = load_time
            info.memory_mb = self._estimate_model_memory(result.model)

            # Cache the ControlNet
            self._controlnets[condition_type] = result.model

            if result.warnings:
                for warning in result.warnings:
                    logger.warning(f"ControlNet '{condition_type}': {warning}")

            logger.info(
                f"ControlNet '{condition_type}' loaded in {load_time:.1f}s "
                f"(~{info.memory_mb:.0f} MB)"
            )

            return result.model

        except ModelLoadError:
            info.status = ModelStatus.ERROR
            raise
        except Exception as e:
            info.status = ModelStatus.ERROR
            info.error_message = str(e)
            raise ModelLoadError(
                message=self._format_user_error(condition_type, str(e)),
                condition_type=condition_type,
                technical_detail=str(e),
            )

    def _load_pipeline(self, condition_type: str, controlnet: Any) -> Any:
        """
        Load the base SD1.5 pipeline with the given ControlNet.

        Args:
            condition_type: Type of conditioning.
            controlnet: Loaded ControlNet model.

        Returns:
            Loaded pipeline.

        Raises:
            ModelLoadError: If loading fails.
        """
        logger.info(f"Loading base pipeline for '{condition_type}'")

        try:
            result = self._loader.load_base_pipeline(
                model_id=self.config.base_model_id,
                controlnet=controlnet,
            )

            if not result.success:
                raise ModelLoadError(
                    message=self._format_user_error(condition_type, result.error_message),
                    condition_type=condition_type,
                    technical_detail=result.error_message,
                )

            logger.info(f"Base pipeline loaded for '{condition_type}'")
            return result.model

        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(
                message=self._format_user_error(condition_type, str(e)),
                condition_type=condition_type,
                technical_detail=str(e),
            )

    def _unload_model(self, condition_type: str) -> None:
        """Unload a model and free its memory."""
        if condition_type not in self._models:
            return

        info = self._models[condition_type]

        # Remove pipeline
        if condition_type in self._pipelines:
            del self._pipelines[condition_type]

        # Remove ControlNet
        if condition_type in self._controlnets:
            del self._controlnets[condition_type]

        # Update status
        info.status = ModelStatus.UNLOADED
        info.memory_mb = 0.0
        info.error_message = ""

        # Clear from underlying loader cache
        cache_key = f"controlnet_{info.model_id}"
        self._loader.clear_cache(cache_key)

        logger.info(f"Unloaded model for '{condition_type}'")

    def _enforce_memory_limits(self, exclude: str = "") -> None:
        """
        Ensure we stay within memory limits by unloading least-recently-used models.

        Args:
            exclude: Condition type to exclude from unloading.
        """
        loaded_count = sum(
            1 for ctype, info in self._models.items()
            if info.status == ModelStatus.READY and ctype != exclude
        )

        # If we're at the limit, unload the least recently used model
        while loaded_count >= self.config.max_loaded_models:
            lru_type = self._find_lru_model(exclude)
            if lru_type is None:
                break

            logger.info(
                f"Memory limit reached ({loaded_count}/{self.config.max_loaded_models}). "
                f"Unloading least recently used: '{lru_type}'"
            )
            self._unload_model(lru_type)
            loaded_count -= 1

        # Also check GPU memory if available
        if torch.cuda.is_available():
            gpu_used_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            if gpu_used_gb > self.config.gpu_memory_limit_gb * 0.9:
                # We're close to the limit, try to free some memory
                gc.collect()
                torch.cuda.empty_cache()

    def _find_lru_model(self, exclude: str = "") -> Optional[str]:
        """Find the least recently used loaded model."""
        lru_type = None
        lru_time = float("inf")

        for ctype, info in self._models.items():
            if ctype == exclude:
                continue
            if info.status == ModelStatus.READY and info.last_used < lru_time:
                lru_time = info.last_used
                lru_type = ctype

        return lru_type

    def _estimate_model_memory(self, model: Any) -> float:
        """Estimate memory usage of a model in MB."""
        if model is None:
            return 0.0

        try:
            # Try to count parameters
            if hasattr(model, "parameters"):
                total_params = sum(p.numel() for p in model.parameters())
                # FP16 = 2 bytes per param, FP32 = 4 bytes per param
                bytes_per_param = 2 if self.config.torch_dtype == torch.float16 else 4
                return (total_params * bytes_per_param) / (1024 ** 2)
        except Exception:
            pass

        # Fallback: estimate based on typical ControlNet size
        return 361.0  # Approximate ControlNet size in MB

    def _format_user_error(self, condition_type: str, technical_error: str) -> str:
        """
        Format a technical error into a user-friendly message.

        Args:
            condition_type: The condition type that failed.
            technical_error: The technical error message.

        Returns:
            A user-friendly error message.
        """
        error_lower = technical_error.lower()

        if "not found" in error_lower or "404" in error_lower:
            return (
                f"The {condition_type} model could not be found. "
                f"Please check your internet connection and try again. "
                f"If the problem persists, the model may have been removed from HuggingFace Hub."
            )
        elif "out of memory" in error_lower or "oom" in error_lower or "cuda" in error_lower:
            return (
                f"Not enough GPU memory to load the {condition_type} model. "
                f"Try closing other applications or restarting the runtime. "
                f"The demo requires at least 6GB of free GPU memory."
            )
        elif "connection" in error_lower or "timeout" in error_lower or "network" in error_lower:
            return (
                f"Network error while downloading the {condition_type} model. "
                f"Please check your internet connection and try again."
            )
        elif "permission" in error_lower or "auth" in error_lower or "token" in error_lower:
            return (
                f"Permission denied when accessing the {condition_type} model. "
                f"This model may require authentication. "
                f"Please provide a valid HuggingFace token."
            )
        elif "incompatible" in error_lower or "mismatch" in error_lower:
            return (
                f"The {condition_type} model is not compatible with the base model. "
                f"Please ensure you are using a ControlNet model designed for Stable Diffusion 1.5."
            )
        else:
            return (
                f"Failed to load the {condition_type} model. "
                f"Error: {technical_error[:200]}. "
                f"Please try again or select a different condition type."
            )
