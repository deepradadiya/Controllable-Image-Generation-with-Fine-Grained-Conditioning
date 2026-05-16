"""
Production Deployment Optimization for ControlNet Inference

This module implements three core optimization classes for deploying ControlNet
models in a HuggingFace Spaces production environment:

1. ModelOptimizer: Handles model quantization (FP16, INT8), torch.compile()
   graph optimization, and ONNX export for cross-platform deployment.

2. InferenceCache: Implements LRU caching strategies for text encoder embeddings,
   condition map processing results, and latent outputs for deterministic generation.

3. ConcurrencyManager: Manages concurrent user requests with memory-aware scheduling,
   request queuing, and graceful degradation under load.

Design Decisions:
- FP16 is the default precision for T4 GPU deployment (already supported in pipeline)
- INT8 quantization uses PyTorch's dynamic quantization for CPU fallback scenarios
- torch.compile() is used when available (PyTorch 2.0+) for graph-level optimization
- LRU caches use configurable max sizes to bound memory usage
- Concurrency management uses asyncio-compatible queuing for Gradio integration
- Graceful degradation reduces inference steps and resolution under heavy load

Requirements satisfied: 7.1, 8.4
"""

import gc
import hashlib
import logging
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Data Classes
# =============================================================================


class QuantizationType(str, Enum):
    """Supported quantization types for model optimization."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"


class RequestPriority(str, Enum):
    """Priority levels for inference requests."""
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class RequestStatus(str, Enum):
    """Status of an inference request in the queue."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEGRADED = "degraded"


@dataclass
class OptimizationConfig:
    """Configuration for model optimization.

    Args:
        quantization: Target quantization type for the model
        enable_torch_compile: Whether to use torch.compile() for graph optimization
        compile_mode: torch.compile mode ('default', 'reduce-overhead', 'max-autotune')
        enable_onnx_export: Whether to export model to ONNX format
        onnx_opset_version: ONNX opset version for export
        onnx_output_path: Path to save exported ONNX model
        enable_attention_slicing: Enable attention slicing for memory reduction
        attention_slice_size: Size of attention slices ('auto' or integer)
        enable_vae_slicing: Enable VAE slicing for large images
        enable_cpu_offload: Enable sequential CPU offloading for low-memory scenarios
    """
    quantization: QuantizationType = QuantizationType.FP16
    enable_torch_compile: bool = True
    compile_mode: str = "reduce-overhead"
    enable_onnx_export: bool = False
    onnx_opset_version: int = 17
    onnx_output_path: Optional[str] = None
    enable_attention_slicing: bool = True
    attention_slice_size: Union[str, int] = "auto"
    enable_vae_slicing: bool = True
    enable_cpu_offload: bool = False


@dataclass
class CacheConfig:
    """Configuration for inference caching.

    Args:
        max_text_embeddings: Maximum number of text embeddings to cache
        max_condition_maps: Maximum number of processed condition maps to cache
        max_latent_cache: Maximum number of latent results to cache
        text_embedding_ttl_seconds: Time-to-live for text embedding cache entries
        condition_map_ttl_seconds: Time-to-live for condition map cache entries
        latent_ttl_seconds: Time-to-live for latent cache entries
        enable_disk_cache: Whether to persist cache to disk
        disk_cache_path: Path for disk-based cache storage
    """
    max_text_embeddings: int = 128
    max_condition_maps: int = 64
    max_latent_cache: int = 32
    text_embedding_ttl_seconds: float = 3600.0  # 1 hour
    condition_map_ttl_seconds: float = 1800.0  # 30 minutes
    latent_ttl_seconds: float = 900.0  # 15 minutes
    enable_disk_cache: bool = False
    disk_cache_path: Optional[str] = None


@dataclass
class ConcurrencyConfig:
    """Configuration for concurrency management.

    Args:
        max_concurrent_requests: Maximum number of simultaneous inference requests
        max_queue_size: Maximum number of requests waiting in queue
        request_timeout_seconds: Timeout for individual requests
        memory_threshold_gb: GPU memory threshold for degradation trigger
        degradation_steps_reduction: Reduce inference steps by this factor under load
        degradation_resolution_factor: Scale down resolution by this factor under load
        enable_priority_queue: Whether to use priority-based scheduling
        health_check_interval_seconds: Interval for memory health checks
    """
    max_concurrent_requests: int = 2
    max_queue_size: int = 10
    request_timeout_seconds: float = 120.0
    memory_threshold_gb: float = 12.0  # T4 has ~15GB, trigger at 12GB
    degradation_steps_reduction: float = 0.5
    degradation_resolution_factor: float = 0.75
    enable_priority_queue: bool = True
    health_check_interval_seconds: float = 5.0


# =============================================================================
# ModelOptimizer Class
# =============================================================================


class ModelOptimizer:
    """
    Optimizes ControlNet and SD1.5 models for production inference.

    Provides model quantization (FP16, INT8), torch.compile() graph optimization,
    and ONNX export capabilities. Designed for HuggingFace Spaces deployment
    where inference speed and memory efficiency are critical.

    Optimization Strategies:
    - FP16: Default for CUDA devices, ~2x memory reduction with minimal quality loss
    - INT8: Dynamic quantization for CPU deployment, ~4x memory reduction
    - torch.compile(): PyTorch 2.0+ graph optimization for ~20-30% speedup
    - ONNX: Cross-platform deployment with runtime optimization

    Example:
        >>> optimizer = ModelOptimizer(OptimizationConfig(quantization=QuantizationType.FP16))
        >>> optimized_model = optimizer.optimize_model(controlnet_model)
        >>> compiled_unet = optimizer.apply_torch_compile(unet_model)
    """

    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize the model optimizer.

        Args:
            config: Optimization configuration. Uses defaults if None.
        """
        self.config = config or OptimizationConfig()
        self._device = self._detect_device()
        self._torch_compile_available = self._check_torch_compile()
        logger.info(
            f"ModelOptimizer initialized (quantization={self.config.quantization.value}, "
            f"torch_compile={'available' if self._torch_compile_available else 'unavailable'}, "
            f"device={self._device})"
        )

    def _detect_device(self) -> torch.device:
        """Detect the best available device for inference."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _check_torch_compile(self) -> bool:
        """Check if torch.compile() is available (PyTorch 2.0+)."""
        try:
            major_version = int(torch.__version__.split(".")[0])
            return major_version >= 2
        except (ValueError, IndexError):
            return False

    def optimize_model(
        self,
        model: nn.Module,
        quantization: Optional[QuantizationType] = None,
    ) -> nn.Module:
        """
        Apply quantization optimization to a model.

        Applies the specified quantization type to reduce model memory footprint
        and improve inference speed. FP16 is recommended for CUDA devices,
        INT8 for CPU-only deployment.

        Args:
            model: PyTorch model to optimize
            quantization: Override quantization type (uses config default if None)

        Returns:
            Optimized model with applied quantization

        Raises:
            ValueError: If quantization type is not supported on the current device
        """
        target_quant = quantization or self.config.quantization

        if target_quant == QuantizationType.FP16:
            return self._apply_fp16(model)
        elif target_quant == QuantizationType.BF16:
            return self._apply_bf16(model)
        elif target_quant == QuantizationType.INT8:
            return self._apply_int8(model)
        elif target_quant == QuantizationType.FP32:
            logger.info("FP32 requested, no quantization applied")
            return model
        else:
            raise ValueError(f"Unsupported quantization type: {target_quant}")

    def _apply_fp16(self, model: nn.Module) -> nn.Module:
        """
        Apply FP16 (half precision) quantization.

        Converts model parameters to float16 for ~2x memory reduction.
        Best suited for CUDA devices where FP16 operations are hardware-accelerated.

        Args:
            model: Model to convert to FP16

        Returns:
            Model with FP16 parameters
        """
        if self._device.type != "cuda":
            logger.warning(
                "FP16 on CPU may be slow. Consider INT8 for CPU deployment."
            )

        model = model.half()
        model = model.to(self._device)
        logger.info(
            f"Applied FP16 quantization. "
            f"Model size: {self._get_model_size_mb(model):.1f} MB"
        )
        return model

    def _apply_bf16(self, model: nn.Module) -> nn.Module:
        """
        Apply BF16 (bfloat16) quantization.

        BFloat16 provides better numerical stability than FP16 for training
        and inference, with the same memory footprint. Requires Ampere+ GPU.

        Args:
            model: Model to convert to BF16

        Returns:
            Model with BF16 parameters
        """
        if not torch.cuda.is_bf16_supported() and self._device.type == "cuda":
            logger.warning("BF16 not supported on this GPU, falling back to FP16")
            return self._apply_fp16(model)

        model = model.to(dtype=torch.bfloat16)
        model = model.to(self._device)
        logger.info(
            f"Applied BF16 quantization. "
            f"Model size: {self._get_model_size_mb(model):.1f} MB"
        )
        return model

    def _apply_int8(self, model: nn.Module) -> nn.Module:
        """
        Apply INT8 dynamic quantization for CPU inference.

        Uses PyTorch's dynamic quantization to convert Linear and Conv2d layers
        to INT8 precision. This provides ~4x memory reduction and faster inference
        on CPU, but is not suitable for CUDA devices.

        For GPU INT8, bitsandbytes integration is used when available.

        Args:
            model: Model to quantize to INT8

        Returns:
            INT8 quantized model
        """
        if self._device.type == "cuda":
            return self._apply_int8_gpu(model)
        else:
            return self._apply_int8_cpu(model)

    def _apply_int8_cpu(self, model: nn.Module) -> nn.Module:
        """
        Apply INT8 dynamic quantization for CPU using PyTorch native quantization.

        Falls back to FP32 if the quantization engine is not available
        (e.g., on ARM/Apple Silicon where qint8 is not supported).
        """
        model = model.to("cpu").float()

        try:
            # Dynamic quantization targets Linear layers (most impactful for transformers)
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear},
                dtype=torch.qint8,
            )

            logger.info(
                f"Applied INT8 CPU quantization. "
                f"Model size: {self._get_model_size_mb(quantized_model):.1f} MB"
            )
            return quantized_model
        except RuntimeError as e:
            if "NoQEngine" in str(e) or "Didn't find engine" in str(e):
                logger.warning(
                    "INT8 quantization engine not available on this platform. "
                    "Returning FP32 model. INT8 is supported on x86 Linux/Windows."
                )
                return model
            raise

    def _apply_int8_gpu(self, model: nn.Module) -> nn.Module:
        """
        Apply INT8 quantization for GPU using bitsandbytes.

        Falls back to FP16 if bitsandbytes is not available.

        Args:
            model: Model to quantize

        Returns:
            INT8 quantized model for GPU
        """
        try:
            import bitsandbytes as bnb

            # Replace Linear layers with 8-bit equivalents
            model = model.to(self._device)
            self._replace_linear_with_int8(model, bnb)
            logger.info(
                f"Applied INT8 GPU quantization via bitsandbytes. "
                f"Model size: {self._get_model_size_mb(model):.1f} MB"
            )
            return model
        except ImportError:
            logger.warning(
                "bitsandbytes not available for GPU INT8. Falling back to FP16."
            )
            return self._apply_fp16(model)

    def _replace_linear_with_int8(self, model: nn.Module, bnb) -> None:
        """
        Recursively replace Linear layers with bitsandbytes Int8 layers.

        Args:
            model: Model to modify in-place
            bnb: bitsandbytes module
        """
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                # Create Int8 linear replacement
                int8_layer = bnb.nn.Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    has_fp16_weights=False,
                    threshold=6.0,
                )
                int8_layer.weight = bnb.nn.Int8Params(
                    module.weight.data,
                    requires_grad=False,
                    has_fp16_weights=False,
                )
                if module.bias is not None:
                    int8_layer.bias = nn.Parameter(module.bias.data)
                setattr(model, name, int8_layer)
            else:
                # Recurse into child modules
                self._replace_linear_with_int8(module, bnb)

    def apply_torch_compile(
        self,
        model: nn.Module,
        mode: Optional[str] = None,
    ) -> nn.Module:
        """
        Apply torch.compile() graph optimization for PyTorch 2.0+.

        torch.compile() traces the model's forward pass and generates optimized
        kernels, providing ~20-30% inference speedup through operator fusion,
        memory planning, and kernel selection.

        Available modes:
        - 'default': Good balance of compile time and runtime performance
        - 'reduce-overhead': Minimizes framework overhead (best for small models)
        - 'max-autotune': Maximum optimization (longer compile, fastest runtime)

        Args:
            model: Model to compile
            mode: Compilation mode (uses config default if None)

        Returns:
            Compiled model (or original if torch.compile unavailable)
        """
        if not self._torch_compile_available:
            logger.warning(
                "torch.compile() not available (requires PyTorch 2.0+). "
                "Returning uncompiled model."
            )
            return model

        if not self.config.enable_torch_compile:
            logger.info("torch.compile() disabled in config")
            return model

        compile_mode = mode or self.config.compile_mode

        try:
            compiled_model = torch.compile(model, mode=compile_mode)
            logger.info(f"Applied torch.compile() with mode='{compile_mode}'")
            return compiled_model
        except Exception as e:
            logger.warning(
                f"torch.compile() failed: {e}. Returning uncompiled model."
            )
            return model

    def export_onnx(
        self,
        model: nn.Module,
        sample_input: Dict[str, torch.Tensor],
        output_path: Optional[str] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    ) -> Optional[str]:
        """
        Export model to ONNX format for cross-platform deployment.

        ONNX export enables deployment on various runtimes (ONNX Runtime,
        TensorRT, OpenVINO) for optimized inference on different hardware.

        Args:
            model: Model to export
            sample_input: Dictionary of sample input tensors for tracing
            output_path: Path to save ONNX model (uses config default if None)
            dynamic_axes: Dynamic axes specification for variable-size inputs

        Returns:
            Path to exported ONNX model, or None if export failed
        """
        if not self.config.enable_onnx_export and output_path is None:
            logger.info("ONNX export disabled in config")
            return None

        export_path = output_path or self.config.onnx_output_path
        if export_path is None:
            export_path = "controlnet_optimized.onnx"

        # Ensure output directory exists
        Path(export_path).parent.mkdir(parents=True, exist_ok=True)

        # Default dynamic axes for batch size flexibility
        if dynamic_axes is None:
            dynamic_axes = {
                name: {0: "batch_size"} for name in sample_input.keys()
            }

        try:
            model.eval()
            input_names = list(sample_input.keys())
            input_tensors = tuple(sample_input.values())

            torch.onnx.export(
                model,
                input_tensors,
                export_path,
                input_names=input_names,
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                opset_version=self.config.onnx_opset_version,
                do_constant_folding=True,
            )

            logger.info(f"ONNX model exported to: {export_path}")
            return export_path

        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return None

    def apply_attention_optimization(self, model: nn.Module) -> nn.Module:
        """
        Apply attention-specific optimizations for memory efficiency.

        Enables attention slicing and/or VAE slicing to reduce peak memory
        usage during inference, critical for T4 GPU deployment.

        Args:
            model: Model with attention layers to optimize

        Returns:
            Model with attention optimizations applied
        """
        if self.config.enable_attention_slicing:
            if hasattr(model, "set_attention_slice"):
                model.set_attention_slice(self.config.attention_slice_size)
                logger.info(
                    f"Enabled attention slicing "
                    f"(slice_size={self.config.attention_slice_size})"
                )

        if self.config.enable_vae_slicing:
            if hasattr(model, "enable_vae_slicing"):
                model.enable_vae_slicing()
                logger.info("Enabled VAE slicing")

        return model

    def get_optimization_report(self, model: nn.Module) -> Dict[str, Any]:
        """
        Generate a report of applied optimizations and model statistics.

        Args:
            model: Optimized model to report on

        Returns:
            Dictionary containing optimization metrics and model info
        """
        report = {
            "quantization": self.config.quantization.value,
            "torch_compile_enabled": self.config.enable_torch_compile,
            "torch_compile_available": self._torch_compile_available,
            "model_size_mb": self._get_model_size_mb(model),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "device": str(self._device),
            "dtype": str(next(model.parameters()).dtype),
        }

        if self._device.type == "cuda":
            report["gpu_memory_allocated_mb"] = (
                torch.cuda.memory_allocated() / (1024**2)
            )
            report["gpu_memory_reserved_mb"] = (
                torch.cuda.memory_reserved() / (1024**2)
            )

        return report

    @staticmethod
    def _get_model_size_mb(model: nn.Module) -> float:
        """Calculate model size in megabytes."""
        param_size = sum(
            p.nelement() * p.element_size() for p in model.parameters()
        )
        buffer_size = sum(
            b.nelement() * b.element_size() for b in model.buffers()
        )
        return (param_size + buffer_size) / (1024**2)



# =============================================================================
# InferenceCache Class
# =============================================================================


class _CacheEntry:
    """Internal cache entry with timestamp for TTL management."""

    __slots__ = ("value", "created_at", "access_count")

    def __init__(self, value: Any):
        self.value = value
        self.created_at = time.time()
        self.access_count = 0


class InferenceCache:
    """
    LRU caching system for ControlNet inference optimization.

    Implements three specialized caches to avoid redundant computation:
    1. Text Embedding Cache: Caches CLIP text encoder outputs for repeated prompts
    2. Condition Map Cache: Caches processed condition maps for same image+type pairs
    3. Latent Cache: Caches final latents for deterministic generation (same seed+params)

    Cache keys are computed using content hashing (SHA-256) to ensure correctness.
    Each cache uses LRU eviction with configurable TTL for memory management.

    Design for HuggingFace Spaces:
    - Text embedding cache is most impactful (same prompts are common in demos)
    - Condition map cache helps when users adjust only generation params
    - Latent cache enables instant replay of previous generations

    Example:
        >>> cache = InferenceCache(CacheConfig(max_text_embeddings=128))
        >>> embedding = cache.get_text_embedding("a beautiful landscape")
        >>> if embedding is None:
        ...     embedding = text_encoder(tokenize("a beautiful landscape"))
        ...     cache.put_text_embedding("a beautiful landscape", embedding)
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize the inference cache system.

        Args:
            config: Cache configuration. Uses defaults if None.
        """
        self.config = config or CacheConfig()

        # LRU caches implemented as OrderedDicts
        self._text_embedding_cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._condition_map_cache: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._latent_cache: OrderedDict[str, _CacheEntry] = OrderedDict()

        # Thread safety for concurrent access
        self._lock = threading.Lock()

        # Statistics tracking
        self._stats = {
            "text_hits": 0,
            "text_misses": 0,
            "condition_hits": 0,
            "condition_misses": 0,
            "latent_hits": 0,
            "latent_misses": 0,
            "evictions": 0,
        }

        logger.info(
            f"InferenceCache initialized "
            f"(text={self.config.max_text_embeddings}, "
            f"condition={self.config.max_condition_maps}, "
            f"latent={self.config.max_latent_cache})"
        )

    # -------------------------------------------------------------------------
    # Text Embedding Cache
    # -------------------------------------------------------------------------

    def get_text_embedding(self, prompt: str) -> Optional[torch.Tensor]:
        """
        Retrieve cached text embedding for a prompt.

        Args:
            prompt: Text prompt to look up

        Returns:
            Cached embedding tensor, or None if not found/expired
        """
        key = self._compute_text_key(prompt)
        return self._get_from_cache(
            self._text_embedding_cache,
            key,
            self.config.text_embedding_ttl_seconds,
            "text",
        )

    def put_text_embedding(self, prompt: str, embedding: torch.Tensor) -> None:
        """
        Store a text embedding in the cache.

        Args:
            prompt: Text prompt (used as cache key)
            embedding: Computed embedding tensor to cache
        """
        key = self._compute_text_key(prompt)
        # Store as CPU tensor to save GPU memory
        cached_value = embedding.detach().cpu()
        self._put_in_cache(
            self._text_embedding_cache,
            key,
            cached_value,
            self.config.max_text_embeddings,
        )

    # -------------------------------------------------------------------------
    # Condition Map Cache
    # -------------------------------------------------------------------------

    def get_condition_map(
        self,
        image_hash: str,
        condition_type: str,
        target_size: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        """
        Retrieve cached processed condition map.

        Args:
            image_hash: Hash of the source image
            condition_type: Type of condition (depth, pose, edge)
            target_size: Target (height, width) for the condition map

        Returns:
            Cached condition tensor, or None if not found/expired
        """
        key = self._compute_condition_key(image_hash, condition_type, target_size)
        return self._get_from_cache(
            self._condition_map_cache,
            key,
            self.config.condition_map_ttl_seconds,
            "condition",
        )

    def put_condition_map(
        self,
        image_hash: str,
        condition_type: str,
        target_size: Tuple[int, int],
        condition_tensor: torch.Tensor,
    ) -> None:
        """
        Store a processed condition map in the cache.

        Args:
            image_hash: Hash of the source image
            condition_type: Type of condition (depth, pose, edge)
            target_size: Target (height, width) for the condition map
            condition_tensor: Processed condition tensor to cache
        """
        key = self._compute_condition_key(image_hash, condition_type, target_size)
        cached_value = condition_tensor.detach().cpu()
        self._put_in_cache(
            self._condition_map_cache,
            key,
            cached_value,
            self.config.max_condition_maps,
        )

    # -------------------------------------------------------------------------
    # Latent Cache (for deterministic generation)
    # -------------------------------------------------------------------------

    def get_latent(
        self,
        prompt: str,
        condition_hash: str,
        seed: int,
        num_steps: int,
        guidance_scale: float,
        conditioning_scale: float,
    ) -> Optional[torch.Tensor]:
        """
        Retrieve cached latent output for deterministic generation.

        If the same prompt, condition, seed, and parameters are used,
        the generation result is deterministic and can be cached.

        Args:
            prompt: Text prompt
            condition_hash: Hash of the condition map
            seed: Random seed used for generation
            num_steps: Number of inference steps
            guidance_scale: Classifier-free guidance scale
            conditioning_scale: ControlNet conditioning strength

        Returns:
            Cached latent tensor, or None if not found/expired
        """
        key = self._compute_latent_key(
            prompt, condition_hash, seed, num_steps, guidance_scale, conditioning_scale
        )
        return self._get_from_cache(
            self._latent_cache,
            key,
            self.config.latent_ttl_seconds,
            "latent",
        )

    def put_latent(
        self,
        prompt: str,
        condition_hash: str,
        seed: int,
        num_steps: int,
        guidance_scale: float,
        conditioning_scale: float,
        latent: torch.Tensor,
    ) -> None:
        """
        Store a generation latent in the cache.

        Args:
            prompt: Text prompt
            condition_hash: Hash of the condition map
            seed: Random seed used for generation
            num_steps: Number of inference steps
            guidance_scale: Classifier-free guidance scale
            conditioning_scale: ControlNet conditioning strength
            latent: Final latent tensor to cache
        """
        key = self._compute_latent_key(
            prompt, condition_hash, seed, num_steps, guidance_scale, conditioning_scale
        )
        cached_value = latent.detach().cpu()
        self._put_in_cache(
            self._latent_cache,
            key,
            cached_value,
            self.config.max_latent_cache,
        )

    # -------------------------------------------------------------------------
    # Cache Management
    # -------------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all caches and reset statistics."""
        with self._lock:
            self._text_embedding_cache.clear()
            self._condition_map_cache.clear()
            self._latent_cache.clear()
            self._stats = {k: 0 for k in self._stats}
        logger.info("All caches cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with hit/miss counts and hit rates for each cache
        """
        with self._lock:
            stats = dict(self._stats)

        # Compute hit rates
        for cache_type in ["text", "condition", "latent"]:
            hits = stats[f"{cache_type}_hits"]
            misses = stats[f"{cache_type}_misses"]
            total = hits + misses
            stats[f"{cache_type}_hit_rate"] = hits / total if total > 0 else 0.0

        stats["text_cache_size"] = len(self._text_embedding_cache)
        stats["condition_cache_size"] = len(self._condition_map_cache)
        stats["latent_cache_size"] = len(self._latent_cache)

        return stats

    def get_memory_usage_mb(self) -> float:
        """
        Estimate total memory used by cached tensors.

        Returns:
            Estimated memory usage in megabytes
        """
        total_bytes = 0

        with self._lock:
            for cache in [
                self._text_embedding_cache,
                self._condition_map_cache,
                self._latent_cache,
            ]:
                for entry in cache.values():
                    if isinstance(entry.value, torch.Tensor):
                        total_bytes += (
                            entry.value.nelement() * entry.value.element_size()
                        )

        return total_bytes / (1024**2)

    # -------------------------------------------------------------------------
    # Internal Cache Operations
    # -------------------------------------------------------------------------

    def _get_from_cache(
        self,
        cache: OrderedDict,
        key: str,
        ttl: float,
        cache_type: str,
    ) -> Optional[Any]:
        """Generic LRU cache get with TTL checking."""
        with self._lock:
            if key in cache:
                entry = cache[key]
                # Check TTL
                if time.time() - entry.created_at > ttl:
                    # Entry expired, remove it
                    del cache[key]
                    self._stats[f"{cache_type}_misses"] += 1
                    return None

                # Move to end (most recently used)
                cache.move_to_end(key)
                entry.access_count += 1
                self._stats[f"{cache_type}_hits"] += 1
                return entry.value
            else:
                self._stats[f"{cache_type}_misses"] += 1
                return None

    def _put_in_cache(
        self,
        cache: OrderedDict,
        key: str,
        value: Any,
        max_size: int,
    ) -> None:
        """Generic LRU cache put with eviction."""
        with self._lock:
            # If key exists, update it
            if key in cache:
                cache.move_to_end(key)
                cache[key] = _CacheEntry(value)
                return

            # Evict oldest entries if at capacity
            while len(cache) >= max_size:
                cache.popitem(last=False)
                self._stats["evictions"] += 1

            # Insert new entry
            cache[key] = _CacheEntry(value)

    # -------------------------------------------------------------------------
    # Key Computation
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_text_key(prompt: str) -> str:
        """Compute cache key for a text prompt."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _compute_condition_key(
        image_hash: str, condition_type: str, target_size: Tuple[int, int]
    ) -> str:
        """Compute cache key for a condition map."""
        raw = f"{image_hash}:{condition_type}:{target_size[0]}x{target_size[1]}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _compute_latent_key(
        prompt: str,
        condition_hash: str,
        seed: int,
        num_steps: int,
        guidance_scale: float,
        conditioning_scale: float,
    ) -> str:
        """Compute cache key for a deterministic generation result."""
        raw = (
            f"{prompt}:{condition_hash}:{seed}:"
            f"{num_steps}:{guidance_scale:.4f}:{conditioning_scale:.4f}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def compute_image_hash(image: Union[np.ndarray, "Image.Image"]) -> str:
        """
        Compute a content hash for an image (for use as cache key).

        Args:
            image: Input image as numpy array or PIL Image

        Returns:
            Hex string hash of the image content
        """
        if hasattr(image, "tobytes"):
            # PIL Image
            data = image.tobytes()
        elif isinstance(image, np.ndarray):
            data = image.tobytes()
        else:
            data = str(image).encode("utf-8")

        return hashlib.sha256(data).hexdigest()[:16]



# =============================================================================
# ConcurrencyManager Class
# =============================================================================


@dataclass
class InferenceRequest:
    """Represents a single inference request in the queue.

    Args:
        request_id: Unique identifier for the request
        priority: Request priority level
        params: Generation parameters
        submitted_at: Timestamp when request was submitted
        status: Current request status
        result: Generation result (populated on completion)
        error: Error message if request failed
        degraded: Whether quality was reduced due to load
    """
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    priority: RequestPriority = RequestPriority.NORMAL
    params: Optional[Dict[str, Any]] = None
    submitted_at: float = field(default_factory=time.time)
    status: RequestStatus = RequestStatus.QUEUED
    result: Optional[Any] = None
    error: Optional[str] = None
    degraded: bool = False


class ConcurrencyManager:
    """
    Manages concurrent inference requests with memory-aware scheduling.

    Designed for HuggingFace Spaces where multiple users may submit requests
    simultaneously. Implements:
    - Priority-based request queuing
    - Memory-aware scheduling (monitors GPU memory before accepting requests)
    - Graceful degradation under load (reduces quality settings automatically)
    - Request timeout and cleanup

    Degradation Strategy:
    When GPU memory exceeds the configured threshold or too many requests are
    active, the manager automatically reduces quality settings:
    1. Reduce inference steps by degradation_steps_reduction factor
    2. Scale down resolution by degradation_resolution_factor
    3. Disable non-essential features (attention slicing becomes more aggressive)

    This ensures the service remains responsive even under heavy load,
    trading some quality for availability.

    Example:
        >>> manager = ConcurrencyManager(ConcurrencyConfig(max_concurrent_requests=2))
        >>> request = manager.submit_request(params={"prompt": "hello"})
        >>> if manager.can_process():
        ...     adjusted_params = manager.acquire_slot(request)
        ...     # ... run inference with adjusted_params ...
        ...     manager.release_slot(request.request_id, result=output)
    """

    def __init__(self, config: Optional[ConcurrencyConfig] = None):
        """
        Initialize the concurrency manager.

        Args:
            config: Concurrency configuration. Uses defaults if None.
        """
        self.config = config or ConcurrencyConfig()

        # Request queue and tracking
        self._queue: List[InferenceRequest] = []
        self._active_requests: Dict[str, InferenceRequest] = {}
        self._completed_requests: OrderedDict[str, InferenceRequest] = OrderedDict()

        # Thread safety
        self._lock = threading.Lock()

        # State tracking
        self._active_count = 0
        self._total_processed = 0
        self._total_degraded = 0
        self._is_degraded = False

        logger.info(
            f"ConcurrencyManager initialized "
            f"(max_concurrent={self.config.max_concurrent_requests}, "
            f"max_queue={self.config.max_queue_size}, "
            f"memory_threshold={self.config.memory_threshold_gb}GB)"
        )

    def submit_request(
        self,
        params: Dict[str, Any],
        priority: RequestPriority = RequestPriority.NORMAL,
    ) -> Optional[InferenceRequest]:
        """
        Submit a new inference request to the queue.

        Args:
            params: Generation parameters for the request
            priority: Request priority level

        Returns:
            InferenceRequest object if accepted, None if queue is full
        """
        with self._lock:
            # Check queue capacity
            if len(self._queue) >= self.config.max_queue_size:
                logger.warning(
                    f"Request queue full ({self.config.max_queue_size}). "
                    f"Rejecting new request."
                )
                return None

            request = InferenceRequest(
                priority=priority,
                params=params,
            )
            self._queue.append(request)

            # Sort by priority if enabled
            if self.config.enable_priority_queue:
                priority_order = {
                    RequestPriority.HIGH: 0,
                    RequestPriority.NORMAL: 1,
                    RequestPriority.LOW: 2,
                }
                self._queue.sort(key=lambda r: priority_order[r.priority])

            logger.debug(
                f"Request {request.request_id} queued "
                f"(priority={priority.value}, queue_size={len(self._queue)})"
            )
            return request

    def can_process(self) -> bool:
        """
        Check if a new request can be processed.

        Considers both the concurrent request limit and available GPU memory.

        Returns:
            True if a request can be processed, False otherwise
        """
        with self._lock:
            # Check concurrent request limit
            if self._active_count >= self.config.max_concurrent_requests:
                return False

            # Check GPU memory availability
            if not self._check_memory_available():
                return False

            # Check if there are requests in the queue
            return len(self._queue) > 0

    def acquire_slot(
        self, request: Optional[InferenceRequest] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Acquire a processing slot for the next request.

        Dequeues the highest-priority request and returns its parameters,
        potentially adjusted for degradation if the system is under load.

        Args:
            request: Specific request to process (dequeues from front if None)

        Returns:
            Adjusted generation parameters, or None if no slot available
        """
        with self._lock:
            if self._active_count >= self.config.max_concurrent_requests:
                return None

            # Get request from queue
            if request is not None and request in self._queue:
                self._queue.remove(request)
                target_request = request
            elif self._queue:
                target_request = self._queue.pop(0)
            else:
                return None

            # Mark as processing
            target_request.status = RequestStatus.PROCESSING
            self._active_requests[target_request.request_id] = target_request
            self._active_count += 1

            # Apply degradation if needed
            params = dict(target_request.params) if target_request.params else {}
            if self._should_degrade():
                params = self._apply_degradation(params)
                target_request.degraded = True
                self._is_degraded = True
                logger.info(
                    f"Request {target_request.request_id} degraded due to load"
                )

            logger.debug(
                f"Slot acquired for {target_request.request_id} "
                f"(active={self._active_count})"
            )
            return params

    def release_slot(
        self,
        request_id: str,
        result: Optional[Any] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Release a processing slot after request completion.

        Args:
            request_id: ID of the completed request
            result: Generation result (if successful)
            error: Error message (if failed)
        """
        with self._lock:
            if request_id not in self._active_requests:
                logger.warning(f"Unknown request ID: {request_id}")
                return

            request = self._active_requests.pop(request_id)
            self._active_count -= 1
            self._total_processed += 1

            if error:
                request.status = RequestStatus.FAILED
                request.error = error
            elif request.degraded:
                request.status = RequestStatus.DEGRADED
                request.result = result
                self._total_degraded += 1
            else:
                request.status = RequestStatus.COMPLETED
                request.result = result

            # Store in completed (bounded)
            self._completed_requests[request_id] = request
            if len(self._completed_requests) > 100:
                self._completed_requests.popitem(last=False)

            # Check if we can exit degraded mode
            if self._is_degraded and not self._should_degrade():
                self._is_degraded = False
                logger.info("Exiting degraded mode - resources recovered")

            logger.debug(
                f"Slot released for {request_id} "
                f"(status={request.status.value}, active={self._active_count})"
            )

    def get_request_status(self, request_id: str) -> Optional[RequestStatus]:
        """
        Get the current status of a request.

        Args:
            request_id: ID of the request to check

        Returns:
            Current status, or None if request not found
        """
        with self._lock:
            # Check active requests
            if request_id in self._active_requests:
                return self._active_requests[request_id].status

            # Check completed requests
            if request_id in self._completed_requests:
                return self._completed_requests[request_id].status

            # Check queue
            for req in self._queue:
                if req.request_id == request_id:
                    return req.status

        return None

    def get_queue_position(self, request_id: str) -> Optional[int]:
        """
        Get the position of a request in the queue.

        Args:
            request_id: ID of the request

        Returns:
            Queue position (0-indexed), or None if not in queue
        """
        with self._lock:
            for i, req in enumerate(self._queue):
                if req.request_id == request_id:
                    return i
        return None

    def cleanup_expired(self) -> int:
        """
        Remove expired requests from the queue.

        Returns:
            Number of requests removed
        """
        removed = 0
        current_time = time.time()

        with self._lock:
            # Clean up timed-out queued requests
            expired = [
                req
                for req in self._queue
                if current_time - req.submitted_at > self.config.request_timeout_seconds
            ]
            for req in expired:
                req.status = RequestStatus.FAILED
                req.error = "Request timed out in queue"
                self._queue.remove(req)
                self._completed_requests[req.request_id] = req
                removed += 1

        if removed > 0:
            logger.info(f"Cleaned up {removed} expired requests")
        return removed

    def get_stats(self) -> Dict[str, Any]:
        """
        Get concurrency manager statistics.

        Returns:
            Dictionary with queue size, active count, and processing metrics
        """
        with self._lock:
            return {
                "queue_size": len(self._queue),
                "active_requests": self._active_count,
                "total_processed": self._total_processed,
                "total_degraded": self._total_degraded,
                "is_degraded": self._is_degraded,
                "degradation_rate": (
                    self._total_degraded / self._total_processed
                    if self._total_processed > 0
                    else 0.0
                ),
                "max_concurrent": self.config.max_concurrent_requests,
                "max_queue": self.config.max_queue_size,
            }

    def get_estimated_wait_time(self, request_id: str) -> Optional[float]:
        """
        Estimate wait time for a queued request.

        Based on average processing time and queue position.

        Args:
            request_id: ID of the request

        Returns:
            Estimated wait time in seconds, or None if not in queue
        """
        position = self.get_queue_position(request_id)
        if position is None:
            return None

        # Estimate based on position and concurrent capacity
        # Assume ~30 seconds per request on T4 GPU with 20 steps
        avg_processing_time = 30.0
        batches_ahead = (position + 1) / self.config.max_concurrent_requests
        return batches_ahead * avg_processing_time

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _check_memory_available(self) -> bool:
        """Check if GPU memory is below the threshold for accepting new requests."""
        if not torch.cuda.is_available():
            return True  # No GPU memory constraint on CPU

        try:
            memory_used_gb = torch.cuda.memory_allocated() / (1024**3)
            return memory_used_gb < self.config.memory_threshold_gb
        except Exception:
            # If we can't check memory, allow the request
            return True

    def _should_degrade(self) -> bool:
        """
        Determine if quality degradation should be applied.

        Degradation triggers:
        1. Active requests at or above max concurrent limit
        2. GPU memory above threshold
        3. Queue is more than 50% full
        """
        # Check concurrent load
        if self._active_count >= self.config.max_concurrent_requests:
            return True

        # Check memory pressure
        if torch.cuda.is_available():
            try:
                memory_used_gb = torch.cuda.memory_allocated() / (1024**3)
                if memory_used_gb > self.config.memory_threshold_gb * 0.9:
                    return True
            except Exception:
                pass

        # Check queue pressure
        if len(self._queue) > self.config.max_queue_size * 0.5:
            return True

        return False

    def _apply_degradation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply quality degradation to generation parameters.

        Reduces inference steps and resolution to maintain responsiveness
        under heavy load.

        Args:
            params: Original generation parameters

        Returns:
            Adjusted parameters with reduced quality settings
        """
        degraded_params = dict(params)

        # Reduce inference steps
        if "num_inference_steps" in degraded_params:
            original_steps = degraded_params["num_inference_steps"]
            reduced_steps = max(
                10,
                int(original_steps * self.config.degradation_steps_reduction),
            )
            degraded_params["num_inference_steps"] = reduced_steps
            logger.debug(f"Degraded steps: {original_steps} -> {reduced_steps}")

        # Reduce resolution
        for dim in ["height", "width"]:
            if dim in degraded_params:
                original = degraded_params[dim]
                # Round down to nearest multiple of 8 (required for VAE)
                reduced = int(original * self.config.degradation_resolution_factor)
                reduced = (reduced // 8) * 8
                reduced = max(256, reduced)  # Minimum 256px
                degraded_params[dim] = reduced
                logger.debug(f"Degraded {dim}: {original} -> {reduced}")

        # Mark as degraded for user notification
        degraded_params["_degraded"] = True
        degraded_params["_degradation_reason"] = "high_load"

        return degraded_params
