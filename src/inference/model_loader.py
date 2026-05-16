"""
Model Loading and Compatibility Verification

This module provides automatic model loading from HuggingFace Hub or local storage,
architecture compatibility verification, and graceful fallback for missing or
incompatible models. Designed for T4 GPU memory constraints (loads in FP16 by default).

Requirements satisfied: 10.3, 10.5

Features:
- Automatic model loading from HuggingFace Hub (using huggingface_hub library)
- Local model loading from file paths
- Architecture compatibility verification (check layer shapes, config matching)
- Graceful fallback when models are missing or incompatible
- Model caching to avoid repeated downloads
- Support for loading SD1.5 base model + ControlNet adapter
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


# Default model identifiers
DEFAULT_SD15_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_CONTROLNET_MODELS = {
    "depth": "lllyasviel/control_v11f1p_sd15_depth",
    "pose": "lllyasviel/control_v11p_sd15_openpose",
    "edge": "lllyasviel/control_v11p_sd15_canny",
}

# Expected architecture parameters for SD1.5 ControlNet
EXPECTED_SD15_CONFIG = {
    "cross_attention_dim": 768,
    "block_out_channels": [320, 640, 1280, 1280],
    "in_channels": 4,
    "layers_per_block": 2,
}


@dataclass
class ModelLoadResult:
    """Result of a model loading operation."""

    model: Any = None
    success: bool = False
    source: str = ""  # "hub", "local", "fallback"
    error_message: str = ""
    config: Optional[Dict[str, Any]] = None
    warnings: list = field(default_factory=list)

    @property
    def is_loaded(self) -> bool:
        return self.success and self.model is not None


@dataclass
class CompatibilityReport:
    """Report from architecture compatibility verification."""

    compatible: bool = True
    issues: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    checked_fields: Dict[str, bool] = field(default_factory=dict)

    def add_issue(self, issue: str):
        self.issues.append(issue)
        self.compatible = False

    def add_warning(self, warning: str):
        self.warnings.append(warning)

    def summary(self) -> str:
        status = "COMPATIBLE" if self.compatible else "INCOMPATIBLE"
        lines = [f"Compatibility: {status}"]
        if self.issues:
            lines.append(f"Issues ({len(self.issues)}):")
            for issue in self.issues:
                lines.append(f"  - {issue}")
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        return "\n".join(lines)


class ModelLoader:
    """
    Handles loading of SD1.5 base model and ControlNet adapters with
    compatibility verification and graceful fallback.

    Designed for T4 GPU memory constraints:
    - Loads models in FP16 by default to save ~50% memory
    - Supports model caching to avoid repeated downloads
    - Provides clear error messages for loading failures

    Usage:
        loader = ModelLoader(cache_dir="./cache/models")

        # Load ControlNet from HuggingFace Hub
        result = loader.load_controlnet("lllyasviel/control_v11f1p_sd15_depth")

        # Load from local path
        result = loader.load_controlnet("./models/trained/depth_controlnet")

        # Load SD1.5 base pipeline
        result = loader.load_base_pipeline()
    """

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        torch_dtype: torch.dtype = torch.float16,
        device: Optional[str] = None,
        use_auth_token: Optional[str] = None,
    ):
        """
        Initialize the model loader.

        Args:
            cache_dir: Directory for caching downloaded models.
                       Defaults to ./cache/models.
            torch_dtype: Data type for model loading. Defaults to float16
                         for T4 GPU memory efficiency.
            device: Target device ("cuda", "cpu", or None for auto-detect).
            use_auth_token: HuggingFace authentication token for private models.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache/models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.torch_dtype = torch_dtype
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_auth_token = use_auth_token

        # Track loaded models to avoid reloading
        self._loaded_models: Dict[str, Any] = {}

        logger.info(
            f"ModelLoader initialized: device={self.device}, "
            f"dtype={self.torch_dtype}, cache_dir={self.cache_dir}"
        )

    def load_controlnet(
        self,
        model_id_or_path: Optional[str] = None,
        condition_type: str = "depth",
        force_reload: bool = False,
        **kwargs,
    ) -> ModelLoadResult:
        """
        Load a ControlNet model from HuggingFace Hub or local storage.

        Attempts loading in this order:
        1. Local path (if model_id_or_path is a valid local directory)
        2. HuggingFace Hub (if model_id_or_path looks like a Hub ID)
        3. Default model for the condition type (fallback)

        Args:
            model_id_or_path: HuggingFace model ID or local path.
                              If None, uses the default for condition_type.
            condition_type: Type of conditioning (depth, pose, edge).
                           Used for default model selection.
            force_reload: If True, bypass cache and reload the model.
            **kwargs: Additional arguments passed to from_pretrained.

        Returns:
            ModelLoadResult with the loaded model or error details.
        """
        # Use default model if none specified
        if model_id_or_path is None:
            model_id_or_path = DEFAULT_CONTROLNET_MODELS.get(condition_type)
            if model_id_or_path is None:
                return ModelLoadResult(
                    success=False,
                    error_message=f"No default ControlNet model for condition type: {condition_type}",
                )

        # Check cache first
        cache_key = f"controlnet_{model_id_or_path}"
        if not force_reload and cache_key in self._loaded_models:
            logger.info(f"Using cached ControlNet: {model_id_or_path}")
            return ModelLoadResult(
                model=self._loaded_models[cache_key],
                success=True,
                source="cache",
            )

        # Try local path first
        local_path = Path(model_id_or_path)
        if local_path.exists() and local_path.is_dir():
            result = self._load_controlnet_local(local_path, **kwargs)
            if result.success:
                self._loaded_models[cache_key] = result.model
                return result

        # Try HuggingFace Hub
        result = self._load_controlnet_hub(model_id_or_path, **kwargs)
        if result.success:
            self._loaded_models[cache_key] = result.model
            return result

        # Fallback: try default model for condition type
        default_model_id = DEFAULT_CONTROLNET_MODELS.get(condition_type)
        if default_model_id and default_model_id != model_id_or_path:
            logger.warning(
                f"Failed to load '{model_id_or_path}', "
                f"falling back to default: {default_model_id}"
            )
            fallback_result = self._load_controlnet_hub(default_model_id, **kwargs)
            if fallback_result.success:
                fallback_result.source = "fallback"
                fallback_result.warnings.append(
                    f"Loaded fallback model '{default_model_id}' "
                    f"instead of requested '{model_id_or_path}'"
                )
                self._loaded_models[cache_key] = fallback_result.model
                return fallback_result

        # All attempts failed
        return ModelLoadResult(
            success=False,
            error_message=(
                f"Failed to load ControlNet model '{model_id_or_path}'. "
                f"Tried local path and HuggingFace Hub. "
                f"Original error: {result.error_message}"
            ),
        )

    def load_base_pipeline(
        self,
        model_id: str = DEFAULT_SD15_MODEL_ID,
        controlnet: Optional[Any] = None,
        force_reload: bool = False,
        **kwargs,
    ) -> ModelLoadResult:
        """
        Load the Stable Diffusion 1.5 base pipeline with optional ControlNet.

        Args:
            model_id: HuggingFace model ID for the base SD model.
            controlnet: Pre-loaded ControlNet model to integrate.
            force_reload: If True, bypass cache and reload.
            **kwargs: Additional arguments passed to from_pretrained.

        Returns:
            ModelLoadResult with the loaded pipeline or error details.
        """
        cache_key = f"pipeline_{model_id}"
        if not force_reload and cache_key in self._loaded_models:
            logger.info(f"Using cached pipeline: {model_id}")
            return ModelLoadResult(
                model=self._loaded_models[cache_key],
                success=True,
                source="cache",
            )

        try:
            from diffusers import (
                StableDiffusionControlNetPipeline,
                StableDiffusionPipeline,
            )

            load_kwargs = {
                "torch_dtype": self.torch_dtype,
                "cache_dir": str(self.cache_dir),
                "safety_checker": None,
            }
            if self.use_auth_token:
                load_kwargs["token"] = self.use_auth_token
            load_kwargs.update(kwargs)

            if controlnet is not None:
                pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                    model_id,
                    controlnet=controlnet,
                    **load_kwargs,
                )
            else:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    **load_kwargs,
                )

            # Move to device
            pipeline = pipeline.to(self.device)

            # Enable memory optimizations for T4
            if self.device == "cuda":
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xformers memory efficient attention")
                except Exception:
                    logger.info(
                        "xformers not available, using default attention"
                    )

            self._loaded_models[cache_key] = pipeline

            logger.info(f"Base pipeline loaded: {model_id}")
            return ModelLoadResult(
                model=pipeline,
                success=True,
                source="hub",
            )

        except Exception as e:
            error_msg = f"Failed to load base pipeline '{model_id}': {e}"
            logger.error(error_msg)
            return ModelLoadResult(
                success=False,
                error_message=error_msg,
            )

    def verify_compatibility(
        self,
        controlnet: Any,
        base_model_id: str = DEFAULT_SD15_MODEL_ID,
    ) -> CompatibilityReport:
        """
        Verify that a ControlNet model is compatible with the base SD1.5 model.

        Checks:
        - Cross attention dimension matches (768 for SD1.5)
        - Block output channels match expected architecture
        - Input channels are correct (4 for latent space)
        - Layer count per block matches

        Args:
            controlnet: The ControlNet model to verify.
            base_model_id: The base model ID to check compatibility against.

        Returns:
            CompatibilityReport with detailed findings.
        """
        report = CompatibilityReport()

        # Extract config from the model
        model_config = self._extract_model_config(controlnet)
        if model_config is None:
            report.add_issue(
                "Cannot extract model configuration. "
                "Model may not be a valid ControlNet."
            )
            return report

        # Check cross attention dimension
        cross_attn_dim = model_config.get("cross_attention_dim")
        expected_dim = EXPECTED_SD15_CONFIG["cross_attention_dim"]
        if cross_attn_dim is not None:
            if cross_attn_dim != expected_dim:
                report.add_issue(
                    f"Cross attention dimension mismatch: "
                    f"got {cross_attn_dim}, expected {expected_dim} for SD1.5"
                )
            report.checked_fields["cross_attention_dim"] = cross_attn_dim == expected_dim
        else:
            report.add_warning("Could not verify cross_attention_dim")

        # Check block output channels
        block_channels = model_config.get("block_out_channels")
        expected_channels = EXPECTED_SD15_CONFIG["block_out_channels"]
        if block_channels is not None:
            block_channels_list = list(block_channels) if not isinstance(block_channels, list) else block_channels
            if block_channels_list != expected_channels:
                report.add_issue(
                    f"Block output channels mismatch: "
                    f"got {block_channels_list}, expected {expected_channels}"
                )
            report.checked_fields["block_out_channels"] = block_channels_list == expected_channels
        else:
            report.add_warning("Could not verify block_out_channels")

        # Check input channels
        in_channels = model_config.get("in_channels")
        expected_in = EXPECTED_SD15_CONFIG["in_channels"]
        if in_channels is not None:
            if in_channels != expected_in:
                report.add_issue(
                    f"Input channels mismatch: "
                    f"got {in_channels}, expected {expected_in}"
                )
            report.checked_fields["in_channels"] = in_channels == expected_in
        else:
            report.add_warning("Could not verify in_channels")

        # Check layers per block
        layers_per_block = model_config.get("layers_per_block")
        expected_layers = EXPECTED_SD15_CONFIG["layers_per_block"]
        if layers_per_block is not None:
            if layers_per_block != expected_layers:
                report.add_warning(
                    f"Layers per block differs: "
                    f"got {layers_per_block}, expected {expected_layers}. "
                    f"This may still work but could affect quality."
                )
            report.checked_fields["layers_per_block"] = layers_per_block == expected_layers
        else:
            report.add_warning("Could not verify layers_per_block")

        # Log the result
        if report.compatible:
            logger.info("ControlNet compatibility check passed")
        else:
            logger.warning(f"ControlNet compatibility issues found:\n{report.summary()}")

        return report

    def load_and_verify_controlnet(
        self,
        model_id_or_path: Optional[str] = None,
        condition_type: str = "depth",
        strict: bool = False,
        **kwargs,
    ) -> ModelLoadResult:
        """
        Load a ControlNet model and verify its compatibility with SD1.5.

        This is the recommended method for loading ControlNet models as it
        combines loading with architecture verification.

        Args:
            model_id_or_path: HuggingFace model ID or local path.
            condition_type: Type of conditioning (depth, pose, edge).
            strict: If True, fail on any compatibility issue.
                    If False, only fail on critical issues.
            **kwargs: Additional arguments for model loading.

        Returns:
            ModelLoadResult with loaded model and compatibility info.
        """
        # Load the model
        result = self.load_controlnet(
            model_id_or_path=model_id_or_path,
            condition_type=condition_type,
            **kwargs,
        )

        if not result.success:
            return result

        # Verify compatibility
        report = self.verify_compatibility(result.model)

        if not report.compatible:
            if strict:
                return ModelLoadResult(
                    success=False,
                    error_message=(
                        f"ControlNet model is incompatible with SD1.5:\n"
                        f"{report.summary()}"
                    ),
                )
            else:
                # Non-strict mode: warn but still return the model
                result.warnings.extend(report.issues)
                logger.warning(
                    f"ControlNet loaded with compatibility warnings:\n"
                    f"{report.summary()}"
                )

        result.warnings.extend(report.warnings)
        return result

    def get_model_info(self, model_id_or_path: str) -> Dict[str, Any]:
        """
        Get information about a model without loading it.

        Args:
            model_id_or_path: HuggingFace model ID or local path.

        Returns:
            Dictionary with model information (config, metadata, etc.)
        """
        info: Dict[str, Any] = {"model_id": model_id_or_path, "available": False}

        # Check local path
        local_path = Path(model_id_or_path)
        if local_path.exists() and local_path.is_dir():
            info["source"] = "local"
            info["available"] = True

            # Read config if available
            config_path = local_path / "config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    info["config"] = json.load(f)

            # Read metadata if available
            metadata_path = local_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    info["metadata"] = json.load(f)

            return info

        # Check HuggingFace Hub
        try:
            from huggingface_hub import model_info as hf_model_info

            hub_info = hf_model_info(model_id_or_path)
            info["source"] = "hub"
            info["available"] = True
            info["model_id"] = hub_info.modelId
            info["author"] = hub_info.author
            info["tags"] = hub_info.tags
            info["last_modified"] = str(hub_info.lastModified) if hub_info.lastModified else None
        except Exception as e:
            info["error"] = str(e)

        return info

    def clear_cache(self, model_key: Optional[str] = None):
        """
        Clear cached models from memory.

        Args:
            model_key: Specific model key to clear. If None, clears all.
        """
        if model_key:
            if model_key in self._loaded_models:
                del self._loaded_models[model_key]
                logger.info(f"Cleared cached model: {model_key}")
        else:
            self._loaded_models.clear()
            logger.info("Cleared all cached models")

        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------

    def _load_controlnet_local(self, path: Path, **kwargs) -> ModelLoadResult:
        """Load ControlNet from a local directory."""
        try:
            from diffusers import ControlNetModel as DiffusersControlNet

            load_kwargs = {"torch_dtype": self.torch_dtype}
            load_kwargs.update(kwargs)

            # Check if it's a diffusers-format model
            config_path = path / "config.json"
            if config_path.exists():
                model = DiffusersControlNet.from_pretrained(
                    str(path), **load_kwargs
                )
                model = model.to(self.device)

                logger.info(f"ControlNet loaded from local path: {path}")
                return ModelLoadResult(
                    model=model,
                    success=True,
                    source="local",
                    config=self._extract_model_config(model),
                )

            # Try loading as a custom ControlNet (our implementation)
            from src.models.controlnet import ControlNetModel

            # Check for pytorch_model.bin or model.safetensors
            state_dict_path = path / "pytorch_model.bin"
            safetensors_path = path / "model.safetensors"

            if state_dict_path.exists():
                model = ControlNetModel.from_pretrained(str(path), **load_kwargs)
                model = model.to(self.device)
                logger.info(f"Custom ControlNet loaded from: {path}")
                return ModelLoadResult(
                    model=model,
                    success=True,
                    source="local",
                    config=self._extract_model_config(model),
                )
            elif safetensors_path.exists():
                model = ControlNetModel.from_pretrained(str(path), **load_kwargs)
                model = model.to(self.device)
                logger.info(f"Custom ControlNet loaded from: {path}")
                return ModelLoadResult(
                    model=model,
                    success=True,
                    source="local",
                    config=self._extract_model_config(model),
                )
            else:
                return ModelLoadResult(
                    success=False,
                    error_message=(
                        f"No model weights found in {path}. "
                        f"Expected config.json, pytorch_model.bin, or model.safetensors."
                    ),
                )

        except ImportError as e:
            return ModelLoadResult(
                success=False,
                error_message=f"Missing dependency for local loading: {e}",
            )
        except Exception as e:
            return ModelLoadResult(
                success=False,
                error_message=f"Failed to load ControlNet from {path}: {e}",
            )

    def _load_controlnet_hub(self, model_id: str, **kwargs) -> ModelLoadResult:
        """Load ControlNet from HuggingFace Hub."""
        try:
            from diffusers import ControlNetModel as DiffusersControlNet

            load_kwargs = {
                "torch_dtype": self.torch_dtype,
                "cache_dir": str(self.cache_dir),
            }
            if self.use_auth_token:
                load_kwargs["token"] = self.use_auth_token
            load_kwargs.update(kwargs)

            model = DiffusersControlNet.from_pretrained(model_id, **load_kwargs)
            model = model.to(self.device)

            logger.info(f"ControlNet loaded from Hub: {model_id}")
            return ModelLoadResult(
                model=model,
                success=True,
                source="hub",
                config=self._extract_model_config(model),
            )

        except OSError as e:
            # Model not found on Hub or network issue
            error_msg = str(e)
            if "404" in error_msg or "not found" in error_msg.lower():
                return ModelLoadResult(
                    success=False,
                    error_message=f"Model '{model_id}' not found on HuggingFace Hub.",
                )
            return ModelLoadResult(
                success=False,
                error_message=f"Network error loading '{model_id}': {e}",
            )
        except ImportError as e:
            return ModelLoadResult(
                success=False,
                error_message=f"Missing dependency for Hub loading: {e}",
            )
        except Exception as e:
            return ModelLoadResult(
                success=False,
                error_message=f"Failed to load ControlNet from Hub '{model_id}': {e}",
            )

    def _extract_model_config(self, model: Any) -> Optional[Dict[str, Any]]:
        """Extract configuration dictionary from a model object."""
        # Try diffusers config attribute
        if hasattr(model, "config"):
            config = model.config
            if isinstance(config, dict):
                return config
            elif hasattr(config, "to_dict"):
                return config.to_dict()
            elif hasattr(config, "__dict__"):
                # Filter out private attributes
                return {
                    k: v
                    for k, v in config.__dict__.items()
                    if not k.startswith("_")
                }

        # Try to read from config.json if model has a path
        if hasattr(model, "config_name"):
            config_path = getattr(model, "_name_or_path", None)
            if config_path:
                json_path = Path(config_path) / "config.json"
                if json_path.exists():
                    with open(json_path, "r") as f:
                        return json.load(f)

        return None


def load_models_for_inference(
    controlnet_model_id: Optional[str] = None,
    condition_type: str = "depth",
    base_model_id: str = DEFAULT_SD15_MODEL_ID,
    cache_dir: Optional[str] = None,
    device: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float16,
) -> Tuple[ModelLoadResult, ModelLoadResult]:
    """
    Convenience function to load both ControlNet and base pipeline for inference.

    This is the simplest way to get a working inference setup. It handles:
    - Loading the ControlNet adapter
    - Verifying compatibility with SD1.5
    - Loading the base pipeline with ControlNet integrated
    - Memory optimization for T4 GPU

    Args:
        controlnet_model_id: HuggingFace model ID or local path for ControlNet.
                             If None, uses the default for condition_type.
        condition_type: Type of conditioning (depth, pose, edge).
        base_model_id: HuggingFace model ID for the base SD model.
        cache_dir: Directory for caching downloaded models.
        device: Target device (None for auto-detect).
        torch_dtype: Data type for model loading.

    Returns:
        Tuple of (controlnet_result, pipeline_result).

    Example:
        controlnet_result, pipeline_result = load_models_for_inference(
            condition_type="depth"
        )
        if pipeline_result.success:
            pipeline = pipeline_result.model
            # Use pipeline for inference
    """
    loader = ModelLoader(
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Load and verify ControlNet
    controlnet_result = loader.load_and_verify_controlnet(
        model_id_or_path=controlnet_model_id,
        condition_type=condition_type,
    )

    if not controlnet_result.success:
        logger.error(f"ControlNet loading failed: {controlnet_result.error_message}")
        return controlnet_result, ModelLoadResult(
            success=False,
            error_message="Cannot load pipeline without ControlNet.",
        )

    # Load base pipeline with ControlNet
    pipeline_result = loader.load_base_pipeline(
        model_id=base_model_id,
        controlnet=controlnet_result.model,
    )

    return controlnet_result, pipeline_result
