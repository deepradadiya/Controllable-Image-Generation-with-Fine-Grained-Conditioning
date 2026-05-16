"""
Conditioning Strength and Parameter Controls for ControlNet Inference

This module implements adjustable conditioning strength, generation parameter management,
scheduler integration with proper timestep handling, and batch inference support for
generating multiple images efficiently.

Designed for T4 GPU memory constraints (15GB VRAM, ~13GB usable).

Requirements satisfied: 7.5
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SchedulerType(str, Enum):
    """Supported noise scheduler types for inference."""
    DDIM = "ddim"
    PNDM = "pndm"
    EULER = "euler"
    EULER_ANCESTRAL = "euler_ancestral"
    DPM_SOLVER = "dpm_solver"


@dataclass
class GenerationParameters:
    """
    Generation parameters for ControlNet inference.

    Controls all aspects of the image generation process including
    conditioning strength, sampling steps, guidance scale, and output size.

    Attributes:
        conditioning_scale: Strength of ControlNet conditioning (0.0 to 2.0).
            0.0 = no conditioning (standard SD1.5 behavior)
            1.0 = full conditioning strength
            >1.0 = amplified conditioning (may cause artifacts)
        num_inference_steps: Number of denoising steps (higher = better quality, slower).
        guidance_scale: Classifier-free guidance scale (higher = more prompt adherence).
        seed: Random seed for reproducibility. None for random generation.
        image_size: Output image dimensions (width, height). Must be multiples of 8.
        negative_prompt: Text describing what to avoid in generation.
        eta: DDIM eta parameter for stochastic sampling (0.0 = deterministic).
    """

    conditioning_scale: float = 1.0
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    image_size: Tuple[int, int] = (512, 512)
    negative_prompt: str = "blurry, bad quality, distorted, deformed"
    eta: float = 0.0

    def __post_init__(self):
        """Validate parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate all generation parameters are within acceptable ranges.

        Raises:
            ValueError: If any parameter is out of range.
        """
        if not 0.0 <= self.conditioning_scale <= 2.0:
            raise ValueError(
                f"conditioning_scale must be between 0.0 and 2.0, got {self.conditioning_scale}"
            )

        if self.num_inference_steps < 1 or self.num_inference_steps > 1000:
            raise ValueError(
                f"num_inference_steps must be between 1 and 1000, got {self.num_inference_steps}"
            )

        if self.guidance_scale < 0.0:
            raise ValueError(
                f"guidance_scale must be non-negative, got {self.guidance_scale}"
            )

        width, height = self.image_size
        if width % 8 != 0 or height % 8 != 0:
            raise ValueError(
                f"image_size dimensions must be multiples of 8, got {self.image_size}"
            )

        if width < 64 or height < 64:
            raise ValueError(
                f"image_size dimensions must be at least 64, got {self.image_size}"
            )

        if width > 1024 or height > 1024:
            logger.warning(
                f"image_size {self.image_size} is large and may exceed T4 GPU memory. "
                "Consider using 512x512 or 768x768."
            )

        if not 0.0 <= self.eta <= 1.0:
            raise ValueError(f"eta must be between 0.0 and 1.0, got {self.eta}")


@dataclass
class ConditioningStrengthSchedule:
    """
    Schedule for varying conditioning strength across timesteps.

    Allows dynamic adjustment of conditioning strength during the denoising
    process. Early steps can use stronger conditioning for structure, while
    later steps can reduce it for finer details.

    Attributes:
        start_scale: Conditioning scale at the beginning of denoising.
        end_scale: Conditioning scale at the end of denoising.
        schedule_type: How to interpolate between start and end ('linear', 'cosine', 'constant').
    """

    start_scale: float = 1.0
    end_scale: float = 1.0
    schedule_type: str = "constant"

    def __post_init__(self):
        if not 0.0 <= self.start_scale <= 2.0:
            raise ValueError(
                f"start_scale must be between 0.0 and 2.0, got {self.start_scale}"
            )
        if not 0.0 <= self.end_scale <= 2.0:
            raise ValueError(
                f"end_scale must be between 0.0 and 2.0, got {self.end_scale}"
            )
        if self.schedule_type not in ("linear", "cosine", "constant"):
            raise ValueError(
                f"schedule_type must be 'linear', 'cosine', or 'constant', "
                f"got '{self.schedule_type}'"
            )

    def get_scale_at_step(self, step: int, total_steps: int) -> float:
        """
        Get the conditioning scale for a specific denoising step.

        Args:
            step: Current denoising step (0-indexed).
            total_steps: Total number of denoising steps.

        Returns:
            Conditioning scale for the given step.
        """
        if total_steps <= 1:
            return self.start_scale

        if self.schedule_type == "constant":
            return self.start_scale

        # Progress from 0.0 (start) to 1.0 (end)
        progress = step / (total_steps - 1)

        if self.schedule_type == "linear":
            return self.start_scale + (self.end_scale - self.start_scale) * progress

        elif self.schedule_type == "cosine":
            # Cosine interpolation for smoother transitions
            cosine_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
            return self.start_scale + (self.end_scale - self.start_scale) * cosine_progress

        return self.start_scale


class SchedulerManager:
    """
    Manages noise schedulers for the inference pipeline.

    Provides a unified interface for creating and configuring different
    scheduler types (DDIM, PNDM, Euler) with proper timestep handling.
    Designed for integration with the diffusers library.
    """

    # Mapping of scheduler types to their diffusers class names
    SCHEDULER_CLASSES = {
        SchedulerType.DDIM: "DDIMScheduler",
        SchedulerType.PNDM: "PNDMScheduler",
        SchedulerType.EULER: "EulerDiscreteScheduler",
        SchedulerType.EULER_ANCESTRAL: "EulerAncestralDiscreteScheduler",
        SchedulerType.DPM_SOLVER: "DPMSolverMultistepScheduler",
    }

    def __init__(
        self,
        scheduler_type: Union[SchedulerType, str] = SchedulerType.DDIM,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        prediction_type: str = "epsilon",
    ):
        """
        Initialize the scheduler manager.

        Args:
            scheduler_type: Type of scheduler to use.
            num_train_timesteps: Number of training timesteps.
            beta_start: Starting beta value for noise schedule.
            beta_end: Ending beta value for noise schedule.
            beta_schedule: Type of beta schedule ('linear', 'scaled_linear', 'squaredcos_cap_v2').
            prediction_type: What the model predicts ('epsilon', 'v_prediction', 'sample').
        """
        if isinstance(scheduler_type, str):
            scheduler_type = SchedulerType(scheduler_type)

        self.scheduler_type = scheduler_type
        self.num_train_timesteps = num_train_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        self._scheduler = None

        logger.info(f"SchedulerManager initialized with {scheduler_type.value} scheduler")

    def create_scheduler(self):
        """
        Create and return the configured scheduler instance.

        Returns:
            A diffusers scheduler instance configured with the stored parameters.

        Raises:
            ImportError: If the required diffusers scheduler class is not available.
        """
        scheduler_class_name = self.SCHEDULER_CLASSES[self.scheduler_type]

        try:
            import diffusers
            scheduler_class = getattr(diffusers, scheduler_class_name)
        except AttributeError:
            raise ImportError(
                f"Scheduler class '{scheduler_class_name}' not found in diffusers. "
                f"Please update the diffusers library."
            )

        # Common scheduler kwargs
        kwargs = {
            "num_train_timesteps": self.num_train_timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "beta_schedule": self.beta_schedule,
            "prediction_type": self.prediction_type,
        }

        # Scheduler-specific configurations
        if self.scheduler_type == SchedulerType.DDIM:
            kwargs["clip_sample"] = False
            kwargs["set_alpha_to_one"] = False

        elif self.scheduler_type == SchedulerType.PNDM:
            kwargs["skip_prk_steps"] = True

        elif self.scheduler_type in (SchedulerType.EULER, SchedulerType.EULER_ANCESTRAL):
            # Euler schedulers don't use clip_sample
            pass

        elif self.scheduler_type == SchedulerType.DPM_SOLVER:
            kwargs["algorithm_type"] = "dpmsolver++"
            kwargs["solver_order"] = 2

        self._scheduler = scheduler_class(**kwargs)
        logger.info(f"Created {scheduler_class_name} scheduler")
        return self._scheduler

    def get_scheduler(self):
        """
        Get the scheduler instance, creating it if necessary.

        Returns:
            The configured scheduler instance.
        """
        if self._scheduler is None:
            self.create_scheduler()
        return self._scheduler

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = "cpu"):
        """
        Set the timesteps for inference on the scheduler.

        Args:
            num_inference_steps: Number of denoising steps.
            device: Device to place timesteps on.
        """
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(num_inference_steps, device=device)
        logger.debug(
            f"Set {num_inference_steps} inference timesteps on {device}"
        )

    def get_timesteps(self) -> torch.Tensor:
        """
        Get the current timesteps from the scheduler.

        Returns:
            Tensor of timesteps for the denoising loop.

        Raises:
            RuntimeError: If timesteps haven't been set yet (scheduler has no
                timesteps attribute or it is empty).
        """
        scheduler = self.get_scheduler()
        if not hasattr(scheduler, "timesteps"):
            raise RuntimeError(
                "Timesteps not set. Call set_timesteps() before get_timesteps()."
            )
        timesteps = scheduler.timesteps
        if timesteps is None or (hasattr(timesteps, "__len__") and len(timesteps) == 0):
            raise RuntimeError(
                "Timesteps not set. Call set_timesteps() before get_timesteps()."
            )
        return timesteps

    def step(
        self,
        model_output: torch.Tensor,
        timestep: Union[int, torch.Tensor],
        sample: torch.Tensor,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> Any:
        """
        Perform a single scheduler step (denoising step).

        Args:
            model_output: The model's noise prediction.
            timestep: Current timestep.
            sample: Current noisy sample.
            eta: DDIM eta parameter (only used for DDIM scheduler).
            generator: Random number generator for reproducibility.

        Returns:
            Scheduler step output containing the denoised sample.
        """
        scheduler = self.get_scheduler()

        step_kwargs = {}
        if self.scheduler_type == SchedulerType.DDIM:
            step_kwargs["eta"] = eta
        if generator is not None:
            step_kwargs["generator"] = generator

        return scheduler.step(
            model_output=model_output,
            timestep=timestep,
            sample=sample,
            **step_kwargs,
        )

    @property
    def init_noise_sigma(self) -> float:
        """Get the initial noise sigma from the scheduler."""
        scheduler = self.get_scheduler()
        return scheduler.init_noise_sigma

    def scale_model_input(
        self, sample: torch.Tensor, timestep: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        Scale the model input according to the scheduler requirements.

        Some schedulers (like Euler) require scaling the input before passing
        to the model.

        Args:
            sample: The input sample tensor.
            timestep: Current timestep.

        Returns:
            Scaled sample tensor.
        """
        scheduler = self.get_scheduler()
        if hasattr(scheduler, "scale_model_input"):
            return scheduler.scale_model_input(sample, timestep)
        return sample


class BatchInferenceManager:
    """
    Manages batch inference for generating multiple images efficiently.

    Handles memory-aware batching, splitting large requests into manageable
    chunks that fit within T4 GPU memory constraints, and aggregating results.
    """

    # Approximate memory per image at different resolutions (in MB, FP16)
    MEMORY_PER_IMAGE_MB = {
        (512, 512): 2500,
        (768, 768): 5500,
        (1024, 1024): 9500,
    }

    def __init__(
        self,
        max_batch_size: int = 4,
        gpu_memory_limit_mb: float = 13000.0,
        enable_memory_optimization: bool = True,
    ):
        """
        Initialize the batch inference manager.

        Args:
            max_batch_size: Maximum number of images to generate in one batch.
            gpu_memory_limit_mb: Available GPU memory in MB.
            enable_memory_optimization: Whether to enable automatic batch size adjustment.
        """
        self.max_batch_size = max_batch_size
        self.gpu_memory_limit_mb = gpu_memory_limit_mb
        self.enable_memory_optimization = enable_memory_optimization

        logger.info(
            f"BatchInferenceManager initialized: max_batch={max_batch_size}, "
            f"memory_limit={gpu_memory_limit_mb}MB"
        )

    def compute_optimal_batch_size(
        self,
        image_size: Tuple[int, int],
        num_images: int,
    ) -> int:
        """
        Compute the optimal batch size based on image size and available memory.

        Args:
            image_size: Target image dimensions (width, height).
            num_images: Total number of images to generate.

        Returns:
            Optimal batch size that fits within memory constraints.
        """
        if not self.enable_memory_optimization:
            return min(num_images, self.max_batch_size)

        # Estimate memory per image based on resolution
        width, height = image_size
        pixels = width * height

        # Linear interpolation of memory usage based on pixel count
        base_pixels = 512 * 512
        base_memory = 2500.0  # MB for 512x512
        estimated_memory_per_image = base_memory * (pixels / base_pixels)

        # Account for model memory overhead (~3GB for SD1.5 + ControlNet in FP16)
        model_overhead_mb = 3000.0
        available_for_batch = self.gpu_memory_limit_mb - model_overhead_mb

        if available_for_batch <= 0:
            logger.warning("Insufficient GPU memory. Using batch size 1.")
            return 1

        # Calculate max batch size that fits in memory
        memory_limited_batch = max(1, int(available_for_batch / estimated_memory_per_image))

        optimal_batch = min(num_images, self.max_batch_size, memory_limited_batch)

        logger.debug(
            f"Optimal batch size: {optimal_batch} "
            f"(memory_limited={memory_limited_batch}, max={self.max_batch_size})"
        )

        return optimal_batch

    def create_batch_schedule(
        self,
        num_images: int,
        image_size: Tuple[int, int],
    ) -> List[int]:
        """
        Create a schedule of batch sizes for generating multiple images.

        Splits the total number of images into batches that fit within
        memory constraints.

        Args:
            num_images: Total number of images to generate.
            image_size: Target image dimensions.

        Returns:
            List of batch sizes for each inference round.
        """
        if num_images <= 0:
            return []

        batch_size = self.compute_optimal_batch_size(image_size, num_images)
        schedule = []
        remaining = num_images

        while remaining > 0:
            current_batch = min(remaining, batch_size)
            schedule.append(current_batch)
            remaining -= current_batch

        logger.info(
            f"Batch schedule for {num_images} images: {schedule} "
            f"({len(schedule)} rounds)"
        )

        return schedule

    def prepare_batch_inputs(
        self,
        prompts: Union[str, List[str]],
        condition_maps: Union[torch.Tensor, List[torch.Tensor]],
        params: GenerationParameters,
        batch_size: int,
        batch_index: int,
    ) -> Dict[str, Any]:
        """
        Prepare inputs for a single batch of inference.

        Args:
            prompts: Text prompt(s) for generation.
            condition_maps: Condition map tensor(s).
            params: Generation parameters.
            batch_size: Size of this batch.
            batch_index: Index of this batch in the schedule.

        Returns:
            Dictionary of prepared batch inputs.
        """
        # Handle single prompt -> replicate for batch
        if isinstance(prompts, str):
            batch_prompts = [prompts] * batch_size
        else:
            start_idx = batch_index * batch_size
            end_idx = start_idx + batch_size
            batch_prompts = prompts[start_idx:end_idx]

        # Handle condition maps
        if isinstance(condition_maps, torch.Tensor):
            if condition_maps.dim() == 3:
                # Single condition map -> replicate for batch
                batch_conditions = condition_maps.unsqueeze(0).expand(batch_size, -1, -1, -1)
            elif condition_maps.dim() == 4:
                if condition_maps.shape[0] == 1:
                    batch_conditions = condition_maps.expand(batch_size, -1, -1, -1)
                else:
                    start_idx = batch_index * batch_size
                    end_idx = start_idx + batch_size
                    batch_conditions = condition_maps[start_idx:end_idx]
            else:
                raise ValueError(
                    f"condition_maps must be 3D or 4D tensor, got {condition_maps.dim()}D"
                )
        elif isinstance(condition_maps, list):
            start_idx = batch_index * batch_size
            end_idx = start_idx + batch_size
            batch_conditions = torch.stack(condition_maps[start_idx:end_idx])
        else:
            raise TypeError(
                f"condition_maps must be a Tensor or list of Tensors, got {type(condition_maps)}"
            )

        # Prepare negative prompts
        if params.negative_prompt:
            batch_negative_prompts = [params.negative_prompt] * batch_size
        else:
            batch_negative_prompts = None

        # Determine seed for this batch
        if params.seed is not None:
            batch_seed = params.seed + batch_index
        else:
            batch_seed = None

        return {
            "prompts": batch_prompts,
            "negative_prompts": batch_negative_prompts,
            "condition_maps": batch_conditions,
            "seed": batch_seed,
            "batch_size": batch_size,
            "batch_index": batch_index,
        }

    def get_memory_estimate(self, image_size: Tuple[int, int], batch_size: int) -> float:
        """
        Estimate GPU memory usage for a given batch configuration.

        Args:
            image_size: Target image dimensions.
            batch_size: Number of images in the batch.

        Returns:
            Estimated memory usage in MB.
        """
        width, height = image_size
        pixels = width * height
        base_pixels = 512 * 512
        base_memory = 2500.0

        per_image_memory = base_memory * (pixels / base_pixels)
        model_overhead = 3000.0

        total_memory = model_overhead + (per_image_memory * batch_size)
        return total_memory


def create_generator(seed: Optional[int], device: Union[str, torch.device] = "cpu") -> Optional[torch.Generator]:
    """
    Create a torch Generator with the given seed for reproducible generation.

    Args:
        seed: Random seed. None returns None (non-deterministic).
        device: Device for the generator.

    Returns:
        Configured Generator or None if seed is None.
    """
    if seed is None:
        return None

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def prepare_latents(
    batch_size: int,
    num_channels: int,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: Union[str, torch.device],
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.Tensor] = None,
    scheduler_init_noise_sigma: float = 1.0,
) -> torch.Tensor:
    """
    Prepare initial latent noise for the denoising process.

    Args:
        batch_size: Number of images in the batch.
        num_channels: Number of latent channels (4 for SD1.5).
        height: Latent height (image_height // 8).
        width: Latent width (image_width // 8).
        dtype: Data type for the latents.
        device: Device to place latents on.
        generator: Random generator for reproducibility.
        latents: Pre-computed latents (if None, generates random noise).
        scheduler_init_noise_sigma: Initial noise sigma from the scheduler.

    Returns:
        Prepared latent tensor scaled by the scheduler's init_noise_sigma.
    """
    latent_shape = (batch_size, num_channels, height, width)

    if latents is None:
        latents = torch.randn(
            latent_shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    else:
        if latents.shape != latent_shape:
            raise ValueError(
                f"Provided latents shape {latents.shape} doesn't match "
                f"expected shape {latent_shape}"
            )
        latents = latents.to(device=device, dtype=dtype)

    # Scale by scheduler's init_noise_sigma
    latents = latents * scheduler_init_noise_sigma

    return latents


def apply_conditioning_scale(
    controlnet_outputs: Dict[str, Any],
    conditioning_scale: float,
) -> Dict[str, Any]:
    """
    Apply conditioning scale to ControlNet outputs.

    Scales the ControlNet feature maps by the conditioning strength factor.
    This allows fine-grained control over how strongly the condition map
    influences the generated image.

    Args:
        controlnet_outputs: Dictionary containing ControlNet feature maps.
            Expected keys: 'down_block_res_samples', 'mid_block_res_sample'
        conditioning_scale: Scale factor (0.0 to 2.0).

    Returns:
        Scaled ControlNet outputs dictionary.
    """
    if conditioning_scale == 1.0:
        return controlnet_outputs

    scaled_outputs = {}

    if "down_block_res_samples" in controlnet_outputs:
        scaled_outputs["down_block_res_samples"] = [
            sample * conditioning_scale
            for sample in controlnet_outputs["down_block_res_samples"]
        ]

    if "mid_block_res_sample" in controlnet_outputs:
        scaled_outputs["mid_block_res_sample"] = (
            controlnet_outputs["mid_block_res_sample"] * conditioning_scale
        )

    return scaled_outputs


def apply_scheduled_conditioning(
    controlnet_outputs: Dict[str, Any],
    schedule: ConditioningStrengthSchedule,
    current_step: int,
    total_steps: int,
) -> Dict[str, Any]:
    """
    Apply scheduled conditioning strength to ControlNet outputs.

    Uses the conditioning schedule to determine the appropriate scale
    for the current denoising step.

    Args:
        controlnet_outputs: Dictionary containing ControlNet feature maps.
        schedule: Conditioning strength schedule.
        current_step: Current denoising step (0-indexed).
        total_steps: Total number of denoising steps.

    Returns:
        Scaled ControlNet outputs.
    """
    scale = schedule.get_scale_at_step(current_step, total_steps)
    return apply_conditioning_scale(controlnet_outputs, scale)
