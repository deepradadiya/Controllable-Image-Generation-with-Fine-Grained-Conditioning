"""
End-to-End Inference Pipeline for ControlNet Image Generation

This module implements the complete inference pipeline combining Stable Diffusion 1.5
with trained ControlNet models for conditioned image generation. It supports all three
condition types (depth, pose, edge) through a unified interface.

Key Features:
- DDIM sampling with ControlNet guidance integration
- Support for depth, pose, and edge conditioning types
- Unified interface for all condition types
- Memory-efficient inference optimized for T4 GPU (15GB VRAM)
- Configurable conditioning strength and generation parameters
- Deterministic generation with seed control

Requirements satisfied: 7.1, 7.2, 7.3, 7.4
"""

import gc
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class ConditionType(str, Enum):
    """Supported conditioning types for ControlNet inference."""
    DEPTH = "depth"
    POSE = "pose"
    EDGE = "edge"


@dataclass
class InferenceConfig:
    """Configuration for the ControlNet inference pipeline.

    Args:
        pretrained_model_path: Path or HuggingFace ID for SD1.5 model
        controlnet_model_path: Path to trained ControlNet checkpoint
        condition_type: Type of conditioning (depth, pose, edge)
        device: Computation device (auto, cuda, cpu)
        dtype: Model precision (float16 recommended for T4 GPU)
        enable_memory_optimization: Enable memory optimizations for T4 GPU
        enable_xformers: Enable xformers memory efficient attention
        scheduler_type: Noise scheduler type (ddim, pndm, euler)
    """
    pretrained_model_path: str = "runwayml/stable-diffusion-v1-5"
    controlnet_model_path: Optional[str] = None
    condition_type: str = "depth"
    device: str = "auto"
    dtype: str = "float16"
    enable_memory_optimization: bool = True
    enable_xformers: bool = False
    scheduler_type: str = "ddim"


@dataclass
class GenerationParams:
    """Parameters for image generation.

    Args:
        prompt: Text prompt for image generation
        negative_prompt: Negative text prompt for guidance
        num_inference_steps: Number of DDIM sampling steps
        guidance_scale: Classifier-free guidance scale
        conditioning_scale: ControlNet conditioning strength (0.0 to 2.0)
        height: Output image height in pixels
        width: Output image width in pixels
        seed: Random seed for reproducibility (None for random)
        num_images: Number of images to generate per prompt
        eta: DDIM eta parameter (0.0 for deterministic, 1.0 for stochastic)
    """
    prompt: str = ""
    negative_prompt: str = ""
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    conditioning_scale: float = 1.0
    height: int = 512
    width: int = 512
    seed: Optional[int] = None
    num_images: int = 1
    eta: float = 0.0


@dataclass
class GenerationResult:
    """Result of image generation.

    Args:
        images: List of generated PIL images
        condition_map: The condition map used for generation
        latents: Final latent tensors (optional, for debugging)
        generation_time_seconds: Total generation time
        memory_peak_mb: Peak GPU memory usage during generation
        seed_used: The random seed used for generation
    """
    images: List[Image.Image] = field(default_factory=list)
    condition_map: Optional[Image.Image] = None
    latents: Optional[torch.Tensor] = None
    generation_time_seconds: float = 0.0
    memory_peak_mb: float = 0.0
    seed_used: int = 0


class DDIMScheduler:
    """
    DDIM (Denoising Diffusion Implicit Models) scheduler for inference.

    Implements the DDIM sampling algorithm which allows for fewer sampling steps
    while maintaining generation quality. Supports deterministic and stochastic
    sampling through the eta parameter.

    Reference: Song et al., "Denoising Diffusion Implicit Models" (2020)
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        clip_sample: bool = False,
        set_alpha_to_one: bool = False,
        prediction_type: str = "epsilon",
    ):
        """
        Initialize DDIM scheduler.

        Args:
            num_train_timesteps: Number of training diffusion steps
            beta_start: Starting beta value for noise schedule
            beta_end: Ending beta value for noise schedule
            beta_schedule: Type of beta schedule (linear, scaled_linear)
            clip_sample: Whether to clip predicted samples
            set_alpha_to_one: Whether to set final alpha to 1.0
            prediction_type: Type of prediction (epsilon, v_prediction)
        """
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample

        # Compute beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "scaled_linear":
            # SD1.5 uses scaled linear schedule
            self.betas = torch.linspace(
                beta_start**0.5, beta_end**0.5, num_train_timesteps
            ) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # Compute alpha schedule
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Set final alpha
        self.final_alpha_cumprod = (
            torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        )

        # Timesteps will be set during set_timesteps
        self.timesteps = None
        self.num_inference_steps = None

    def set_timesteps(self, num_inference_steps: int, device: torch.device = None):
        """
        Set the discrete timesteps for DDIM sampling.

        Args:
            num_inference_steps: Number of inference steps
            device: Device to place timesteps on
        """
        self.num_inference_steps = num_inference_steps

        # Compute evenly spaced timesteps
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)
            .round()[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)

        if device is not None:
            self.timesteps = self.timesteps.to(device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Perform a single DDIM denoising step.

        Implements the DDIM update rule:
        x_{t-1} = sqrt(alpha_{t-1}) * predicted_x0
                  + sqrt(1 - alpha_{t-1} - sigma^2) * predicted_direction
                  + sigma * noise

        Args:
            model_output: Predicted noise from the UNet
            timestep: Current timestep
            sample: Current noisy sample x_t
            eta: DDIM eta parameter (0=deterministic, 1=DDPM equivalent)
            generator: Random number generator for reproducibility

        Returns:
            Denoised sample x_{t-1}
        """
        # Get current and previous timestep indices
        timestep_idx = (self.timesteps == timestep).nonzero(as_tuple=True)[0]
        prev_timestep_idx = timestep_idx + 1

        # Get alpha values
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[self.timesteps[prev_timestep_idx]]
            if prev_timestep_idx < len(self.timesteps)
            else self.final_alpha_cumprod
        )

        # Compute predicted original sample (x_0) from noise prediction
        if self.prediction_type == "epsilon":
            # x_0 = (x_t - sqrt(1 - alpha_t) * epsilon) / sqrt(alpha_t)
            pred_original_sample = (
                sample - (1 - alpha_prod_t) ** 0.5 * model_output
            ) / alpha_prod_t**0.5
        elif self.prediction_type == "v_prediction":
            # v-prediction parameterization
            pred_original_sample = (
                alpha_prod_t**0.5 * sample - (1 - alpha_prod_t) ** 0.5 * model_output
            )
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # Clip predicted x_0 if configured
        if self.clip_sample:
            pred_original_sample = pred_original_sample.clamp(-1, 1)

        # Compute variance (sigma) for stochastic sampling
        # sigma_t = eta * sqrt((1 - alpha_{t-1}) / (1 - alpha_t)) * sqrt(1 - alpha_t / alpha_{t-1})
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        sigma = eta * variance**0.5

        # Compute predicted direction pointing to x_t
        pred_sample_direction = (1 - alpha_prod_t_prev - sigma**2) ** 0.5 * model_output

        # Compute x_{t-1}
        prev_sample = (
            alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
        )

        # Add noise for stochastic sampling (eta > 0)
        if eta > 0:
            noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample + sigma * noise

        return prev_sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples according to the noise schedule.

        Args:
            original_samples: Clean samples to add noise to
            noise: Gaussian noise to add
            timesteps: Timesteps for noise level

        Returns:
            Noisy samples
        """
        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5

        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples


class ConditionProcessor:
    """
    Unified condition map processor for all conditioning types.

    Handles preprocessing of condition maps (depth, pose, edge) into the format
    expected by the ControlNet model. Supports both pre-extracted condition maps
    and on-the-fly extraction from source images.
    """

    SUPPORTED_TYPES = {ConditionType.DEPTH, ConditionType.POSE, ConditionType.EDGE}

    def __init__(self, condition_type: ConditionType, device: torch.device, dtype: torch.dtype):
        """
        Initialize condition processor.

        Args:
            condition_type: Type of conditioning to process
            device: Target device for tensors
            dtype: Target dtype for tensors
        """
        self.condition_type = condition_type
        self.device = device
        self.dtype = dtype

    def preprocess(
        self,
        condition_image: Union[Image.Image, np.ndarray, torch.Tensor],
        height: int = 512,
        width: int = 512,
    ) -> torch.Tensor:
        """
        Preprocess a condition map for ControlNet input.

        Converts various input formats into a normalized tensor suitable for
        ControlNet conditioning. Handles resizing, channel adjustment, and
        normalization based on the condition type.

        Args:
            condition_image: Input condition map (PIL Image, numpy array, or tensor)
            height: Target height for the condition map
            width: Target width for the condition map

        Returns:
            Preprocessed condition tensor of shape (1, C, H, W) normalized to [0, 1]
        """
        # Convert to numpy array
        if isinstance(condition_image, Image.Image):
            condition_image = np.array(condition_image)
        elif isinstance(condition_image, torch.Tensor):
            condition_image = condition_image.cpu().numpy()

        # Ensure float32 for processing
        if condition_image.dtype == np.uint8:
            condition_image = condition_image.astype(np.float32) / 255.0
        else:
            condition_image = condition_image.astype(np.float32)

        # Handle different channel configurations
        if condition_image.ndim == 2:
            # Single channel (H, W) -> (H, W, 3) by repeating
            condition_image = np.stack([condition_image] * 3, axis=-1)
        elif condition_image.ndim == 3 and condition_image.shape[2] == 1:
            # Single channel (H, W, 1) -> (H, W, 3)
            condition_image = np.concatenate([condition_image] * 3, axis=-1)
        elif condition_image.ndim == 3 and condition_image.shape[2] == 4:
            # RGBA -> RGB
            condition_image = condition_image[:, :, :3]

        # Resize to target dimensions
        if condition_image.shape[0] != height or condition_image.shape[1] != width:
            condition_pil = Image.fromarray(
                (condition_image * 255).astype(np.uint8)
            )
            condition_pil = condition_pil.resize((width, height), Image.BILINEAR)
            condition_image = np.array(condition_pil).astype(np.float32) / 255.0

        # Normalize to [0, 1] range
        cond_min = condition_image.min()
        cond_max = condition_image.max()
        if cond_max > cond_min:
            condition_image = (condition_image - cond_min) / (cond_max - cond_min)

        # Convert to tensor: (H, W, C) -> (1, C, H, W)
        condition_tensor = torch.from_numpy(condition_image).permute(2, 0, 1).unsqueeze(0)
        condition_tensor = condition_tensor.to(device=self.device, dtype=self.dtype)

        return condition_tensor

    def validate_condition(
        self, condition_image: Union[Image.Image, np.ndarray, torch.Tensor]
    ) -> Tuple[bool, str]:
        """
        Validate a condition map for compatibility.

        Args:
            condition_image: Condition map to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if condition_image is None:
            return False, "Condition image is None"

        if isinstance(condition_image, Image.Image):
            if condition_image.size[0] == 0 or condition_image.size[1] == 0:
                return False, "Condition image has zero dimensions"
        elif isinstance(condition_image, np.ndarray):
            if condition_image.size == 0:
                return False, "Condition array is empty"
            if np.any(np.isnan(condition_image)):
                return False, "Condition array contains NaN values"
            if np.any(np.isinf(condition_image)):
                return False, "Condition array contains infinite values"
        elif isinstance(condition_image, torch.Tensor):
            if condition_image.numel() == 0:
                return False, "Condition tensor is empty"
            if torch.any(torch.isnan(condition_image)):
                return False, "Condition tensor contains NaN values"

        return True, ""


class ControlNetInferencePipeline:
    """
    End-to-end inference pipeline combining Stable Diffusion 1.5 with trained ControlNet.

    This pipeline implements the complete image generation workflow:
    1. Load and configure SD1.5 components (VAE, UNet, text encoder, tokenizer)
    2. Load trained ControlNet model
    3. Process condition maps through the ControlNet
    4. Perform DDIM sampling with ControlNet guidance
    5. Decode latents to pixel space

    The pipeline supports all three condition types (depth, pose, edge) through
    a unified interface and is optimized for T4 GPU memory constraints.

    Example:
        >>> pipeline = ControlNetInferencePipeline(config)
        >>> result = pipeline.generate(
        ...     prompt="a beautiful landscape",
        ...     condition_image=depth_map,
        ...     params=GenerationParams(num_inference_steps=50)
        ... )
        >>> result.images[0].save("output.png")
    """

    def __init__(self, config: Optional[InferenceConfig] = None):
        """
        Initialize the inference pipeline.

        Args:
            config: Pipeline configuration. Uses defaults if None.
        """
        self.config = config or InferenceConfig()
        self.device = self._setup_device()
        self.dtype = self._get_dtype()

        # Model components (lazy loaded)
        self.vae = None
        self.unet = None
        self.text_encoder = None
        self.tokenizer = None
        self.controlnet = None
        self.scheduler = None
        self.condition_processor = None

        # State tracking
        self._models_loaded = False
        self._current_condition_type = ConditionType(self.config.condition_type)

        logger.info(
            f"ControlNetInferencePipeline initialized "
            f"(device={self.device}, dtype={self.dtype}, "
            f"condition_type={self._current_condition_type.value})"
        )

    def _setup_device(self) -> torch.device:
        """Determine computation device."""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                logger.info("CUDA not available, using CPU")
        else:
            device = torch.device(self.config.device)
        return device

    def _get_dtype(self) -> torch.dtype:
        """Get the model dtype based on configuration."""
        if self.config.dtype == "float16":
            if self.device.type == "cuda":
                return torch.float16
            else:
                logger.warning("float16 not recommended on CPU, using float32")
                return torch.float32
        elif self.config.dtype == "bfloat16":
            return torch.bfloat16
        return torch.float32

    def load_models(self) -> None:
        """
        Load all model components for inference.

        Loads the VAE, UNet, text encoder, tokenizer, and ControlNet model.
        Applies memory optimizations if configured (attention slicing, xformers).

        Raises:
            RuntimeError: If model loading fails
        """
        if self._models_loaded:
            logger.info("Models already loaded, skipping")
            return

        logger.info("Loading inference pipeline models...")
        start_time = time.time()

        try:
            from diffusers import AutoencoderKL
            from transformers import CLIPTextModel, CLIPTokenizer

            # Load tokenizer
            logger.info(f"Loading tokenizer from {self.config.pretrained_model_path}")
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.config.pretrained_model_path,
                subfolder="tokenizer",
            )

            # Load text encoder
            logger.info("Loading text encoder...")
            self.text_encoder = CLIPTextModel.from_pretrained(
                self.config.pretrained_model_path,
                subfolder="text_encoder",
                torch_dtype=self.dtype,
            ).to(self.device)
            self.text_encoder.eval()

            # Load VAE
            logger.info("Loading VAE...")
            self.vae = AutoencoderKL.from_pretrained(
                self.config.pretrained_model_path,
                subfolder="vae",
                torch_dtype=self.dtype,
            ).to(self.device)
            self.vae.eval()

            # Load UNet
            logger.info("Loading UNet...")
            from diffusers import UNet2DConditionModel

            self.unet = UNet2DConditionModel.from_pretrained(
                self.config.pretrained_model_path,
                subfolder="unet",
                torch_dtype=self.dtype,
            ).to(self.device)
            self.unet.eval()

            # Load ControlNet
            self._load_controlnet()

            # Initialize scheduler
            self.scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                set_alpha_to_one=False,
                prediction_type="epsilon",
            )

            # Initialize condition processor
            self.condition_processor = ConditionProcessor(
                condition_type=self._current_condition_type,
                device=self.device,
                dtype=self.dtype,
            )

            # Apply memory optimizations
            if self.config.enable_memory_optimization:
                self._apply_memory_optimizations()

            self._models_loaded = True
            load_time = time.time() - start_time
            logger.info(f"All models loaded in {load_time:.1f}s")

            if self.device.type == "cuda":
                memory_used = torch.cuda.memory_allocated() / (1024**3)
                logger.info(f"GPU memory used after loading: {memory_used:.2f} GB")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def _load_controlnet(self) -> None:
        """Load the ControlNet model from checkpoint or initialize fresh."""
        from src.models.controlnet import ControlNetModel

        if self.config.controlnet_model_path:
            logger.info(
                f"Loading ControlNet from {self.config.controlnet_model_path}"
            )
            try:
                self.controlnet = ControlNetModel.from_pretrained(
                    self.config.controlnet_model_path,
                    torch_dtype=self.dtype,
                ).to(self.device)
            except Exception as e:
                logger.warning(
                    f"Failed to load ControlNet from path, initializing fresh: {e}"
                )
                self.controlnet = ControlNetModel(
                    conditioning_channels=3,
                    in_channels=4,
                ).to(self.device)
        else:
            logger.info("No ControlNet path specified, initializing fresh model")
            self.controlnet = ControlNetModel(
                conditioning_channels=3,
                in_channels=4,
            ).to(self.device)

        self.controlnet.eval()
        if self.dtype == torch.float16:
            self.controlnet = self.controlnet.half()

    def _apply_memory_optimizations(self) -> None:
        """Apply memory optimizations for T4 GPU constraints."""
        logger.info("Applying memory optimizations...")

        # Enable attention slicing for reduced memory usage
        if hasattr(self.unet, "set_attention_slice"):
            self.unet.set_attention_slice("auto")
            logger.info("Enabled UNet attention slicing")

        # Enable xformers if available and configured
        if self.config.enable_xformers:
            try:
                self.unet.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xformers memory efficient attention")
            except Exception as e:
                logger.warning(f"xformers not available: {e}")

        # Disable gradient computation for all models
        for model in [self.vae, self.unet, self.text_encoder, self.controlnet]:
            if model is not None:
                for param in model.parameters():
                    param.requires_grad = False

    def _encode_prompt(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_images: int = 1,
    ) -> torch.Tensor:
        """
        Encode text prompts into CLIP embeddings for classifier-free guidance.

        Args:
            prompt: Positive text prompt
            negative_prompt: Negative text prompt for guidance
            num_images: Number of images (for batch dimension)

        Returns:
            Concatenated prompt embeddings [negative, positive] for CFG
        """
        # Tokenize positive prompt
        text_inputs = self.tokenizer(
            [prompt] * num_images,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.device)

        # Encode positive prompt
        with torch.no_grad():
            prompt_embeds = self.text_encoder(text_input_ids)[0]

        # Tokenize and encode negative prompt
        uncond_inputs = self.tokenizer(
            [negative_prompt] * num_images,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_input_ids = uncond_inputs.input_ids.to(self.device)

        with torch.no_grad():
            negative_prompt_embeds = self.text_encoder(uncond_input_ids)[0]

        # Concatenate for classifier-free guidance: [negative, positive]
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def _prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Prepare initial random latents for the diffusion process.

        Args:
            batch_size: Number of latents to generate
            height: Image height (will be divided by VAE scale factor 8)
            width: Image width (will be divided by VAE scale factor 8)
            generator: Random number generator for reproducibility

        Returns:
            Random latent tensor of shape (batch_size, 4, height//8, width//8)
        """
        latent_height = height // 8  # VAE downscale factor
        latent_width = width // 8
        shape = (batch_size, 4, latent_height, latent_width)

        latents = torch.randn(
            shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        # Scale initial noise by the scheduler's init noise sigma
        # For DDIM, this is typically 1.0
        return latents

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        condition_image: Union[Image.Image, np.ndarray, torch.Tensor],
        params: Optional[GenerationParams] = None,
        condition_type: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate images conditioned on a spatial condition map.

        This is the main entry point for image generation. It performs the full
        DDIM sampling loop with ControlNet guidance.

        Args:
            prompt: Text prompt describing the desired image
            condition_image: Condition map (depth, pose, or edge map)
            params: Generation parameters (uses defaults if None)
            condition_type: Override condition type for this generation

        Returns:
            GenerationResult containing generated images and metadata

        Raises:
            RuntimeError: If models are not loaded
            ValueError: If condition image is invalid
        """
        # Ensure models are loaded
        if not self._models_loaded:
            self.load_models()

        # Use default params if not provided
        if params is None:
            params = GenerationParams(prompt=prompt)
        else:
            params.prompt = prompt

        # Override condition type if specified
        active_condition_type = (
            ConditionType(condition_type)
            if condition_type
            else self._current_condition_type
        )

        # Validate condition image
        is_valid, error_msg = self.condition_processor.validate_condition(condition_image)
        if not is_valid:
            raise ValueError(f"Invalid condition image: {error_msg}")

        logger.info(
            f"Generating {params.num_images} image(s) with "
            f"{active_condition_type.value} conditioning, "
            f"{params.num_inference_steps} steps, "
            f"guidance_scale={params.guidance_scale}, "
            f"conditioning_scale={params.conditioning_scale}"
        )

        start_time = time.time()

        # Setup random generator for reproducibility
        generator = None
        seed = params.seed
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        # Preprocess condition map
        condition_tensor = self.condition_processor.preprocess(
            condition_image, height=params.height, width=params.width
        )

        # Encode text prompt
        prompt_embeds = self._encode_prompt(
            prompt=params.prompt,
            negative_prompt=params.negative_prompt,
            num_images=params.num_images,
        )

        # Prepare initial latents
        latents = self._prepare_latents(
            batch_size=params.num_images,
            height=params.height,
            width=params.width,
            generator=generator,
        )

        # Setup scheduler timesteps
        self.scheduler.set_timesteps(params.num_inference_steps, device=self.device)

        # DDIM sampling loop with ControlNet guidance
        latents = self._ddim_sampling_loop(
            latents=latents,
            prompt_embeds=prompt_embeds,
            condition_tensor=condition_tensor,
            params=params,
            generator=generator,
        )

        # Decode latents to pixel space
        images = self._decode_latents(latents)

        # Compute generation metrics
        generation_time = time.time() - start_time
        memory_peak = 0.0
        if self.device.type == "cuda":
            memory_peak = torch.cuda.max_memory_allocated() / (1024**2)
            torch.cuda.reset_peak_memory_stats()

        # Convert condition to PIL for result
        condition_pil = None
        if isinstance(condition_image, Image.Image):
            condition_pil = condition_image
        elif isinstance(condition_image, np.ndarray):
            if condition_image.dtype != np.uint8:
                condition_image_display = (condition_image * 255).astype(np.uint8)
            else:
                condition_image_display = condition_image
            if condition_image_display.ndim == 2:
                condition_pil = Image.fromarray(condition_image_display, mode="L")
            else:
                condition_pil = Image.fromarray(condition_image_display)

        result = GenerationResult(
            images=images,
            condition_map=condition_pil,
            latents=latents,
            generation_time_seconds=generation_time,
            memory_peak_mb=memory_peak,
            seed_used=seed,
        )

        logger.info(
            f"Generation complete: {len(images)} image(s) in "
            f"{generation_time:.2f}s, peak memory: {memory_peak:.0f} MB"
        )

        return result

    def _ddim_sampling_loop(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        condition_tensor: torch.Tensor,
        params: GenerationParams,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Perform the DDIM sampling loop with ControlNet guidance.

        This implements the core denoising loop where at each timestep:
        1. The ControlNet processes the condition map to produce guidance features
        2. The UNet predicts noise, incorporating ControlNet features
        3. Classifier-free guidance is applied
        4. The DDIM scheduler computes the next latent

        Args:
            latents: Initial noisy latents
            prompt_embeds: Text prompt embeddings [negative, positive]
            condition_tensor: Preprocessed condition map tensor
            params: Generation parameters
            generator: Random number generator

        Returns:
            Denoised latent tensor
        """
        num_images = params.num_images

        for i, t in enumerate(self.scheduler.timesteps):
            # Expand latents for classifier-free guidance (unconditional + conditional)
            latent_model_input = torch.cat([latents] * 2)

            # Get ControlNet outputs
            # Only apply ControlNet to the conditional branch
            controlnet_output = self.controlnet(
                sample=latents,
                timestep=t,
                encoder_hidden_states=prompt_embeds[num_images:],  # Positive prompt only
                controlnet_cond=condition_tensor,
                conditioning_scale=params.conditioning_scale,
                return_dict=True,
            )

            down_block_res_samples = controlnet_output["down_block_res_samples"]
            mid_block_res_sample = controlnet_output["mid_block_res_sample"]

            # Prepare ControlNet residuals for both unconditional and conditional
            # For unconditional: zero out ControlNet contribution
            # For conditional: use full ControlNet features
            zero_down_samples = [torch.zeros_like(s) for s in down_block_res_samples]
            zero_mid_sample = torch.zeros_like(mid_block_res_sample)

            combined_down_samples = [
                torch.cat([zero, cond])
                for zero, cond in zip(zero_down_samples, down_block_res_samples)
            ]
            combined_mid_sample = torch.cat([zero_mid_sample, mid_block_res_sample])

            # UNet noise prediction with ControlNet features
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=combined_down_samples,
                mid_block_additional_residual=combined_mid_sample,
                return_dict=False,
            )[0]

            # Apply classifier-free guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + params.guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # DDIM step
            latents = self.scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=latents,
                eta=params.eta,
                generator=generator,
            )

        return latents

    def _decode_latents(self, latents: torch.Tensor) -> List[Image.Image]:
        """
        Decode latent tensors to PIL images using the VAE decoder.

        Args:
            latents: Latent tensor of shape (B, 4, H/8, W/8)

        Returns:
            List of PIL images
        """
        # Scale latents by VAE scaling factor
        latents = latents / self.vae.config.scaling_factor

        # Decode latents
        with torch.no_grad():
            image_tensor = self.vae.decode(latents).sample

        # Convert to PIL images
        images = []
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        image_tensor = image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()

        for img_array in image_tensor:
            img_array = (img_array * 255).round().astype(np.uint8)
            images.append(Image.fromarray(img_array))

        return images

    def set_condition_type(self, condition_type: str) -> None:
        """
        Change the active condition type.

        Args:
            condition_type: New condition type (depth, pose, edge)
        """
        self._current_condition_type = ConditionType(condition_type)
        if self.condition_processor is not None:
            self.condition_processor = ConditionProcessor(
                condition_type=self._current_condition_type,
                device=self.device,
                dtype=self.dtype,
            )
        logger.info(f"Condition type set to: {condition_type}")

    def set_controlnet(self, controlnet_path: str) -> None:
        """
        Load a different ControlNet model.

        Args:
            controlnet_path: Path to the new ControlNet checkpoint
        """
        self.config.controlnet_model_path = controlnet_path
        self._load_controlnet()
        logger.info(f"ControlNet updated from: {controlnet_path}")

    def unload_models(self) -> None:
        """Unload all models to free GPU memory."""
        self.vae = None
        self.unet = None
        self.text_encoder = None
        self.tokenizer = None
        self.controlnet = None
        self._models_loaded = False

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

        logger.info("All models unloaded")

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage statistics.

        Returns:
            Dictionary with memory usage in MB
        """
        if self.device.type != "cuda":
            return {"allocated_mb": 0.0, "reserved_mb": 0.0, "total_mb": 0.0}

        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
            "total_mb": torch.cuda.get_device_properties(0).total_mem / (1024**2),
        }

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded and ready for inference."""
        return self._models_loaded

    @property
    def condition_type(self) -> str:
        """Get the current condition type."""
        return self._current_condition_type.value
