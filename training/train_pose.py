"""
Pose-Conditioned ControlNet Training Script
============================================

This script trains a ControlNet adapter for pose-conditioned image generation
on top of Stable Diffusion 1.5. The adapter learns to use pose skeleton maps
as spatial conditioning signals to guide image generation.

Architecture:
- SD1.5 VAE (frozen) — encodes images to latent space
- SD1.5 UNet (frozen) — predicts noise, receives ControlNet features
- CLIP text encoder (frozen) — encodes text prompts
- ControlNet adapter (TRAINABLE) — learns spatial conditioning from pose maps

Optimizer: AdamW, lr=1e-5, cosine schedule with 500 warmup steps
Precision: FP16 mixed precision (halves VRAM usage)
Gradient clipping: max_norm=1.0 (prevents exploding gradients)
Logging: W&B — loss, lr, samples every 250 steps
Checkpoints: Google Drive every 250 steps (max 3 retained)
Final: Upload to HuggingFace Hub "{username}/controlnet-sd15-pose"
"""

# ⚠️ WARNING: Estimated training time on T4 GPU is approximately 3 hours.
# Consider splitting across multiple Colab sessions.
# Use checkpoint saving (every 250 steps) and --resume_from_checkpoint to resume.

import argparse
import logging
import os
import sys
from typing import List, Optional

import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.controlnet import ControlNet
from model.pipeline import ControlNetPipeline
from training.utils import (
    TrainConfig,
    load_checkpoint,
    log_to_wandb,
    save_checkpoint,
    setup_optimizer,
    setup_scheduler,
    upload_to_hub,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Fixed validation prompts and condition images for W&B sample logging.
# These are used every 250 steps to visually track training progress.
# ─────────────────────────────────────────────────────────────────────────────
VALIDATION_PROMPTS = [
    "a person standing with arms raised",
    "a dancer in a ballet pose",
    "a person sitting on a bench",
    "two people walking side by side",
]


def init_wandb(train_config: TrainConfig) -> None:
    """
    Initialize a Weights & Biases run for experiment tracking.

    Logs training hyperparameters (lr, optimizer, schedule, etc.) as the
    run config so they're searchable and comparable across experiments.

    If W&B is not installed or login fails, training continues without
    experiment tracking — this is a non-fatal error.

    Args:
        train_config: The training configuration dataclass.
    """
    try:
        import wandb

        wandb.init(
            project="controlnet-pose",
            config={
                "condition_type": train_config.condition_type,
                "learning_rate": train_config.learning_rate,
                "optimizer": "AdamW",
                "betas": train_config.betas,
                "weight_decay": train_config.weight_decay,
                "lr_schedule": "cosine",
                "lr_warmup_steps": train_config.lr_warmup_steps,
                "max_grad_norm": train_config.max_grad_norm,
                "mixed_precision": train_config.mixed_precision,
                "max_train_steps": train_config.max_train_steps,
                "checkpoint_every": train_config.checkpoint_every,
                "gradient_accumulation_steps": train_config.gradient_accumulation_steps,
                "hub_model_id": train_config.hub_model_id,
            },
        )
        logger.info("W&B initialized successfully for project 'controlnet-pose'")
    except ImportError:
        logger.warning("wandb not installed — training will proceed without experiment tracking")
    except Exception as e:
        logger.warning(f"W&B initialization failed: {e}. Training will proceed without tracking.")


def generate_validation_samples(
    controlnet: ControlNet,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    device: torch.device,
) -> List[Image.Image]:
    """
    Generate 4 sample images from fixed validation prompts for visual inspection.

    Uses the ControlNetPipeline with fixed prompts and synthetic pose condition images
    to produce samples that can be logged to W&B every 250 steps. This allows
    visual tracking of how the model's conditioning ability improves over training.

    The condition images are synthetic pose-like stick figure patterns (deterministic)
    so that the same conditions are used at every logging step, making progress comparable.

    Args:
        controlnet: The ControlNet adapter (current training state).
        unet: Frozen SD1.5 UNet.
        vae: Frozen SD1.5 VAE.
        text_encoder: Frozen CLIP text encoder.
        tokenizer: CLIP tokenizer.
        device: Device to run inference on.

    Returns:
        List of 4 PIL Images generated from the fixed validation prompts.
    """
    # Create a DDIM scheduler for inference (faster than DDPM)
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
    )

    # Build the inference pipeline
    pipeline = ControlNetPipeline(
        controlnet=controlnet,
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )

    # Generate 4 fixed synthetic pose-like condition images (deterministic)
    # These are simple stick figure patterns that simulate pose skeletons
    condition_images = _create_fixed_condition_images()

    # Set models to eval mode for inference
    controlnet.eval()

    sample_images = []
    with torch.no_grad():
        for i, (prompt, condition) in enumerate(
            zip(VALIDATION_PROMPTS, condition_images)
        ):
            try:
                generated = pipeline(
                    text_prompt=prompt,
                    condition_image=condition,
                    condition_type="pose",
                    guidance_scale=7.5,
                    num_inference_steps=20,
                    seed=42 + i,  # Fixed seeds for reproducibility across steps
                )
                sample_images.append(generated)
            except Exception as e:
                logger.warning(f"Validation sample {i} generation failed: {e}")
                # Create a placeholder image on failure
                placeholder = Image.new("RGB", (512, 512), (128, 128, 128))
                sample_images.append(placeholder)

    # Restore training mode
    controlnet.train()

    return sample_images


def _create_fixed_condition_images() -> List[Image.Image]:
    """
    Create 4 fixed synthetic pose-like condition images for validation.

    These are deterministic stick figure patterns that simulate pose skeletons,
    ensuring consistent visual comparison across training steps. Pose images
    are RGB with colored lines representing body joints and limbs on a black
    background.

    Returns:
        List of 4 PIL Images (512x512 RGB) representing synthetic pose maps.
    """
    size = 512
    images = []

    # 1. Standing pose — vertical line with arms extended horizontally
    img1 = np.zeros((size, size, 3), dtype=np.uint8)
    # Spine (vertical line, white)
    img1[100:400, 254:258, :] = 255
    # Arms (horizontal line, red)
    img1[180:184, 150:362, 0] = 255
    # Head (circle approximation, green)
    y, x = np.ogrid[:size, :size]
    head_mask = ((x - 256) ** 2 + (y - 80) ** 2) < 25 ** 2
    img1[head_mask, 1] = 255
    # Legs (V shape, blue)
    for i in range(100):
        img1[400 + i, 256 - i : 258 - i, 2] = 255
        img1[400 + i, 256 + i : 258 + i, 2] = 255
    images.append(Image.fromarray(img1, mode="RGB"))

    # 2. Arms raised pose — T-shape with arms up
    img2 = np.zeros((size, size, 3), dtype=np.uint8)
    # Spine (white)
    img2[150:400, 254:258, :] = 255
    # Arms raised (yellow)
    img2[80:184, 180:184, 0] = 255
    img2[80:184, 180:184, 1] = 255
    img2[80:184, 328:332, 0] = 255
    img2[80:184, 328:332, 1] = 255
    # Head (green)
    head_mask2 = ((x - 256) ** 2 + (y - 130) ** 2) < 25 ** 2
    img2[head_mask2, 1] = 255
    # Legs (cyan)
    img2[400:480, 230:234, 1] = 255
    img2[400:480, 230:234, 2] = 255
    img2[400:480, 278:282, 1] = 255
    img2[400:480, 278:282, 2] = 255
    images.append(Image.fromarray(img2, mode="RGB"))

    # 3. Sitting pose — bent legs
    img3 = np.zeros((size, size, 3), dtype=np.uint8)
    # Spine (white)
    img3[100:300, 254:258, :] = 255
    # Arms down (magenta)
    img3[180:300, 200:204, 0] = 255
    img3[180:300, 200:204, 2] = 255
    img3[180:300, 308:312, 0] = 255
    img3[180:300, 308:312, 2] = 255
    # Head (green)
    head_mask3 = ((x - 256) ** 2 + (y - 80) ** 2) < 25 ** 2
    img3[head_mask3, 1] = 255
    # Bent legs (horizontal, blue)
    img3[298:302, 200:310, 2] = 255
    # Lower legs (vertical, blue)
    img3[300:400, 200:204, 2] = 255
    img3[300:400, 306:310, 2] = 255
    images.append(Image.fromarray(img3, mode="RGB"))

    # 4. Walking pose — asymmetric limbs
    img4 = np.zeros((size, size, 3), dtype=np.uint8)
    # Spine (white)
    img4[100:350, 254:258, :] = 255
    # Head (green)
    head_mask4 = ((x - 256) ** 2 + (y - 80) ** 2) < 25 ** 2
    img4[head_mask4, 1] = 255
    # Left arm forward (red)
    for i in range(80):
        img4[180 + i // 2, 256 - i : 258 - i, 0] = 255
    # Right arm back (red)
    for i in range(80):
        img4[180 + i // 2, 256 + i : 258 + i, 0] = 255
    # Left leg forward (blue)
    for i in range(120):
        img4[350 + i, 256 - i // 2 : 258 - i // 2, 2] = 255
    # Right leg back (blue)
    for i in range(120):
        img4[350 + i, 256 + i // 2 : 258 + i // 2, 2] = 255
    images.append(Image.fromarray(img4, mode="RGB"))

    return images


def log_and_checkpoint(
    step: int,
    loss: float,
    controlnet: ControlNet,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    unet: UNet2DConditionModel,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    train_config: TrainConfig,
    device: torch.device,
) -> None:
    """
    Perform logging and checkpointing at the current training step.

    Called every 250 steps during training to:
    1. Generate 4 validation sample images for visual inspection
    2. Log loss, learning_rate, and sample_images to W&B
    3. Save a checkpoint (model state_dict, optimizer state, step) to Google Drive

    This function is designed to be called from within the training loop
    at every checkpoint_every interval.

    Args:
        step: Current global training step.
        loss: Current training loss value.
        controlnet: The ControlNet adapter model.
        optimizer: The AdamW optimizer.
        lr_scheduler: The cosine LR scheduler.
        unet: Frozen SD1.5 UNet (needed for sample generation).
        vae: Frozen SD1.5 VAE (needed for sample generation).
        text_encoder: Frozen CLIP text encoder (needed for sample generation).
        tokenizer: CLIP tokenizer (needed for sample generation).
        train_config: Training configuration.
        device: Device to run on.
    """
    logger.info(f"Step {step}: Logging metrics and saving checkpoint...")

    # Get current learning rate from scheduler
    current_lr = lr_scheduler.get_last_lr()[0]

    # Generate 4 validation sample images for visual tracking
    sample_images = generate_validation_samples(
        controlnet=controlnet,
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        device=device,
    )

    # Log loss, learning_rate, and sample_images to W&B
    log_to_wandb(
        step=step,
        loss=loss,
        lr=current_lr,
        sample_images=sample_images,
    )

    # Save checkpoint to Google Drive every 250 steps
    # Contains: ControlNet state_dict, optimizer state, LR scheduler state, step
    save_checkpoint(
        model=controlnet,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        step=step,
        output_dir=train_config.drive_checkpoint_path,
        max_checkpoints=train_config.max_checkpoints,
    )

    logger.info(
        f"Step {step}: loss={loss:.6f}, lr={current_lr:.2e}, "
        f"checkpoint saved to {train_config.drive_checkpoint_path}"
    )


def finalize_training(
    controlnet: ControlNet,
    train_config: TrainConfig,
) -> None:
    """
    Finalize training: save adapter weights and upload to HuggingFace Hub.

    Called at the end of training (after all steps complete) to:
    1. Save the final adapter weights in safetensors format
    2. Upload to HuggingFace Hub at "{username}/controlnet-sd15-pose"
    3. Close the W&B run

    If Hub upload fails (auth, network), weights are saved locally and
    a warning is logged — training results are never lost.

    Args:
        controlnet: The trained ControlNet adapter model.
        train_config: Training configuration with hub_model_id and output_dir.
    """
    logger.info("Training complete. Saving final model and uploading to Hub...")

    # Upload adapter weights to HuggingFace Hub
    # Saves in safetensors format with a model card containing training metadata
    upload_to_hub(
        model=controlnet,
        hub_model_id=train_config.hub_model_id,
        condition_type=train_config.condition_type,
        training_config=train_config,
    )

    # Close W&B run
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
            logger.info("W&B run finished successfully")
    except (ImportError, Exception) as e:
        logger.warning(f"W&B finish failed: {e}")

    logger.info(
        f"Training finalized. Model uploaded to Hub: {train_config.hub_model_id}"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for pose ControlNet training."""
    parser = argparse.ArgumentParser(
        description="Train a pose-conditioned ControlNet adapter for SD1.5"
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained SD1.5 model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./controlnet-pose-model",
        help="Directory to save trained model and checkpoints.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="{username}/controlnet-sd15-pose",
        help="HuggingFace Hub repository ID for uploading the trained model.",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=10000,
        help="Total number of training steps to perform.",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=250,
        help="Save a checkpoint every N training steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from.",
    )
    parser.add_argument(
        "--drive_checkpoint_path",
        type=str,
        default="/content/drive/MyDrive/controlnet_checkpoints",
        help="Google Drive path for saving checkpoints (Colab environment).",
    )

    return parser.parse_args()


def train_pose():
    """
    Main training function for pose-conditioned ControlNet.

    ⚠️ WARNING: Estimated training time on T4 GPU is ~3 hours.
    Consider splitting across multiple Colab sessions.
    Use checkpoint saving (every 250 steps) to resume.

    This function:
    1. Loads frozen SD1.5 components (VAE, UNet, CLIP text encoder)
    2. Creates a trainable ControlNet adapter
    3. Sets up AdamW optimizer with cosine LR schedule
    4. Sets up mixed precision training with GradScaler
    5. Optionally resumes from a checkpoint
    6. Runs the training loop
    """
    args = parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Starting pose-conditioned ControlNet training")
    logger.info(f"Pretrained model: {args.pretrained_model_name_or_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Max training steps: {args.max_train_steps}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Load frozen SD1.5 models
    #    ALL pretrained components are frozen (requires_grad=False) — their
    #    weights never change during training. Only the ControlNet adapter
    #    parameters will be updated.
    # ─────────────────────────────────────────────────────────────────────────

    # Load VAE — encodes images from pixel space to latent space (frozen)
    logger.info("Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    vae.requires_grad_(False)
    vae.eval()

    # Load UNet — the noise prediction backbone (frozen)
    logger.info("Loading UNet...")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    unet.requires_grad_(False)
    unet.eval()

    # Load CLIP text encoder — encodes text prompts to embeddings (frozen)
    logger.info("Loading CLIP text encoder...")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    text_encoder.requires_grad_(False)
    text_encoder.eval()

    # Load tokenizer for text encoding
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Create trainable ControlNet adapter
    #    The ControlNet copies the UNet encoder and adds zero convolutions.
    #    Only the condition_embedding and zero_conv layers are trainable.
    #    condition_channels=3 for pose maps (RGB pose skeletons).
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Creating ControlNet adapter...")
    controlnet = ControlNet(unet=unet, condition_channels=3)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Set up optimizer — AdamW with only ControlNet parameters
    #    Uses shared utility to ensure consistent config across all training
    #    scripts (depth, pose, edge).
    #    Config: lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-2
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Setting up optimizer (AdamW, lr=1e-5, only ControlNet params)...")
    optimizer = setup_optimizer(controlnet.parameters())

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Set up cosine LR schedule with warmup
    #    - Linear warmup from 0 to 1e-5 over first 500 steps
    #    - Cosine decay from 1e-5 toward 0 for remaining steps
    #    This prevents early training instability and allows smooth convergence.
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Setting up cosine LR schedule with 500 warmup steps...")
    lr_scheduler = setup_scheduler(
        optimizer=optimizer,
        num_training_steps=args.max_train_steps,
        warmup_steps=500,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Set up mixed precision training with GradScaler
    #    FP16 mixed precision halves VRAM usage by storing activations in
    #    float16 instead of float32. The GradScaler handles loss scaling to
    #    prevent underflow in FP16 gradients.
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("Setting up GradScaler for FP16 mixed precision training...")
    scaler = torch.cuda.amp.GradScaler()

    # Load noise scheduler for adding noise during training
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Resume from checkpoint if specified
    #    Restores model weights, optimizer state, LR scheduler state, and
    #    step counter so training continues exactly where it left off.
    # ─────────────────────────────────────────────────────────────────────────
    global_step = 0
    if args.resume_from_checkpoint is not None:
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        global_step = load_checkpoint(
            model=controlnet,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            checkpoint_path=args.resume_from_checkpoint,
        )
        logger.info(f"Resumed training from step {global_step}")

    # Move models to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device)
    unet.to(device)
    text_encoder.to(device)
    controlnet.to(device)

    logger.info(f"All models loaded on device: {device}")
    logger.info(f"Training will run for {args.max_train_steps - global_step} remaining steps")

    # Create training config for shared utilities
    train_config = TrainConfig(
        condition_type="pose",
        max_train_steps=args.max_train_steps,
        checkpoint_every=args.checkpoint_every,
        hub_model_id=args.hub_model_id,
        output_dir=args.output_dir,
        drive_checkpoint_path=args.drive_checkpoint_path,
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Initialize W&B for experiment tracking
    #    Logs training config as run parameters for searchability.
    #    Non-fatal if W&B is unavailable — training proceeds without tracking.
    # ─────────────────────────────────────────────────────────────────────────
    init_wandb(train_config)

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Training loop
    #    - Encode images to latent space with VAE
    #    - Sample random noise and timesteps
    #    - Encode text prompts with CLIP
    #    - Run ControlNet forward pass with pose condition
    #    - Compute MSE loss between predicted and actual noise
    #    - Backpropagate only through ControlNet parameters
    #    - Apply gradient clipping (max_norm=1.0)
    #    - Log to W&B and save checkpoints every 250 steps
    #    - Upload final model to HuggingFace Hub
    # ─────────────────────────────────────────────────────────────────────────

    from training.losses import compute_diffusion_loss

    # Gradient accumulation: batch_size=1, accumulate over 8 steps
    # for an effective batch size of 8
    gradient_accumulation_steps = train_config.gradient_accumulation_steps

    # Simple synthetic dataloader for now (real dataset loading is separate).
    # Generates random training batches with image, condition, and text.
    def synthetic_dataloader(device, tokenizer, batch_size=1):
        """
        Yields synthetic training batches for development/testing.
        Each batch contains:
          - image: random (B, 3, 512, 512) tensor normalized to [-1, 1]
          - condition: random (B, 3, 512, 512) pose map normalized to [0, 1]
          - input_ids: tokenized text prompt (B, 77)
        """
        sample_prompts = [
            "a person doing yoga in a park",
            "a dancer performing on stage",
            "a person jogging along a path",
            "an athlete stretching before exercise",
        ]
        while True:
            image = torch.randn(batch_size, 3, 512, 512, device=device)
            condition = torch.rand(batch_size, 3, 512, 512, device=device)
            prompt = sample_prompts[torch.randint(0, len(sample_prompts), (1,)).item()]
            tokens = tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokens.input_ids.to(device)
            yield {
                "image": image,
                "condition": condition,
                "input_ids": input_ids,
            }

    dataloader = synthetic_dataloader(device, tokenizer, batch_size=1)

    logger.info("Starting training loop...")
    logger.info(
        f"Gradient accumulation steps: {gradient_accumulation_steps} "
        f"(effective batch size = {gradient_accumulation_steps})"
    )

    controlnet.train()
    optimizer.zero_grad()

    for step in range(global_step, args.max_train_steps):
        batch = next(dataloader)
        image = batch["image"]
        condition = batch["condition"]
        input_ids = batch["input_ids"]

        # ─────────────────────────────────────────────────────────────────
        # Forward pass wrapped in mixed precision autocast
        # FP16 halves VRAM by storing activations in float16
        # ─────────────────────────────────────────────────────────────────
        with torch.autocast("cuda", dtype=torch.float16):
            # (a) Encode image to latent space with VAE
            latents = vae.encode(image).latent_dist.sample() * 0.18215

            # (b) Sample random noise
            noise = torch.randn_like(latents)

            # (c) Sample random timestep in [0, 999]
            batch_size = latents.shape[0]
            timesteps = torch.randint(
                0, 1000, (batch_size,), device=device, dtype=torch.long
            )

            # (d) Add noise to latent at the sampled timestep
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # (e) Encode text prompt with CLIP (max 77 tokens)
            text_emb = text_encoder(input_ids)[0]

            # (f) Run ControlNet forward pass with pose condition image
            controlnet_output = controlnet(
                noisy_latents, timesteps, text_emb, condition
            )

            # (g) Run frozen UNet with ControlNet features injected
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_emb,
                down_block_additional_residuals=controlnet_output[
                    "down_block_res_samples"
                ],
                mid_block_additional_residual=controlnet_output[
                    "mid_block_res_sample"
                ],
            ).sample

            # (h) Compute MSE loss between predicted and actual noise
            loss = compute_diffusion_loss(noise_pred, noise, timesteps, step)

            # Divide loss by gradient accumulation steps
            loss = loss / gradient_accumulation_steps

        # ─────────────────────────────────────────────────────────────────
        # Backward pass — gradients flow ONLY through ControlNet adapter
        # (frozen models have requires_grad=False, so they get no gradients)
        # ─────────────────────────────────────────────────────────────────
        scaler.scale(loss).backward()

        # ─────────────────────────────────────────────────────────────────
        # Optimizer step after accumulating gradients
        # ─────────────────────────────────────────────────────────────────
        if (step + 1) % gradient_accumulation_steps == 0:
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)

            # Prevents exploding gradients that destabilize training
            torch.nn.utils.clip_grad_norm_(
                controlnet.parameters(), max_norm=1.0
            )

            # Step optimizer, scaler, and LR scheduler
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Print peak GPU memory on first step (Requirement 9.5)
        if step == 0 and torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
            logger.info(f"Peak GPU memory after first step: {peak_mem:.1f} MB")

        # ─────────────────────────────────────────────────────────────────
        # Log to W&B and save checkpoint every 250 steps
        # Generates 4 validation samples, logs metrics, saves to Drive.
        # ─────────────────────────────────────────────────────────────────
        if (step + 1) % train_config.checkpoint_every == 0:
            # Use the unscaled loss value for logging (multiply back by accum steps)
            log_loss = loss.item() * gradient_accumulation_steps
            log_and_checkpoint(
                step=step + 1,
                loss=log_loss,
                controlnet=controlnet,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                unet=unet,
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                train_config=train_config,
                device=device,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 9. Training complete — save final model and upload to HuggingFace Hub
    # ─────────────────────────────────────────────────────────────────────────
    finalize_training(controlnet=controlnet, train_config=train_config)

    logger.info(f"Training complete after {args.max_train_steps} steps.")


if __name__ == "__main__":
    train_pose()
