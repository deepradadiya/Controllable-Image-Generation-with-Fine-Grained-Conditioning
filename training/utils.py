"""
Shared Training Utilities for ControlNet Adapter Training
==========================================================

This module provides shared infrastructure used by all three training scripts
(train_depth.py, train_pose.py, train_edge.py) to ensure consistent:

- Optimizer configuration (AdamW, lr=1e-5, cosine schedule with warmup)
- Checkpoint saving/loading (every 250 steps to Google Drive, max 3 kept)
- W&B logging (loss, learning rate, sample images every 250 steps)
- HuggingFace Hub upload (safetensors format with model card)

All three training scripts differ ONLY in condition_type and Hub repo name.
The training loop, optimizer, scheduler, logging, and checkpointing are
identical across depth, pose, and edge conditioning.
"""

import glob
import logging
import os
import random
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training Configuration Dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """
    Shared training configuration for all three ControlNet training scripts.

    All three scripts (depth, pose, edge) use identical hyperparameters.
    They differ only in condition_type and hub_model_id.
    """

    condition_type: str  # "depth", "pose", or "edge"

    # Optimizer: AdamW
    learning_rate: float = 1e-5
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-2
    eps: float = 1e-8

    # LR Schedule: Cosine with warmup
    lr_warmup_steps: int = 500

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Mixed precision
    mixed_precision: bool = True

    # Checkpointing
    checkpoint_every: int = 250
    log_every: int = 250
    max_train_steps: int = 10000
    gradient_accumulation_steps: int = 8

    # Hub and output
    hub_model_id: str = ""
    output_dir: str = ""
    drive_checkpoint_path: str = "/content/drive/MyDrive/controlnet_checkpoints"
    max_checkpoints: int = 3


# ---------------------------------------------------------------------------
# Optimizer Setup
# ---------------------------------------------------------------------------


def setup_optimizer(model_params) -> torch.optim.AdamW:
    """
    Create AdamW optimizer with the shared ControlNet training configuration.

    Only ControlNet adapter parameters should be passed here — the frozen
    SD1.5 UNet, VAE, and CLIP encoder parameters must NOT be included.

    Args:
        model_params: Iterable of parameters to optimize. Should be
                      controlnet.parameters() (only the trainable adapter).

    Returns:
        Configured AdamW optimizer with:
        - lr=1e-5 (conservative for fine-tuning pretrained weights)
        - betas=(0.9, 0.999) (standard Adam momentum)
        - weight_decay=1e-2 (regularization)
        - eps=1e-8 (numerical stability)
    """
    optimizer = torch.optim.AdamW(
        model_params,
        lr=1e-5,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )
    return optimizer


# ---------------------------------------------------------------------------
# LR Scheduler Setup
# ---------------------------------------------------------------------------


def setup_scheduler(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    warmup_steps: int = 500,
):
    """
    Create a cosine learning rate schedule with linear warmup.

    The schedule:
    1. Linearly increases LR from 0 to 1e-5 over the first 500 steps (warmup)
    2. Then decays following a cosine curve from 1e-5 toward 0

    This prevents early training instability (warmup) and allows the model
    to converge smoothly as training progresses (cosine decay).

    Args:
        optimizer: The optimizer to schedule.
        num_training_steps: Total number of training steps.
        warmup_steps: Number of warmup steps (default 500).

    Returns:
        LR scheduler with cosine decay and linear warmup.
    """
    try:
        from diffusers.optimization import get_cosine_schedule_with_warmup
    except ImportError:
        from transformers import get_cosine_schedule_with_warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=0.5,  # Standard cosine decay (half cycle)
    )
    return scheduler


# ---------------------------------------------------------------------------
# Checkpoint Saving
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    output_dir: str,
    max_checkpoints: int = 3,
) -> str:
    """
    Save a training checkpoint to disk (typically Google Drive).

    Saves model weights, optimizer state, scheduler state, step count,
    and random states so training can be resumed exactly where it left off.

    Only the 3 most recent checkpoints are kept to conserve Drive storage.

    Args:
        model: The ControlNet adapter model.
        optimizer: The AdamW optimizer.
        scheduler: The LR scheduler.
        step: Current training step number.
        output_dir: Directory to save checkpoints (e.g., Drive path).
        max_checkpoints: Maximum number of checkpoints to retain (default 3).

    Returns:
        Path to the saved checkpoint directory.
    """
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save all training state
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
    }

    # Include CUDA random state if available
    if torch.cuda.is_available():
        checkpoint["cuda_random_state"] = torch.cuda.get_rng_state()

    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
    torch.save(checkpoint, checkpoint_path)

    logger.info(f"Saved checkpoint at step {step} to {checkpoint_dir}")

    # Remove old checkpoints, keeping only the most recent max_checkpoints
    _cleanup_old_checkpoints(output_dir, max_checkpoints)

    return checkpoint_dir


def _cleanup_old_checkpoints(output_dir: str, max_checkpoints: int) -> None:
    """Remove old checkpoints, keeping only the most recent ones."""
    checkpoint_dirs = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda x: int(x.split("-")[-1]),
    )

    if len(checkpoint_dirs) > max_checkpoints:
        dirs_to_remove = checkpoint_dirs[: len(checkpoint_dirs) - max_checkpoints]
        for dir_path in dirs_to_remove:
            logger.info(f"Removing old checkpoint: {dir_path}")
            shutil.rmtree(dir_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Checkpoint Loading
# ---------------------------------------------------------------------------


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    checkpoint_path: str,
) -> int:
    """
    Restore training state from a checkpoint.

    Loads model weights, optimizer state, scheduler state, and random states
    so that training resumes exactly where it left off.

    Args:
        model: The ControlNet adapter model to restore weights into.
        optimizer: The optimizer to restore state into.
        scheduler: The LR scheduler to restore state into.
        checkpoint_path: Path to the checkpoint directory or .pt file.

    Returns:
        The training step to resume from.
    """
    # Support both directory and direct file paths
    if os.path.isdir(checkpoint_path):
        checkpoint_file = os.path.join(checkpoint_path, "checkpoint.pt")
    else:
        checkpoint_file = checkpoint_path

    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_file}. "
            f"Cannot resume training without a valid checkpoint."
        )

    # weights_only=False is needed because we save numpy/random states
    # These checkpoints are self-generated during training, so this is safe
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Restore random states for reproducibility
    if "random_state" in checkpoint:
        random.setstate(checkpoint["random_state"])
    if "numpy_random_state" in checkpoint:
        np.random.set_state(checkpoint["numpy_random_state"])
    if "torch_random_state" in checkpoint:
        torch.random.set_rng_state(checkpoint["torch_random_state"])
    if "cuda_random_state" in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state(checkpoint["cuda_random_state"])

    step = checkpoint["step"]
    logger.info(f"Resumed from checkpoint at step {step}")

    return step


# ---------------------------------------------------------------------------
# W&B Logging
# ---------------------------------------------------------------------------


def log_to_wandb(
    step: int,
    loss: float,
    lr: float,
    sample_images: Optional[List] = None,
) -> None:
    """
    Log training metrics and optional sample images to Weights & Biases.

    Called every 250 steps during training. Logs loss and learning rate
    on every call, and optionally logs sample images for visual inspection.

    Handles wandb not being initialized gracefully — if wandb.run is None
    or wandb is not imported, this function is a no-op.

    Args:
        step: Current training step.
        loss: Training loss value.
        lr: Current learning rate.
        sample_images: Optional list of PIL Images to log as samples.
    """
    try:
        import wandb

        if wandb.run is None:
            # W&B not initialized — skip logging silently
            return

        log_dict: Dict = {
            "loss": loss,
            "learning_rate": lr,
            "step": step,
        }

        if sample_images is not None and len(sample_images) > 0:
            log_dict["sample_images"] = [
                wandb.Image(img, caption=f"Step {step} - Sample {i}")
                for i, img in enumerate(sample_images)
            ]

        wandb.log(log_dict, step=step)

    except ImportError:
        # wandb not installed — skip logging
        pass
    except Exception as e:
        # Handle any W&B connection errors gracefully
        logger.warning(f"W&B logging failed at step {step}: {e}")


# ---------------------------------------------------------------------------
# HuggingFace Hub Upload
# ---------------------------------------------------------------------------


def upload_to_hub(
    model: nn.Module,
    hub_model_id: str,
    condition_type: str,
    training_config: TrainConfig,
) -> None:
    """
    Upload trained ControlNet adapter to HuggingFace Hub.

    Saves the model in safetensors format with a model card containing
    training metadata (base model, condition type, lr, steps, etc.).

    If authentication fails or upload errors occur, the model is saved
    locally and a warning is logged instead of crashing.

    Args:
        model: The trained ControlNet adapter model.
        hub_model_id: HuggingFace Hub repository ID (e.g., "user/controlnet-sd15-depth").
        condition_type: The condition type ("depth", "pose", or "edge").
        training_config: The TrainConfig used during training.
    """
    try:
        from huggingface_hub import HfApi, ModelCard, ModelCardData
        from safetensors.torch import save_file
    except ImportError as e:
        logger.warning(
            f"Required packages not available for Hub upload: {e}. "
            f"Saving model locally instead."
        )
        _save_model_locally(model, training_config.output_dir, condition_type)
        return

    # Save model in safetensors format to a temporary directory
    save_dir = os.path.join(
        training_config.output_dir, f"controlnet-{condition_type}-final"
    )
    os.makedirs(save_dir, exist_ok=True)

    # Save weights as safetensors
    state_dict = model.state_dict()
    safetensors_path = os.path.join(save_dir, "model.safetensors")
    save_file(state_dict, safetensors_path)

    # Create model card
    model_card_content = _create_model_card(
        hub_model_id=hub_model_id,
        condition_type=condition_type,
        config=training_config,
    )
    model_card_path = os.path.join(save_dir, "README.md")
    with open(model_card_path, "w") as f:
        f.write(model_card_content)

    # Attempt upload to Hub
    try:
        api = HfApi()
        api.create_repo(repo_id=hub_model_id, exist_ok=True)
        api.upload_folder(
            folder_path=save_dir,
            repo_id=hub_model_id,
            commit_message=f"Upload ControlNet {condition_type} adapter",
        )
        logger.info(f"Successfully uploaded model to {hub_model_id}")
    except Exception as e:
        logger.warning(
            f"HuggingFace Hub upload failed: {e}. "
            f"Model saved locally at {save_dir}"
        )


def _save_model_locally(
    model: nn.Module, output_dir: str, condition_type: str
) -> str:
    """Save model locally when Hub upload is not possible."""
    save_dir = os.path.join(output_dir, f"controlnet-{condition_type}-final")
    os.makedirs(save_dir, exist_ok=True)

    try:
        from safetensors.torch import save_file

        state_dict = model.state_dict()
        save_file(state_dict, os.path.join(save_dir, "model.safetensors"))
    except ImportError:
        # Fallback to PyTorch native format
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))

    logger.info(f"Model saved locally at {save_dir}")
    return save_dir


def _create_model_card(
    hub_model_id: str,
    condition_type: str,
    config: TrainConfig,
) -> str:
    """Create a model card with training metadata."""
    precision = "fp16" if config.mixed_precision else "fp32"

    model_card = f"""---
tags:
- controlnet
- stable-diffusion
- {condition_type}
- image-generation
base_model: runwayml/stable-diffusion-v1-5
license: openrail++
---

# ControlNet SD1.5 - {condition_type.capitalize()} Conditioning

This is a ControlNet adapter trained for {condition_type} conditioning on top of
Stable Diffusion 1.5.

## Model Details

- **Base Model:** runwayml/stable-diffusion-v1-5
- **Condition Type:** {condition_type}
- **Learning Rate:** {config.learning_rate}
- **Training Steps:** {config.max_train_steps}
- **Dataset:** COCO 2017 (10k subset)
- **Precision:** {precision}
- **Optimizer:** AdamW (betas={config.betas}, weight_decay={config.weight_decay})
- **LR Schedule:** Cosine with {config.lr_warmup_steps} warmup steps
- **Gradient Clipping:** max_norm={config.max_grad_norm}

## Usage

```python
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

controlnet = ControlNetModel.from_pretrained("{hub_model_id}")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
)
```

## Training

Trained on Google Colab T4 GPU with:
- Batch size 1 with {config.gradient_accumulation_steps}x gradient accumulation
- Mixed precision ({precision})
- Gradient checkpointing for memory efficiency
- Checkpoints saved every {config.checkpoint_every} steps
"""
    return model_card
