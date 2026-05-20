"""
Pipeline loading utilities for the evaluation pipeline.

Provides functions to load ControlNetPipeline instances with trained adapter
checkpoints, load vanilla SD1.5 for baseline generation, and validate which
condition types have available checkpoints.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from model.controlnet import ControlNet
from model.pipeline import ControlNetPipeline

logger = logging.getLogger(__name__)

# Default SD1.5 model identifier
SD15_MODEL_ID = "runwayml/stable-diffusion-v1-5"


def load_controlnet_pipeline(
    condition_type: str,
    checkpoint_dir: str = "models/trained",
    device: Optional[torch.device] = None,
) -> Optional[ControlNetPipeline]:
    """Load the ControlNetPipeline with a trained adapter checkpoint.

    Loads the SD1.5 base model components (UNet, VAE, text encoder, tokenizer,
    scheduler) and the trained ControlNet adapter for the specified condition type.

    Args:
        condition_type: One of "depth", "pose", or "edge".
        checkpoint_dir: Directory containing trained ControlNet checkpoints.
            Expected structure: {checkpoint_dir}/controlnet-sd15-{condition_type}/
        device: Device to load the pipeline on. Defaults to CUDA if available.

    Returns:
        A loaded ControlNetPipeline ready for inference, or None if the
        checkpoint is not found for the given condition type.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Construct expected checkpoint path
    checkpoint_path = Path(checkpoint_dir) / f"controlnet-sd15-{condition_type}"

    if not checkpoint_path.exists():
        logger.warning(
            f"Checkpoint not found for condition type '{condition_type}' "
            f"at path: {checkpoint_path}"
        )
        return None

    try:
        # Load SD1.5 base model components
        tokenizer = CLIPTokenizer.from_pretrained(
            SD15_MODEL_ID, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            SD15_MODEL_ID, subfolder="text_encoder"
        ).to(device)
        vae = AutoencoderKL.from_pretrained(
            SD15_MODEL_ID, subfolder="vae"
        ).to(device)
        unet = UNet2DConditionModel.from_pretrained(
            SD15_MODEL_ID, subfolder="unet"
        ).to(device)
        scheduler = DDIMScheduler.from_pretrained(
            SD15_MODEL_ID, subfolder="scheduler"
        )

        # Load trained ControlNet adapter
        controlnet = ControlNet(unet=unet, condition_channels=3)

        # Load the trained weights from checkpoint
        checkpoint_file = checkpoint_path / "controlnet_state_dict.pt"
        if checkpoint_file.exists():
            state_dict = torch.load(checkpoint_file, map_location=device)
            controlnet.load_state_dict(state_dict)
        else:
            # Try alternative checkpoint format (diffusers-style)
            config_file = checkpoint_path / "config.json"
            if config_file.exists():
                # Reinitialize from checkpoint directory using state dict
                state_dict_file = checkpoint_path / "pytorch_model.bin"
                safetensors_file = checkpoint_path / "model.safetensors"
                if state_dict_file.exists():
                    state_dict = torch.load(state_dict_file, map_location=device)
                    controlnet.load_state_dict(state_dict)
                elif safetensors_file.exists():
                    from safetensors.torch import load_file

                    state_dict = load_file(str(safetensors_file))
                    controlnet.load_state_dict(state_dict)
                else:
                    logger.warning(
                        f"No model weights found in checkpoint directory: "
                        f"{checkpoint_path}"
                    )
                    return None
            else:
                logger.warning(
                    f"No valid checkpoint format found at: {checkpoint_path}"
                )
                return None

        controlnet = controlnet.to(device)
        controlnet.eval()

        # Create and return the pipeline
        pipeline = ControlNetPipeline(
            controlnet=controlnet,
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )

        logger.info(
            f"Successfully loaded ControlNet pipeline for '{condition_type}' "
            f"from {checkpoint_path}"
        )
        return pipeline

    except Exception as e:
        logger.warning(
            f"Failed to load ControlNet pipeline for '{condition_type}': {e}"
        )
        return None


def load_baseline_pipeline(
    device: Optional[torch.device] = None,
) -> Optional[ControlNetPipeline]:
    """Load vanilla SD1.5 without ControlNet for baseline generation.

    Creates a ControlNetPipeline with a zero-initialized ControlNet adapter,
    effectively producing vanilla SD1.5 outputs since the ControlNet contribution
    starts at zero.

    Args:
        device: Device to load the pipeline on. Defaults to CUDA if available.

    Returns:
        A ControlNetPipeline configured for baseline (unconditioned) generation,
        or None if loading fails.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Load SD1.5 base model components
        tokenizer = CLIPTokenizer.from_pretrained(
            SD15_MODEL_ID, subfolder="tokenizer"
        )
        text_encoder = CLIPTextModel.from_pretrained(
            SD15_MODEL_ID, subfolder="text_encoder"
        ).to(device)
        vae = AutoencoderKL.from_pretrained(
            SD15_MODEL_ID, subfolder="vae"
        ).to(device)
        unet = UNet2DConditionModel.from_pretrained(
            SD15_MODEL_ID, subfolder="unet"
        ).to(device)
        scheduler = DDIMScheduler.from_pretrained(
            SD15_MODEL_ID, subfolder="scheduler"
        )

        # Create a zero-initialized ControlNet (no trained weights)
        # The zero convolutions ensure ControlNet output is zero at initialization,
        # so the pipeline behaves as vanilla SD1.5
        controlnet = ControlNet(unet=unet, condition_channels=3)
        controlnet = controlnet.to(device)
        controlnet.eval()

        # Create and return the pipeline
        pipeline = ControlNetPipeline(
            controlnet=controlnet,
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )

        logger.info("Successfully loaded baseline SD1.5 pipeline")
        return pipeline

    except Exception as e:
        logger.warning(f"Failed to load baseline SD1.5 pipeline: {e}")
        return None


def validate_checkpoints(
    checkpoint_dir: str = "models/trained",
    condition_types: Optional[List[str]] = None,
) -> List[str]:
    """Check which condition types have available checkpoints.

    Scans the checkpoint directory for valid ControlNet checkpoints and
    returns the subset of condition types that have checkpoints available.

    Args:
        checkpoint_dir: Directory containing trained ControlNet checkpoints.
        condition_types: List of condition types to check. Defaults to
            ["depth", "pose", "edge"].

    Returns:
        List of condition types that have valid checkpoints available.
        May be empty if no checkpoints are found.
    """
    if condition_types is None:
        condition_types = ["depth", "pose", "edge"]

    valid_types = []
    checkpoint_base = Path(checkpoint_dir)

    for condition_type in condition_types:
        checkpoint_path = checkpoint_base / f"controlnet-sd15-{condition_type}"

        if not checkpoint_path.exists():
            logger.warning(
                f"No checkpoint found for '{condition_type}' at: {checkpoint_path}"
            )
            continue

        # Check for valid model files
        has_state_dict = (checkpoint_path / "controlnet_state_dict.pt").exists()
        has_pytorch_model = (checkpoint_path / "pytorch_model.bin").exists()
        has_safetensors = (checkpoint_path / "model.safetensors").exists()

        if has_state_dict or has_pytorch_model or has_safetensors:
            valid_types.append(condition_type)
            logger.info(
                f"Valid checkpoint found for '{condition_type}' at: {checkpoint_path}"
            )
        else:
            logger.warning(
                f"Checkpoint directory exists for '{condition_type}' but contains "
                f"no valid model weights at: {checkpoint_path}"
            )

    if not valid_types:
        logger.warning(
            f"No valid checkpoints found in {checkpoint_dir} for any condition type"
        )

    return valid_types
