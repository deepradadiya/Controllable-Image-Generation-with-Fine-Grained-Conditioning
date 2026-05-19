#!/usr/bin/env python3
"""
Memory Budget Estimation Script for ControlNet Training on T4 GPU
=================================================================

This script estimates the VRAM usage for ControlNet training without
requiring pretrained weights or GPU hardware. It creates the model
architecture with random initialization and calculates parameter counts
and memory requirements.

Usage:
    python scripts/estimate_memory.py

Output:
    Formatted table showing each component's parameter count and
    estimated memory usage, with a total and T4 budget comparison.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def count_parameters(model) -> int:
    """Count total parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model) -> int:
    """Count only trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_frozen_parameters(model) -> int:
    """Count only frozen parameters in a model."""
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def params_to_gb(num_params: int, bytes_per_param: int = 2) -> float:
    """Convert parameter count to GB given bytes per parameter."""
    return (num_params * bytes_per_param) / (1024**3)


def format_params(num_params: int) -> str:
    """Format parameter count in human-readable form (e.g., 860.3M)."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.1f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.1f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.1f}K"
    return str(num_params)


def estimate_memory():
    """
    Estimate VRAM usage for ControlNet training on T4 GPU.

    Creates model architectures (without pretrained weights) and calculates:
    - Parameter counts for each component
    - FP16 model memory
    - FP32 optimizer state memory (Adam: 2 states per trainable param)
    - FP16 gradient memory (only for trainable params)
    - Estimated activation memory
    - CUDA overhead
    """
    import torch
    import torch.nn as nn

    print("=" * 70)
    print("  ControlNet Training Memory Budget Estimation")
    print("  Target: NVIDIA T4 GPU (15 GB VRAM)")
    print("=" * 70)
    print()

    # ─────────────────────────────────────────────────────────────────────
    # Estimate SD1.5 UNet parameters
    # The UNet2DConditionModel from diffusers has ~860M parameters
    # We use the known architecture specs rather than instantiating it
    # to avoid requiring the diffusers config files.
    # ─────────────────────────────────────────────────────────────────────
    unet_params = 860_000_000  # ~860M params (SD1.5 UNet)
    vae_params = 84_000_000   # ~84M params (SD1.5 VAE)
    clip_params = 123_000_000  # ~123M params (CLIP ViT-L/14 text encoder)

    # ─────────────────────────────────────────────────────────────────────
    # Estimate ControlNet adapter parameters
    # The adapter copies the UNet encoder (~half of UNet) plus adds:
    # - condition_embedding: small CNN (~2M params)
    # - zero_convs: 1x1 convolutions (~5M params)
    # Total trainable: ~360M params
    # ─────────────────────────────────────────────────────────────────────

    # Try to instantiate actual ControlNet for precise count
    controlnet_trainable_params = None
    controlnet_frozen_params = None

    try:
        from model.controlnet import ControlNet
        from diffusers import UNet2DConditionModel

        # Create a minimal UNet config for parameter counting
        # This uses random weights — we only need the architecture
        unet = UNet2DConditionModel(
            sample_size=64,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(320, 640, 1280, 1280),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
            ),
            cross_attention_dim=768,
        )

        controlnet = ControlNet(unet=unet, condition_channels=3)
        controlnet_trainable_params = count_trainable_parameters(controlnet)
        controlnet_frozen_params = count_frozen_parameters(controlnet)
        unet_params = count_parameters(unet)

        print(f"  [✓] Instantiated actual ControlNet model for precise counts")
        print()

    except Exception as e:
        print(f"  [!] Could not instantiate ControlNet: {e}")
        print(f"      Using estimated parameter counts instead.")
        print()
        controlnet_trainable_params = 360_000_000
        controlnet_frozen_params = 430_000_000  # Copied encoder blocks (frozen)

    # ─────────────────────────────────────────────────────────────────────
    # Calculate memory for each component
    # ─────────────────────────────────────────────────────────────────────

    # FP16 model weights (2 bytes per parameter)
    unet_mem_gb = params_to_gb(unet_params, bytes_per_param=2)
    vae_mem_gb = params_to_gb(vae_params, bytes_per_param=2)
    clip_mem_gb = params_to_gb(clip_params, bytes_per_param=2)
    controlnet_mem_gb = params_to_gb(
        controlnet_trainable_params + (controlnet_frozen_params or 0),
        bytes_per_param=2,
    )

    # Optimizer states: Adam keeps 2 FP32 states per trainable param
    # (first moment + second moment)
    optimizer_mem_gb = params_to_gb(controlnet_trainable_params, bytes_per_param=4) * 2

    # Gradients: FP16 for trainable params only
    gradient_mem_gb = params_to_gb(controlnet_trainable_params, bytes_per_param=2)

    # Activation memory estimate (empirical for batch_size=1, 512x512, FP16)
    # Based on typical transformer/UNet activation patterns
    activation_mem_gb = 3.0

    # CUDA overhead (context, allocator fragmentation, temp buffers)
    cuda_overhead_gb = 1.6

    # Total
    total_mem_gb = (
        unet_mem_gb
        + vae_mem_gb
        + clip_mem_gb
        + controlnet_mem_gb
        + optimizer_mem_gb
        + gradient_mem_gb
        + activation_mem_gb
        + cuda_overhead_gb
    )

    # ─────────────────────────────────────────────────────────────────────
    # Print formatted memory budget table
    # ─────────────────────────────────────────────────────────────────────

    print("─" * 70)
    print(f"  {'Component':<40} {'Params':<12} {'Memory (GB)':<12}")
    print("─" * 70)

    print(f"  {'SD1.5 UNet (FP16, frozen)':<40} {format_params(unet_params):<12} {unet_mem_gb:.2f}")
    print(f"  {'SD1.5 VAE (FP16, frozen)':<40} {format_params(vae_params):<12} {vae_mem_gb:.2f}")
    print(f"  {'CLIP Text Encoder (FP16, frozen)':<40} {format_params(clip_params):<12} {clip_mem_gb:.2f}")

    total_cn_params = controlnet_trainable_params + (controlnet_frozen_params or 0)
    print(f"  {'ControlNet Adapter (FP16)':<40} {format_params(total_cn_params):<12} {controlnet_mem_gb:.2f}")

    print("─" * 70)
    print(f"  {'Optimizer States (Adam, FP32)':<40} {'—':<12} {optimizer_mem_gb:.2f}")
    print(f"  {'Gradients (FP16, trainable only)':<40} {'—':<12} {gradient_mem_gb:.2f}")
    print(f"  {'Activations (FP16, batch_size=1)':<40} {'—':<12} {activation_mem_gb:.2f}")
    print(f"  {'CUDA Overhead / Buffer':<40} {'—':<12} {cuda_overhead_gb:.2f}")
    print("─" * 70)
    print(f"  {'TOTAL ESTIMATED PEAK VRAM':<40} {'':12} {total_mem_gb:.2f}")
    print(f"  {'T4 GPU VRAM BUDGET':<40} {'':12} {'15.00'}")
    print(f"  {'HEADROOM':<40} {'':12} {15.0 - total_mem_gb:.2f}")
    print("─" * 70)
    print()

    # ─────────────────────────────────────────────────────────────────────
    # Summary and verdict
    # ─────────────────────────────────────────────────────────────────────

    if total_mem_gb <= 15.0:
        print(f"  ✅ FITS on T4 GPU ({total_mem_gb:.1f} GB < 15.0 GB)")
        print(f"     Headroom: {15.0 - total_mem_gb:.1f} GB available for spikes")
    else:
        print(f"  ❌ EXCEEDS T4 GPU budget ({total_mem_gb:.1f} GB > 15.0 GB)")
        print(f"     Over budget by: {total_mem_gb - 15.0:.1f} GB")
        print(f"     Consider: smaller batch size, more aggressive checkpointing")

    print()
    print("  Memory Optimization Techniques Applied:")
    print("  ─────────────────────────────────────────")
    print("  • FP16 mixed precision (halves model + activation memory)")
    print("  • Frozen UNet/VAE/CLIP (no gradient/optimizer storage for ~1B params)")
    print("  • Batch size 1 + gradient accumulation (minimal per-step memory)")
    print("  • Gradient checkpointing available (trades compute for memory)")
    print()

    # ─────────────────────────────────────────────────────────────────────
    # Parameter breakdown
    # ─────────────────────────────────────────────────────────────────────

    print("  Parameter Breakdown:")
    print("  ─────────────────────────────────────────")
    print(f"  Trainable (ControlNet adapter):  {format_params(controlnet_trainable_params)}")
    if controlnet_frozen_params:
        print(f"  Frozen (ControlNet encoder copy): {format_params(controlnet_frozen_params)}")
    print(f"  Frozen (UNet):                   {format_params(unet_params)}")
    print(f"  Frozen (VAE):                    {format_params(vae_params)}")
    print(f"  Frozen (CLIP):                   {format_params(clip_params)}")
    total_frozen = unet_params + vae_params + clip_params + (controlnet_frozen_params or 0)
    print(f"  ─────────────────────────────────────────")
    print(f"  Total trainable:                 {format_params(controlnet_trainable_params)}")
    print(f"  Total frozen:                    {format_params(total_frozen)}")
    print(f"  Trainable ratio:                 {controlnet_trainable_params / (controlnet_trainable_params + total_frozen) * 100:.1f}%")
    print()

    return total_mem_gb


if __name__ == "__main__":
    estimated_gb = estimate_memory()
    sys.exit(0 if estimated_gb <= 15.0 else 1)
