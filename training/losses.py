"""
Diffusion Training Loss
========================

In diffusion-based image generation, training works as follows:

1. Random Gaussian noise is added to the image latent at a random timestep t.
   The higher the timestep, the more noise is added, eventually destroying
   the original image signal entirely.

2. The model is trained to predict WHAT noise was added. This is the
   "epsilon-prediction" objective from DDPM (Ho et al., 2020).

3. The text embedding and condition image (depth map, pose skeleton, or
   Canny edge map) guide the model toward reconstructing the RIGHT image.
   They don't appear in the loss formula directly — instead, they enter
   through the model's forward pass, biasing the noise prediction so that
   the model learns to denoise toward the conditioned target image.

Loss = MSE(predicted_noise, actual_noise)

The condition image enters through the ControlNet adapter's forward pass,
which injects spatial features into the frozen SD1.5 UNet via zero
convolutions. This guides the UNet's noise prediction without changing
the loss formulation itself.
"""

import torch
import torch.nn.functional as F


def compute_diffusion_loss(
    model_pred: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    step: int,
) -> torch.Tensor:
    """
    Compute the diffusion training loss (MSE between predicted and actual noise).

    Args:
        model_pred: Predicted noise from the ControlNet-modified UNet.
                    Shape: (B, 4, H, W)
        noise: Actual Gaussian noise that was added to the latent.
               Shape: (B, 4, H, W) — must match model_pred shape exactly.
        timesteps: Diffusion timesteps used for noise addition. Shape: (B,)
        step: Current training step number, used for periodic logging.

    Returns:
        Scalar MSE loss value suitable for backpropagation.

    Raises:
        ValueError: If model_pred and noise have mismatched shapes.
    """
    if model_pred.shape != noise.shape:
        raise ValueError(
            f"Shape mismatch between predicted noise {model_pred.shape} "
            f"and actual noise {noise.shape}. Both tensors must have "
            f"identical shapes for MSE loss computation."
        )

    loss = F.mse_loss(model_pred, noise, reduction="mean")

    if step % 10 == 0:
        print(f"[Step {step}] Loss: {loss.item():.6f}")

    return loss
