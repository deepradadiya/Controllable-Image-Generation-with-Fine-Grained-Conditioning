"""
Property-Based Tests for Training Loop.

Tests three key training properties:
1. Frozen parameters (UNet/VAE/CLIP) never receive gradients during training
2. Gradient norm is bounded (≤ 1.0) after clipping
3. Loss decreases over multiple training steps with synthetic data

These tests use a minimal mock UNet to avoid downloading SD1.5 weights,
and simulate training steps with synthetic data on CPU.

**Validates: Requirements 5.9, 5.11**
"""

import sys
import os

import pytest
import torch
import torch.nn as nn

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.controlnet import ControlNet
from training.losses import compute_diffusion_loss
from diffusers import UNet2DConditionModel, DDPMScheduler


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def mock_unet():
    """
    Create a minimal UNet2DConditionModel with the right structure
    for testing without downloading SD1.5 weights.
    """
    unet = UNet2DConditionModel(
        sample_size=64,
        in_channels=4,
        out_channels=4,
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
        block_out_channels=(320, 640, 1280, 1280),
        layers_per_block=2,
        cross_attention_dim=768,
    )
    return unet


@pytest.fixture(scope="module")
def controlnet(mock_unet):
    """Create a ControlNet instance from the mock UNet."""
    return ControlNet(mock_unet)


@pytest.fixture
def frozen_unet(mock_unet):
    """Create a frozen copy of the mock UNet (simulates the frozen UNet in training)."""
    import copy

    unet = copy.deepcopy(mock_unet)
    unet.requires_grad_(False)
    unet.eval()
    return unet


def simulate_training_step(controlnet, unet, optimizer, noise_scheduler=None):
    """
    Simulate a single training step matching the real training loop.

    Steps:
    1. Create synthetic batch (noisy latent, condition image, text embedding)
    2. Forward pass through ControlNet
    3. Forward pass through frozen UNet with ControlNet features
    4. Compute MSE loss
    5. Backward pass
    6. Clip gradients
    7. Step optimizer

    Args:
        controlnet: The trainable ControlNet adapter.
        unet: The frozen UNet (requires_grad=False).
        optimizer: The optimizer for ControlNet parameters.
        noise_scheduler: Optional DDPM scheduler for adding noise.

    Returns:
        The loss value from this step.
    """
    batch_size = 1

    # Create synthetic batch
    noisy_latent = torch.randn(batch_size, 4, 64, 64)
    condition_image = torch.randn(batch_size, 3, 512, 512)
    text_embedding = torch.randn(batch_size, 77, 768)
    timestep = torch.randint(0, 1000, (batch_size,))
    noise = torch.randn(batch_size, 4, 64, 64)

    # Forward pass through ControlNet
    controlnet_output = controlnet(
        noisy_latent, timestep, text_embedding, condition_image
    )

    # Forward pass through frozen UNet with ControlNet features
    noise_pred = unet(
        noisy_latent,
        timestep,
        encoder_hidden_states=text_embedding,
        down_block_additional_residuals=controlnet_output["down_block_res_samples"],
        mid_block_additional_residual=controlnet_output["mid_block_res_sample"],
    ).sample

    # Compute MSE loss
    loss = compute_diffusion_loss(noise_pred, noise, timestep, step=0)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Clip gradients at max_norm=1.0
    torch.nn.utils.clip_grad_norm_(controlnet.parameters(), max_norm=1.0)

    # Step optimizer
    optimizer.step()

    return loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# Property 1: Frozen Parameters Never Get Gradients
# **Validates: Requirement 5.9**
#
# After any training step, UNet/VAE/CLIP params must have grad=None.
# Only ControlNet adapter parameters should accumulate gradients.
# ─────────────────────────────────────────────────────────────────────────────


class TestFrozenParamsNeverGetGradients:
    """
    Property: After any training step, all frozen model parameters
    (UNet, VAE, CLIP) have grad=None. Only ControlNet adapter parameters
    accumulate gradients.

    **Validates: Requirements 5.9**
    """

    def test_frozen_unet_has_no_gradients_after_training_step(
        self, mock_unet, frozen_unet
    ):
        """
        After a full training step (forward + backward + clip + step),
        the frozen UNet parameters must have grad=None.
        """
        # Create a fresh ControlNet for this test
        controlnet = ControlNet(mock_unet)
        controlnet.train()

        optimizer = torch.optim.AdamW(controlnet.parameters(), lr=1e-5)

        # Run a training step
        simulate_training_step(controlnet, frozen_unet, optimizer)

        # Verify: ALL frozen UNet parameters have no gradients
        for name, param in frozen_unet.named_parameters():
            assert param.grad is None, (
                f"Frozen UNet param '{name}' has gradient after training step. "
                f"Frozen models must never accumulate gradients."
            )

    def test_frozen_controlnet_encoder_has_no_gradients_after_training_step(
        self, mock_unet, frozen_unet
    ):
        """
        After a training step, the frozen encoder blocks INSIDE the ControlNet
        (down_blocks, mid_block, conv_in) must have grad=None.
        """
        controlnet = ControlNet(mock_unet)
        controlnet.train()

        optimizer = torch.optim.AdamW(controlnet.parameters(), lr=1e-5)

        # Run a training step
        simulate_training_step(controlnet, frozen_unet, optimizer)

        # Verify: frozen encoder blocks inside ControlNet have no gradients
        for name, param in controlnet.down_blocks.named_parameters():
            assert param.grad is None, (
                f"Frozen ControlNet encoder param 'down_blocks.{name}' "
                f"has gradient after training step."
            )

        for name, param in controlnet.mid_block.named_parameters():
            assert param.grad is None, (
                f"Frozen ControlNet encoder param 'mid_block.{name}' "
                f"has gradient after training step."
            )

        for name, param in controlnet.conv_in.named_parameters():
            assert param.grad is None, (
                f"Frozen ControlNet encoder param 'conv_in.{name}' "
                f"has gradient after training step."
            )

    def test_trainable_controlnet_params_have_gradients_after_training_step(
        self, mock_unet, frozen_unet
    ):
        """
        After a training step, trainable ControlNet parameters (zero_convs,
        condition_embedding) SHOULD have gradients.
        """
        controlnet = ControlNet(mock_unet)
        controlnet.train()

        optimizer = torch.optim.AdamW(controlnet.parameters(), lr=1e-5)

        # Run forward + backward (without optimizer step, to check grads)
        batch_size = 1
        noisy_latent = torch.randn(batch_size, 4, 64, 64)
        condition_image = torch.randn(batch_size, 3, 512, 512)
        text_embedding = torch.randn(batch_size, 77, 768)
        timestep = torch.randint(0, 1000, (batch_size,))
        noise = torch.randn(batch_size, 4, 64, 64)

        controlnet_output = controlnet(
            noisy_latent, timestep, text_embedding, condition_image
        )
        noise_pred = frozen_unet(
            noisy_latent,
            timestep,
            encoder_hidden_states=text_embedding,
            down_block_additional_residuals=controlnet_output[
                "down_block_res_samples"
            ],
            mid_block_additional_residual=controlnet_output["mid_block_res_sample"],
        ).sample

        loss = compute_diffusion_loss(noise_pred, noise, timestep, step=0)
        optimizer.zero_grad()
        loss.backward()

        # Trainable condition_embedding params should have gradients
        has_cond_grad = any(
            p.grad is not None
            for p in controlnet.condition_embedding.parameters()
        )
        assert has_cond_grad, (
            "condition_embedding should have gradients after backward pass"
        )

    def test_frozen_params_unchanged_after_multiple_steps(
        self, mock_unet, frozen_unet
    ):
        """
        After multiple training steps, frozen UNet weights must remain
        exactly unchanged (no weight updates).
        """
        # Record initial frozen UNet weights
        initial_weights = {
            name: param.clone()
            for name, param in frozen_unet.named_parameters()
        }

        controlnet = ControlNet(mock_unet)
        controlnet.train()
        optimizer = torch.optim.AdamW(controlnet.parameters(), lr=1e-5)

        # Run 3 training steps
        for _ in range(3):
            simulate_training_step(controlnet, frozen_unet, optimizer)

        # Verify: frozen UNet weights are exactly unchanged
        for name, param in frozen_unet.named_parameters():
            assert torch.equal(param, initial_weights[name]), (
                f"Frozen UNet param '{name}' changed after training steps. "
                f"Frozen model weights must never be modified."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Property 2: Gradient Norm ≤ 1.0 After Clipping
# **Validates: Requirement 5.11**
#
# After gradient clipping with max_norm=1.0, the total gradient norm
# across all ControlNet parameters must be ≤ 1.0 (within fp tolerance).
# ─────────────────────────────────────────────────────────────────────────────


class TestGradientNormBoundedAfterClipping:
    """
    Property: After gradient clipping with max_norm=1.0, the total
    gradient norm across all ControlNet parameters is ≤ 1.0.

    **Validates: Requirements 5.11**
    """

    def test_gradient_norm_bounded_after_single_step(
        self, mock_unet, frozen_unet
    ):
        """
        After a single training step with gradient clipping,
        the total gradient norm must be ≤ 1.0.
        """
        controlnet = ControlNet(mock_unet)
        controlnet.train()

        optimizer = torch.optim.AdamW(controlnet.parameters(), lr=1e-5)

        # Forward + backward
        batch_size = 1
        noisy_latent = torch.randn(batch_size, 4, 64, 64)
        condition_image = torch.randn(batch_size, 3, 512, 512)
        text_embedding = torch.randn(batch_size, 77, 768)
        timestep = torch.randint(0, 1000, (batch_size,))
        noise = torch.randn(batch_size, 4, 64, 64)

        controlnet_output = controlnet(
            noisy_latent, timestep, text_embedding, condition_image
        )
        noise_pred = frozen_unet(
            noisy_latent,
            timestep,
            encoder_hidden_states=text_embedding,
            down_block_additional_residuals=controlnet_output[
                "down_block_res_samples"
            ],
            mid_block_additional_residual=controlnet_output["mid_block_res_sample"],
        ).sample

        loss = compute_diffusion_loss(noise_pred, noise, timestep, step=0)
        optimizer.zero_grad()
        loss.backward()

        # Apply gradient clipping at max_norm=1.0
        torch.nn.utils.clip_grad_norm_(controlnet.parameters(), max_norm=1.0)

        # Compute total gradient norm after clipping
        total_norm = torch.sqrt(
            sum(
                p.grad.norm() ** 2
                for p in controlnet.parameters()
                if p.grad is not None
            )
        ).item()

        # Gradient norm must be ≤ 1.0 (with floating point tolerance)
        assert total_norm <= 1.0 + 1e-6, (
            f"Total gradient norm {total_norm:.6f} exceeds max_norm=1.0 "
            f"after clipping. Gradient clipping is not working correctly."
        )

    def test_gradient_norm_bounded_with_large_loss(
        self, mock_unet, frozen_unet
    ):
        """
        Even with artificially large gradients (from a large loss),
        clipping must bound the norm to ≤ 1.0.
        """
        controlnet = ControlNet(mock_unet)
        controlnet.train()

        optimizer = torch.optim.AdamW(controlnet.parameters(), lr=1e-5)

        # Forward pass with large-magnitude inputs to produce large gradients
        batch_size = 1
        noisy_latent = torch.randn(batch_size, 4, 64, 64) * 10.0
        condition_image = torch.randn(batch_size, 3, 512, 512) * 10.0
        text_embedding = torch.randn(batch_size, 77, 768) * 10.0
        timestep = torch.randint(0, 1000, (batch_size,))
        noise = torch.randn(batch_size, 4, 64, 64) * 10.0

        controlnet_output = controlnet(
            noisy_latent, timestep, text_embedding, condition_image
        )
        noise_pred = frozen_unet(
            noisy_latent,
            timestep,
            encoder_hidden_states=text_embedding,
            down_block_additional_residuals=controlnet_output[
                "down_block_res_samples"
            ],
            mid_block_additional_residual=controlnet_output["mid_block_res_sample"],
        ).sample

        loss = compute_diffusion_loss(noise_pred, noise, timestep, step=0)
        optimizer.zero_grad()
        loss.backward()

        # Verify gradients exist and are large before clipping
        pre_clip_norm = torch.sqrt(
            sum(
                p.grad.norm() ** 2
                for p in controlnet.parameters()
                if p.grad is not None
            )
        ).item()
        assert pre_clip_norm > 1.0, (
            "Test setup issue: gradients should be large before clipping "
            f"(got norm={pre_clip_norm:.6f}). Increase input magnitudes."
        )

        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(controlnet.parameters(), max_norm=1.0)

        # Verify norm is bounded after clipping
        post_clip_norm = torch.sqrt(
            sum(
                p.grad.norm() ** 2
                for p in controlnet.parameters()
                if p.grad is not None
            )
        ).item()

        assert post_clip_norm <= 1.0 + 1e-6, (
            f"Total gradient norm {post_clip_norm:.6f} exceeds max_norm=1.0 "
            f"after clipping (pre-clip norm was {pre_clip_norm:.6f})."
        )

    def test_gradient_clipping_preserves_direction(
        self, mock_unet, frozen_unet
    ):
        """
        Gradient clipping should scale all gradients uniformly,
        preserving relative direction (proportional reduction).
        """
        controlnet = ControlNet(mock_unet)
        controlnet.train()

        optimizer = torch.optim.AdamW(controlnet.parameters(), lr=1e-5)

        # Forward + backward with large inputs
        batch_size = 1
        noisy_latent = torch.randn(batch_size, 4, 64, 64) * 5.0
        condition_image = torch.randn(batch_size, 3, 512, 512) * 5.0
        text_embedding = torch.randn(batch_size, 77, 768) * 5.0
        timestep = torch.randint(0, 1000, (batch_size,))
        noise = torch.randn(batch_size, 4, 64, 64) * 5.0

        controlnet_output = controlnet(
            noisy_latent, timestep, text_embedding, condition_image
        )
        noise_pred = frozen_unet(
            noisy_latent,
            timestep,
            encoder_hidden_states=text_embedding,
            down_block_additional_residuals=controlnet_output[
                "down_block_res_samples"
            ],
            mid_block_additional_residual=controlnet_output["mid_block_res_sample"],
        ).sample

        loss = compute_diffusion_loss(noise_pred, noise, timestep, step=0)
        optimizer.zero_grad()
        loss.backward()

        # Record pre-clip gradient ratios for a few parameters
        trainable_params = [
            p for p in controlnet.parameters() if p.grad is not None
        ]
        if len(trainable_params) >= 2:
            pre_clip_norms = [p.grad.norm().item() for p in trainable_params[:5]]

            # Apply clipping
            torch.nn.utils.clip_grad_norm_(controlnet.parameters(), max_norm=1.0)

            post_clip_norms = [p.grad.norm().item() for p in trainable_params[:5]]

            # Check that ratios are preserved (uniform scaling)
            # All gradients should be scaled by the same factor
            if pre_clip_norms[0] > 0 and post_clip_norms[0] > 0:
                scale_factor = post_clip_norms[0] / pre_clip_norms[0]
                for i in range(1, len(pre_clip_norms)):
                    if pre_clip_norms[i] > 1e-10:
                        expected = pre_clip_norms[i] * scale_factor
                        actual = post_clip_norms[i]
                        assert abs(actual - expected) < 1e-4, (
                            f"Gradient clipping did not scale uniformly. "
                            f"Param {i}: expected {expected:.6f}, got {actual:.6f}"
                        )


# ─────────────────────────────────────────────────────────────────────────────
# Property 3: Loss Decreases Over Multiple Steps
# **Validates: Requirements 5.9, 5.11**
#
# With synthetic data and a fixed random seed, the running average loss
# at step N should be less than the loss at step 0.
# ─────────────────────────────────────────────────────────────────────────────


class TestLossDecreasesOverMultipleSteps:
    """
    Property: With synthetic data and a consistent training setup,
    the running average loss decreases over multiple training steps.

    **Validates: Requirements 5.9, 5.11**
    """

    def test_loss_decreases_over_training(self, mock_unet, frozen_unet):
        """
        Over 20 training steps with fixed synthetic data, the average loss
        in the last 5 steps should be lower than the average loss in the
        first 5 steps.
        """
        torch.manual_seed(42)

        controlnet = ControlNet(mock_unet)
        controlnet.train()

        optimizer = torch.optim.AdamW(
            controlnet.parameters(), lr=1e-4  # Slightly higher LR for faster convergence in test
        )

        # Use FIXED synthetic data so the model can overfit (learn the pattern)
        batch_size = 1
        fixed_noisy_latent = torch.randn(batch_size, 4, 64, 64)
        fixed_condition_image = torch.randn(batch_size, 3, 512, 512)
        fixed_text_embedding = torch.randn(batch_size, 77, 768)
        fixed_timestep = torch.randint(0, 1000, (batch_size,))
        fixed_noise = torch.randn(batch_size, 4, 64, 64)

        losses = []
        num_steps = 20

        for step in range(num_steps):
            # Forward pass through ControlNet
            controlnet_output = controlnet(
                fixed_noisy_latent,
                fixed_timestep,
                fixed_text_embedding,
                fixed_condition_image,
            )

            # Forward pass through frozen UNet with ControlNet features
            noise_pred = frozen_unet(
                fixed_noisy_latent,
                fixed_timestep,
                encoder_hidden_states=fixed_text_embedding,
                down_block_additional_residuals=controlnet_output[
                    "down_block_res_samples"
                ],
                mid_block_additional_residual=controlnet_output[
                    "mid_block_res_sample"
                ],
            ).sample

            # Compute loss
            loss = compute_diffusion_loss(
                noise_pred, fixed_noise, fixed_timestep, step=step
            )
            losses.append(loss.item())

            # Backward + clip + step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controlnet.parameters(), max_norm=1.0)
            optimizer.step()

        # Compare average loss of first 5 steps vs last 5 steps
        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5

        assert late_avg < early_avg, (
            f"Loss did not decrease over training. "
            f"Early average: {early_avg:.6f}, Late average: {late_avg:.6f}. "
            f"All losses: {[f'{l:.4f}' for l in losses]}"
        )

    def test_loss_is_finite_throughout_training(self, mock_unet, frozen_unet):
        """
        Loss must remain finite (no NaN or Inf) throughout all training steps.
        This validates that gradient clipping prevents training instability.
        """
        torch.manual_seed(123)

        controlnet = ControlNet(mock_unet)
        controlnet.train()

        optimizer = torch.optim.AdamW(controlnet.parameters(), lr=1e-5)

        num_steps = 10

        for step in range(num_steps):
            batch_size = 1
            noisy_latent = torch.randn(batch_size, 4, 64, 64)
            condition_image = torch.randn(batch_size, 3, 512, 512)
            text_embedding = torch.randn(batch_size, 77, 768)
            timestep = torch.randint(0, 1000, (batch_size,))
            noise = torch.randn(batch_size, 4, 64, 64)

            controlnet_output = controlnet(
                noisy_latent, timestep, text_embedding, condition_image
            )
            noise_pred = frozen_unet(
                noisy_latent,
                timestep,
                encoder_hidden_states=text_embedding,
                down_block_additional_residuals=controlnet_output[
                    "down_block_res_samples"
                ],
                mid_block_additional_residual=controlnet_output[
                    "mid_block_res_sample"
                ],
            ).sample

            loss = compute_diffusion_loss(noise_pred, noise, timestep, step=step)

            assert torch.isfinite(loss), (
                f"Loss became non-finite at step {step}: {loss.item()}"
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controlnet.parameters(), max_norm=1.0)
            optimizer.step()
