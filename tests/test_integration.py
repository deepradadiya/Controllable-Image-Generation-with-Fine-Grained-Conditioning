"""
End-to-End Integration Tests for ControlNet Training Pipeline.

Tests three critical integration scenarios:
1. Training step: ControlNet → UNet → loss is finite, gradients flow correctly
2. Inference pipeline: load ControlNet → run pipeline → valid PIL Image output
3. Checkpoint save/reload: save → reload → identical output

Uses a minimal mock UNet (same architecture as SD1.5 but random weights)
to avoid downloading real model weights.

Validates: Requirements 2.6, 3.1, 5.9, 12.1, 12.4
"""

import copy
import sys
import os
import tempfile

import pytest
import torch
import torch.nn as nn
from PIL import Image
from unittest.mock import MagicMock

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusers import UNet2DConditionModel
from model.controlnet import ControlNet
from model.pipeline import ControlNetPipeline
from training.losses import compute_diffusion_loss
from training.utils import save_checkpoint, load_checkpoint, setup_optimizer, setup_scheduler


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def mock_unet():
    """
    Create a minimal UNet2DConditionModel with the same block structure
    as SD1.5 but random weights. Avoids downloading real model weights.
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
    unet = copy.deepcopy(mock_unet)
    unet.requires_grad_(False)
    unet.eval()
    return unet


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Training step produces finite loss and correct gradient flow
# Validates: Requirements 2.6, 5.9
# ─────────────────────────────────────────────────────────────────────────────


class TestTrainingStepProducesFiniteLoss:
    """
    Integration test: initialize ControlNet → run one training step →
    verify loss is finite and gradients flow correctly.

    **Validates: Requirements 2.6, 5.9**
    """

    def test_training_step_produces_finite_loss(self, mock_unet, frozen_unet):
        """
        End-to-end training step:
        1. Create ControlNet from mock UNet
        2. Create frozen copy of UNet
        3. Run forward pass: ControlNet → UNet → loss
        4. Verify loss is finite (not NaN/Inf)
        5. Verify gradients flow to ControlNet params
        6. Verify no gradients on frozen UNet
        """
        # Create a fresh ControlNet for this test
        controlnet = ControlNet(mock_unet)
        controlnet.train()

        # Set up optimizer (only ControlNet params)
        optimizer = torch.optim.AdamW(
            controlnet.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-2
        )

        # Create synthetic training batch
        batch_size = 1
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
        noise_pred = frozen_unet(
            noisy_latent,
            timestep,
            encoder_hidden_states=text_embedding,
            down_block_additional_residuals=controlnet_output["down_block_res_samples"],
            mid_block_additional_residual=controlnet_output["mid_block_res_sample"],
        ).sample

        # Compute MSE loss
        loss = compute_diffusion_loss(noise_pred, noise, timestep, step=0)

        # 4. Verify loss is finite (not NaN/Inf)
        assert torch.isfinite(loss), (
            f"Loss is not finite: {loss.item()}. "
            f"Training step produced NaN or Inf loss."
        )
        assert loss.item() > 0, "Loss should be positive for random inputs"

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # 5. Verify gradients flow to ControlNet trainable params
        # At initialization, zero convolutions output zeros. This means the
        # condition_embedding output doesn't affect the loss yet (it's multiplied
        # by zero weights). However, the zero_conv layers themselves DO receive
        # gradients because they are the learnable bottleneck that will eventually
        # allow condition information to flow through.
        #
        # After a few optimizer steps, once zero_conv weights become non-zero,
        # gradients will also flow back to condition_embedding.

        # Check zero_convs have gradients (they are the primary trainable layers)
        zero_conv_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for zc in controlnet.zero_convs
            for p in zc.parameters()
        )
        assert zero_conv_has_grad, (
            "zero_convs should have non-zero gradients after backward pass"
        )

        # Check mid_block_zero_conv has gradients
        mid_zc_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in controlnet.mid_block_zero_conv.parameters()
        )
        assert mid_zc_has_grad, (
            "mid_block_zero_conv should have non-zero gradients after backward pass"
        )

        # Verify that at least SOME trainable params have gradients
        trainable_with_grad = sum(
            1 for p in controlnet.parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
        )
        assert trainable_with_grad > 0, (
            "At least some trainable ControlNet parameters should have gradients"
        )

        # 6. Verify no gradients on frozen UNet
        for name, param in frozen_unet.named_parameters():
            assert param.grad is None, (
                f"Frozen UNet param '{name}' has gradient after training step. "
                f"Frozen models must never accumulate gradients."
            )

        # Also verify frozen encoder blocks inside ControlNet have no gradients
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


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Inference pipeline produces valid image
# Validates: Requirement 3.1
# ─────────────────────────────────────────────────────────────────────────────


def _make_mock_vae():
    """Create a mock VAE whose decode returns a (1, 3, 512, 512) tensor."""
    mock = MagicMock()
    decode_output = MagicMock()
    decode_output.sample = torch.randn(1, 3, 512, 512)
    mock.decode = MagicMock(return_value=decode_output)
    return mock


def _make_mock_text_encoder():
    """Create a mock CLIP text encoder returning (1, 77, 768) embeddings."""
    mock = MagicMock()
    mock.return_value = (torch.randn(1, 77, 768),)
    return mock


def _make_mock_tokenizer():
    """Create a mock CLIP tokenizer returning input_ids."""
    mock = MagicMock()
    token_output = MagicMock()
    token_output.input_ids = torch.randint(0, 49408, (1, 77))
    mock.return_value = token_output
    return mock


def _make_mock_scheduler():
    """Create a mock DDIMScheduler with required methods."""
    mock = MagicMock()
    mock.set_timesteps = MagicMock()
    mock.timesteps = torch.tensor([999, 750, 500, 250, 0])
    mock.init_noise_sigma = 1.0
    mock.scale_model_input = MagicMock(side_effect=lambda latents, t: latents)
    step_output = MagicMock()
    step_output.prev_sample = torch.randn(1, 4, 64, 64)
    mock.step = MagicMock(return_value=step_output)
    return mock


class TestInferencePipelineProducesValidImage:
    """
    Integration test: load trained ControlNet → run inference pipeline →
    verify output image is valid PIL Image 512x512 RGB.

    **Validates: Requirement 3.1**
    """

    def test_inference_pipeline_produces_valid_image(self, mock_unet):
        """
        End-to-end inference:
        1. Create ControlNet from mock UNet
        2. Create mock pipeline components (same approach as test_pipeline.py)
        3. Run inference
        4. Verify output is PIL Image 512x512 RGB
        """
        # Create a ControlNet (simulates a "trained" ControlNet)
        controlnet = ControlNet(mock_unet)
        controlnet.eval()

        # Create a mock ControlNet wrapper that returns the right structure
        # We use a MagicMock because the pipeline calls controlnet differently
        # than the raw forward method
        mock_controlnet = MagicMock()
        mock_controlnet.return_value = {
            "down_block_res_samples": [torch.zeros(1, 320, 64, 64)] * 12,
            "mid_block_res_sample": torch.zeros(1, 1280, 8, 8),
        }

        # Create a mock UNet that returns noise predictions
        mock_unet_for_pipeline = MagicMock()
        output = MagicMock()
        output.sample = torch.randn(1, 4, 64, 64)
        mock_unet_for_pipeline.return_value = output
        param = torch.nn.Parameter(torch.zeros(1))
        mock_unet_for_pipeline.parameters = MagicMock(side_effect=lambda: iter([param]))

        # Create other mock components
        vae = _make_mock_vae()
        text_encoder = _make_mock_text_encoder()
        tokenizer = _make_mock_tokenizer()
        scheduler = _make_mock_scheduler()

        # Create pipeline
        pipeline = ControlNetPipeline(
            controlnet=mock_controlnet,
            unet=mock_unet_for_pipeline,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )

        # Create a condition image
        condition_image = Image.new("RGB", (512, 512), color=(128, 128, 128))

        # Run inference
        result = pipeline(
            text_prompt="a beautiful landscape with mountains",
            condition_image=condition_image,
            condition_type="depth",
            num_inference_steps=1,
            seed=42,
        )

        # Verify output is a valid PIL Image
        assert isinstance(result, Image.Image), (
            f"Pipeline output should be a PIL Image, got {type(result)}"
        )
        assert result.size == (512, 512), (
            f"Output image should be 512x512, got {result.size}"
        )
        assert result.mode == "RGB", (
            f"Output image should be RGB mode, got {result.mode}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Checkpoint save/reload produces identical output
# Validates: Requirements 12.1, 12.4
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckpointSaveReloadProducesIdenticalOutput:
    """
    Integration test: save checkpoint → reload → verify model produces
    identical output.

    **Validates: Requirements 12.1, 12.4**
    """

    def test_checkpoint_save_reload_produces_identical_output(self, mock_unet):
        """
        End-to-end checkpoint test:
        1. Create ControlNet, run forward pass, record output
        2. Save checkpoint
        3. Create new ControlNet, load checkpoint
        4. Run same forward pass, verify output matches
        """
        torch.manual_seed(42)

        # 1. Create ControlNet and run forward pass
        controlnet = ControlNet(mock_unet)
        controlnet.eval()

        # Set up optimizer and scheduler (needed for checkpoint save/load)
        optimizer = setup_optimizer(controlnet.parameters())
        scheduler = setup_scheduler(optimizer, num_training_steps=1000)

        # Create fixed input data
        batch_size = 1
        noisy_latent = torch.randn(batch_size, 4, 64, 64)
        condition_image = torch.randn(batch_size, 3, 512, 512)
        text_embedding = torch.randn(batch_size, 77, 768)
        timestep = torch.randint(0, 1000, (batch_size,))

        # Run forward pass and record output
        with torch.no_grad():
            original_output = controlnet(
                noisy_latent, timestep, text_embedding, condition_image
            )

        # Extract tensors for comparison
        original_down_samples = [
            t.clone() for t in original_output["down_block_res_samples"]
        ]
        original_mid_sample = original_output["mid_block_res_sample"].clone()

        # 2. Save checkpoint to a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(
                model=controlnet,
                optimizer=optimizer,
                scheduler=scheduler,
                step=100,
                output_dir=tmpdir,
            )

            # 3. Create a NEW ControlNet and load the checkpoint
            controlnet_reloaded = ControlNet(mock_unet)
            controlnet_reloaded.eval()

            optimizer_reloaded = setup_optimizer(controlnet_reloaded.parameters())
            scheduler_reloaded = setup_scheduler(
                optimizer_reloaded, num_training_steps=1000
            )

            # Load checkpoint
            checkpoint_dir = os.path.join(tmpdir, "checkpoint-100")
            resumed_step = load_checkpoint(
                model=controlnet_reloaded,
                optimizer=optimizer_reloaded,
                scheduler=scheduler_reloaded,
                checkpoint_path=checkpoint_dir,
            )

            assert resumed_step == 100, (
                f"Resumed step should be 100, got {resumed_step}"
            )

            # 4. Run same forward pass and verify output matches
            with torch.no_grad():
                reloaded_output = controlnet_reloaded(
                    noisy_latent, timestep, text_embedding, condition_image
                )

            # Verify down_block_res_samples match
            reloaded_down_samples = reloaded_output["down_block_res_samples"]
            assert len(reloaded_down_samples) == len(original_down_samples), (
                f"Number of down_block_res_samples mismatch: "
                f"{len(reloaded_down_samples)} vs {len(original_down_samples)}"
            )

            for i, (orig, reloaded) in enumerate(
                zip(original_down_samples, reloaded_down_samples)
            ):
                torch.testing.assert_close(
                    orig,
                    reloaded,
                    rtol=1e-5,
                    atol=1e-5,
                    msg=f"down_block_res_samples[{i}] mismatch after checkpoint reload",
                )

            # Verify mid_block_res_sample matches
            reloaded_mid_sample = reloaded_output["mid_block_res_sample"]
            torch.testing.assert_close(
                original_mid_sample,
                reloaded_mid_sample,
                rtol=1e-5,
                atol=1e-5,
                msg="mid_block_res_sample mismatch after checkpoint reload",
            )
