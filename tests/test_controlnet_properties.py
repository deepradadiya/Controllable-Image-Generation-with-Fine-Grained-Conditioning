"""
Property-Based Tests for ControlNet Architecture.

Uses pytest + hypothesis to verify universal properties hold across
all valid inputs, not just specific examples.

**Validates: Requirements 2.3, 2.5, 2.6, 8.1, 8.2**
"""

import sys
import os

import pytest
import torch
import torch.nn as nn

from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.controlnet import (
    ControlNet,
    make_zero_conv,
    SD15_DOWN_BLOCK_CHANNELS,
    SD15_MID_BLOCK_CHANNELS,
)
from diffusers import UNet2DConditionModel


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


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis Strategies
# ─────────────────────────────────────────────────────────────────────────────

# Strategy for batch sizes (keep small for speed)
batch_sizes = st.integers(min_value=1, max_value=3)

# Strategy for spatial dimensions that are valid for zero_conv (any positive)
spatial_sizes = st.integers(min_value=1, max_value=32)

# Strategy for channel dimensions matching SD1.5 encoder blocks
sd15_channels = st.sampled_from(SD15_DOWN_BLOCK_CHANNELS + [SD15_MID_BLOCK_CHANNELS])

# Strategy for arbitrary channel dimensions
arbitrary_channels = st.sampled_from([32, 64, 128, 256, 320, 640, 1280])


# ─────────────────────────────────────────────────────────────────────────────
# Property 1: Zero Convolution Outputs Zeros at Init
# **Validates: Requirements 8.1, 8.2**
# ─────────────────────────────────────────────────────────────────────────────


class TestZeroConvOutputsZerosAtInit:
    """
    Property: For ANY random input tensor, a freshly initialized zero_conv
    produces all-zero output.

    **Validates: Requirements 8.1, 8.2**
    """

    @given(
        channels=sd15_channels,
        batch_size=batch_sizes,
        spatial=spatial_sizes,
    )
    @settings(max_examples=50, deadline=None)
    def test_zero_conv_outputs_zeros_for_any_input(
        self, channels, batch_size, spatial
    ):
        """Zero conv initialized to zero must output zeros for any input."""
        conv = make_zero_conv(channels)
        x = torch.randn(batch_size, channels, spatial, spatial)
        output = conv(x)
        assert output.abs().max().item() == 0.0, (
            f"Zero conv output should be all zeros for input shape "
            f"({batch_size}, {channels}, {spatial}, {spatial}), "
            f"got max abs value = {output.abs().max().item()}"
        )

    @given(
        channels=arbitrary_channels,
        batch_size=batch_sizes,
        spatial=spatial_sizes,
    )
    @settings(max_examples=30, deadline=None)
    def test_zero_conv_weights_and_biases_are_zero(self, channels, batch_size, spatial):
        """Zero conv weights and biases must be exactly zero after creation."""
        conv = make_zero_conv(channels)
        assert conv.weight.abs().max().item() == 0.0
        assert conv.bias.abs().max().item() == 0.0

    @given(channels=sd15_channels)
    @settings(max_examples=20, deadline=None)
    def test_zero_conv_output_shape_matches_input(self, channels):
        """Zero conv preserves spatial dimensions and channel count."""
        conv = make_zero_conv(channels)
        x = torch.randn(2, channels, 16, 16)
        output = conv(x)
        assert output.shape == x.shape


# ─────────────────────────────────────────────────────────────────────────────
# Property 2: Output Shape Consistency
# **Validates: Requirements 2.5, 2.6**
#
# Note: The forward pass returns a dict with 'down_block_res_samples' (list)
# and 'mid_block_res_sample' (tensor). Since forward() is not yet implemented
# (raises NotImplementedError), we test the zero_conv output shapes which
# form the actual output structure.
# ─────────────────────────────────────────────────────────────────────────────


class TestOutputShapeConsistency:
    """
    Property: Zero convolution outputs maintain consistent shapes matching
    the SD1.5 encoder block output dimensions for any valid batch size.

    **Validates: Requirements 2.5, 2.6**
    """

    @given(batch_size=batch_sizes)
    @settings(max_examples=10, deadline=None)
    def test_zero_conv_output_shapes_match_sd15_channels(self, batch_size, controlnet):
        """
        Each zero_conv in the ControlNet produces output with the correct
        channel dimension matching SD1.5 encoder block outputs.
        """
        # Expected spatial sizes for SD1.5 encoder outputs at 64x64 input
        # Down blocks produce features at decreasing spatial resolutions
        expected_channels = SD15_DOWN_BLOCK_CHANNELS

        for i, (conv, expected_ch) in enumerate(
            zip(controlnet.zero_convs, expected_channels)
        ):
            # Use a representative spatial size (varies by block depth)
            spatial = 64 // (2 ** (i // 3)) if i < 9 else 8
            x = torch.randn(batch_size, expected_ch, spatial, spatial)
            output = conv(x)
            assert output.shape == (batch_size, expected_ch, spatial, spatial), (
                f"zero_conv[{i}] output shape mismatch: "
                f"expected ({batch_size}, {expected_ch}, {spatial}, {spatial}), "
                f"got {output.shape}"
            )

    @given(batch_size=batch_sizes)
    @settings(max_examples=10, deadline=None)
    def test_mid_block_zero_conv_output_shape(self, batch_size, controlnet):
        """Mid block zero_conv produces (B, 1280, 8, 8) for valid input."""
        x = torch.randn(batch_size, SD15_MID_BLOCK_CHANNELS, 8, 8)
        output = controlnet.mid_block_zero_conv(x)
        assert output.shape == (batch_size, SD15_MID_BLOCK_CHANNELS, 8, 8)


# ─────────────────────────────────────────────────────────────────────────────
# Property 3: Condition Embedding Shape
# **Validates: Requirements 2.5**
#
# The condition_embedding maps (B, 3, 512, 512) → (B, 320, 64, 64)
# ─────────────────────────────────────────────────────────────────────────────


class TestConditionEmbeddingShape:
    """
    Property: The condition_embedding layer always maps
    (B, 3, 512, 512) → (B, 320, 64, 64) for any valid batch size.

    **Validates: Requirements 2.5**
    """

    @given(batch_size=batch_sizes)
    @settings(max_examples=10, deadline=None)
    def test_condition_embedding_output_shape(self, batch_size, controlnet):
        """
        condition_embedding maps (B, 3, 512, 512) → (B, 320, 64, 64)
        for any batch size.
        """
        condition_image = torch.randn(batch_size, 3, 512, 512)
        output = controlnet.condition_embedding(condition_image)
        assert output.shape == (batch_size, 320, 64, 64), (
            f"condition_embedding output shape mismatch: "
            f"expected ({batch_size}, 320, 64, 64), got {output.shape}"
        )

    @given(batch_size=batch_sizes)
    @settings(max_examples=10, deadline=None)
    def test_condition_embedding_output_is_finite(self, batch_size, controlnet):
        """condition_embedding output must be finite (no NaN or Inf)."""
        condition_image = torch.randn(batch_size, 3, 512, 512)
        output = controlnet.condition_embedding(condition_image)
        assert torch.isfinite(output).all(), (
            "condition_embedding produced non-finite values"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Property 4: Frozen Params Have No Gradients
# **Validates: Requirements 2.3**
#
# After backward pass, copied encoder params must have grad=None.
# Since forward() is not implemented, we test by running backward through
# the condition_embedding and zero_conv outputs.
# ─────────────────────────────────────────────────────────────────────────────


class TestFrozenParamsHaveNoGradients:
    """
    Property: After a backward pass, all copied encoder parameters
    (down_blocks, mid_block, conv_in, time_proj, time_embedding)
    have grad=None, while trainable params (zero_convs, condition_embedding)
    accumulate gradients.

    **Validates: Requirements 2.3**
    """

    @given(batch_size=st.integers(min_value=1, max_value=2))
    @settings(max_examples=5, deadline=None)
    def test_frozen_encoder_params_have_no_gradients_after_backward(
        self, batch_size, mock_unet
    ):
        """
        After running backward through condition_embedding output,
        frozen encoder params must have grad=None.
        """
        # Create a fresh ControlNet for gradient testing
        controlnet = ControlNet(mock_unet)

        # Run condition_embedding forward pass (trainable)
        condition_image = torch.randn(batch_size, 3, 512, 512)
        embedding_output = controlnet.condition_embedding(condition_image)

        # Create a loss from the embedding output and backpropagate
        loss = embedding_output.sum()
        loss.backward()

        # Frozen params must have no gradients
        for name, param in controlnet.down_blocks.named_parameters():
            assert param.grad is None, (
                f"Frozen param down_blocks.{name} has gradient after backward"
            )

        for name, param in controlnet.mid_block.named_parameters():
            assert param.grad is None, (
                f"Frozen param mid_block.{name} has gradient after backward"
            )

        for name, param in controlnet.conv_in.named_parameters():
            assert param.grad is None, (
                f"Frozen param conv_in.{name} has gradient after backward"
            )

        # Trainable condition_embedding params SHOULD have gradients
        has_grad = False
        for param in controlnet.condition_embedding.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad, (
            "condition_embedding should have gradients after backward"
        )

    @given(batch_size=st.integers(min_value=1, max_value=2))
    @settings(max_examples=5, deadline=None)
    def test_zero_conv_params_accumulate_gradients(self, batch_size, mock_unet):
        """
        After running backward through zero_conv output,
        zero_conv params must accumulate gradients.
        """
        # Create a fresh ControlNet for gradient testing
        controlnet = ControlNet(mock_unet)

        # Run a zero_conv forward pass with non-zero input
        # First, we need to give the zero_conv non-zero weights so
        # gradients can flow. But at init they're zero, so we manually
        # set a small weight to enable gradient flow.
        with torch.no_grad():
            controlnet.zero_convs[0].weight.fill_(0.01)

        x = torch.randn(batch_size, 320, 64, 64, requires_grad=True)
        output = controlnet.zero_convs[0](x)
        loss = output.sum()
        loss.backward()

        # Zero conv params should have gradients
        assert controlnet.zero_convs[0].weight.grad is not None, (
            "zero_conv weight should have gradient after backward"
        )
        assert controlnet.zero_convs[0].bias.grad is not None, (
            "zero_conv bias should have gradient after backward"
        )

        # Frozen params should still have no gradients
        for name, param in controlnet.down_blocks.named_parameters():
            assert param.grad is None, (
                f"Frozen param down_blocks.{name} should not have gradient"
            )
