"""
Tests for zero convolution layers in ControlNet.

Validates Requirements 2.4, 8.1, 8.2:
- Zero convolutions are 1x1 Conv2d with weights and biases initialized to 0.0
- At init, zero_convs output zero tensors for ANY input
- Zero_convs are trainable (requires_grad=True)
- Channel dimensions match SD1.5 encoder block outputs
"""

import pytest
import torch
import torch.nn as nn

from controlnet import (
    make_zero_conv,
    ControlNet,
    SD15_DOWN_BLOCK_CHANNELS,
    SD15_MID_BLOCK_CHANNELS,
)


class TestMakeZeroConv:
    """Test the make_zero_conv helper function."""

    def test_returns_conv2d_with_kernel_size_1(self):
        """Zero conv is a 1x1 convolution."""
        conv = make_zero_conv(320)
        assert isinstance(conv, nn.Conv2d)
        assert conv.kernel_size == (1, 1)

    def test_input_output_channels_match(self):
        """Input and output channels must be equal."""
        for ch in [320, 640, 1280]:
            conv = make_zero_conv(ch)
            assert conv.in_channels == ch
            assert conv.out_channels == ch

    def test_weights_initialized_to_zero(self):
        """All weights must be exactly 0.0 after initialization."""
        conv = make_zero_conv(320)
        assert conv.weight.sum().item() == 0.0
        assert conv.weight.abs().max().item() == 0.0

    def test_biases_initialized_to_zero(self):
        """All biases must be exactly 0.0 after initialization."""
        conv = make_zero_conv(320)
        assert conv.bias.sum().item() == 0.0
        assert conv.bias.abs().max().item() == 0.0

    def test_output_is_zero_for_any_input(self):
        """For ANY input, a freshly initialized zero_conv outputs all zeros."""
        conv = make_zero_conv(640)
        # Test with random inputs of various shapes
        for spatial in [8, 16, 32, 64]:
            x = torch.randn(2, 640, spatial, spatial)
            output = conv(x)
            assert output.abs().max().item() == 0.0, (
                f"Zero conv output should be all zeros, got max={output.abs().max().item()}"
            )

    def test_output_shape_matches_input(self):
        """Output spatial dimensions and channels match input."""
        conv = make_zero_conv(1280)
        x = torch.randn(1, 1280, 8, 8)
        output = conv(x)
        assert output.shape == x.shape

    def test_requires_grad_is_true(self):
        """Zero conv layers must be trainable."""
        conv = make_zero_conv(320)
        assert conv.weight.requires_grad is True
        assert conv.bias.requires_grad is True


class TestZeroConvsInControlNet:
    """Test zero_convs integration within the ControlNet class."""

    @pytest.fixture
    def mock_unet(self):
        """Create a minimal mock UNet for testing without downloading SD1.5."""
        from unittest.mock import MagicMock, patch
        from diffusers import UNet2DConditionModel

        # We need a real UNet2DConditionModel with the right structure
        # Use a minimal config that matches SD1.5 channel structure
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

    def test_zero_convs_is_module_list(self, mock_unet):
        """zero_convs should be an nn.ModuleList."""
        controlnet = ControlNet(mock_unet)
        assert isinstance(controlnet.zero_convs, nn.ModuleList)

    def test_correct_number_of_zero_convs(self, mock_unet):
        """Should have 12 zero_convs for down blocks + 1 for mid block."""
        controlnet = ControlNet(mock_unet)
        assert len(controlnet.zero_convs) == 12  # SD1.5 has 12 down block outputs
        assert isinstance(controlnet.mid_block_zero_conv, nn.Conv2d)

    def test_zero_conv_channel_dimensions(self, mock_unet):
        """Each zero_conv channel dimension matches the encoder block output."""
        controlnet = ControlNet(mock_unet)
        for i, (conv, expected_ch) in enumerate(
            zip(controlnet.zero_convs, SD15_DOWN_BLOCK_CHANNELS)
        ):
            assert conv.in_channels == expected_ch, (
                f"zero_conv[{i}] in_channels={conv.in_channels}, expected {expected_ch}"
            )
            assert conv.out_channels == expected_ch, (
                f"zero_conv[{i}] out_channels={conv.out_channels}, expected {expected_ch}"
            )

    def test_mid_block_zero_conv_channels(self, mock_unet):
        """Mid block zero_conv should have 1280 channels."""
        controlnet = ControlNet(mock_unet)
        assert controlnet.mid_block_zero_conv.in_channels == SD15_MID_BLOCK_CHANNELS
        assert controlnet.mid_block_zero_conv.out_channels == SD15_MID_BLOCK_CHANNELS

    def test_all_zero_convs_initialized_to_zero(self, mock_unet):
        """All zero_conv weights and biases must be exactly 0.0."""
        controlnet = ControlNet(mock_unet)
        for i, conv in enumerate(controlnet.zero_convs):
            assert conv.weight.abs().max().item() == 0.0, (
                f"zero_conv[{i}] weights not zero"
            )
            assert conv.bias.abs().max().item() == 0.0, (
                f"zero_conv[{i}] biases not zero"
            )
        # Mid block
        assert controlnet.mid_block_zero_conv.weight.abs().max().item() == 0.0
        assert controlnet.mid_block_zero_conv.bias.abs().max().item() == 0.0

    def test_zero_convs_are_trainable(self, mock_unet):
        """Zero convs must have requires_grad=True."""
        controlnet = ControlNet(mock_unet)
        for conv in controlnet.zero_convs:
            assert conv.weight.requires_grad is True
            assert conv.bias.requires_grad is True
        assert controlnet.mid_block_zero_conv.weight.requires_grad is True
        assert controlnet.mid_block_zero_conv.bias.requires_grad is True

    def test_only_zero_convs_and_condition_embedding_are_trainable(self, mock_unet):
        """Only zero_convs and condition_embedding should be trainable."""
        controlnet = ControlNet(mock_unet)

        trainable_modules = set()
        for name, param in controlnet.named_parameters():
            if param.requires_grad:
                # Extract the top-level module name
                top_module = name.split(".")[0]
                trainable_modules.add(top_module)

        # Only these should be trainable
        expected_trainable = {"zero_convs", "mid_block_zero_conv", "condition_embedding"}
        assert trainable_modules == expected_trainable, (
            f"Unexpected trainable modules: {trainable_modules - expected_trainable}"
        )

    def test_encoder_blocks_are_frozen(self, mock_unet):
        """Copied encoder blocks must have requires_grad=False."""
        controlnet = ControlNet(mock_unet)
        for param in controlnet.down_blocks.parameters():
            assert param.requires_grad is False
        for param in controlnet.mid_block.parameters():
            assert param.requires_grad is False
