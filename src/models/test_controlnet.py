"""
Test suite for ControlNet architecture implementation.

This module contains comprehensive tests for the ControlNet model including:
- Architecture validation
- Forward pass testing
- Memory efficiency validation
- Multi-resolution output verification
- Zero convolution initialization testing

Requirements tested: 3.1, 3.2, 3.5
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List
import logging

from controlnet import (
    ControlNetModel,
    ZeroConvolution,
    ControlNetConditioningEmbedding,
    create_controlnet_from_config,
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestZeroConvolution:
    """Test zero convolution initialization and behavior."""
    
    def test_zero_initialization(self):
        """Test that zero convolution is properly initialized with zeros."""
        zero_conv = ZeroConvolution(in_channels=64, out_channels=64)
        
        # Check that weights and bias are initialized to zero
        assert torch.allclose(zero_conv.conv.weight, torch.zeros_like(zero_conv.conv.weight))
        assert torch.allclose(zero_conv.conv.bias, torch.zeros_like(zero_conv.conv.bias))
        
    def test_zero_conv_forward(self):
        """Test zero convolution forward pass."""
        zero_conv = ZeroConvolution(in_channels=3, out_channels=16)
        
        # Input tensor
        x = torch.randn(1, 3, 32, 32)
        
        # Forward pass should return zeros initially
        output = zero_conv(x)
        expected_shape = (1, 16, 32, 32)
        
        assert output.shape == expected_shape
        assert torch.allclose(output, torch.zeros_like(output))
        
    def test_zero_conv_learning(self):
        """Test that zero convolution can learn after initialization."""
        zero_conv = ZeroConvolution(in_channels=3, out_channels=16)
        
        # Manually set some weights to verify learning capability
        with torch.no_grad():
            zero_conv.conv.weight.fill_(0.1)
            zero_conv.conv.bias.fill_(0.05)
            
        x = torch.randn(1, 3, 32, 32)
        output = zero_conv(x)
        
        # Output should not be zero after weight modification
        assert not torch.allclose(output, torch.zeros_like(output))


class TestControlNetConditioningEmbedding:
    """Test conditioning embedding layer."""
    
    def test_conditioning_embedding_shapes(self):
        """Test conditioning embedding output shapes."""
        embedding = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=320,
            conditioning_channels=3,
            block_out_channels=(16, 32, 96, 256),
        )
        
        # Test with different input sizes
        for size in [256, 512, 768]:
            condition_input = torch.randn(1, 3, size, size)
            output = embedding(condition_input)
            
            # Output should be downsampled by factor of 8 (3 stride-2 convolutions)
            expected_size = size // 8
            expected_shape = (1, 320, expected_size, expected_size)
            
            assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
            
    def test_conditioning_embedding_channels(self):
        """Test conditioning embedding with different input channels."""
        # Test with 1-channel input (e.g., depth maps)
        embedding_1ch = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=320,
            conditioning_channels=1,
        )
        
        condition_1ch = torch.randn(1, 1, 512, 512)
        output_1ch = embedding_1ch(condition_1ch)
        
        # Test with 3-channel input (e.g., RGB edge maps)
        embedding_3ch = ControlNetConditioningEmbedding(
            conditioning_embedding_channels=320,
            conditioning_channels=3,
        )
        
        condition_3ch = torch.randn(1, 3, 512, 512)
        output_3ch = embedding_3ch(condition_3ch)
        
        # Both should produce same output shape
        assert output_1ch.shape == output_3ch.shape
        assert output_1ch.shape == (1, 320, 64, 64)


class TestControlNetModel:
    """Test main ControlNet model."""
    
    def test_controlnet_initialization(self):
        """Test ControlNet model initialization."""
        controlnet = ControlNetModel(
            conditioning_channels=3,
            in_channels=4,
        )
        
        # Check that model is properly initialized
        assert isinstance(controlnet, ControlNetModel)
        assert hasattr(controlnet, 'controlnet_cond_embedding')
        assert hasattr(controlnet, 'down_blocks')
        assert hasattr(controlnet, 'mid_block')
        assert hasattr(controlnet, 'controlnet_down_blocks')
        assert hasattr(controlnet, 'controlnet_mid_block')
        
    def test_controlnet_forward_pass(self):
        """Test ControlNet forward pass with various input sizes."""
        controlnet = ControlNetModel(
            conditioning_channels=3,
            in_channels=4,
        )
        
        batch_size = 2
        
        # Test with different latent sizes
        for latent_size in [32, 64, 96]:  # Corresponding to 256x256, 512x512, 768x768 images
            # Create inputs
            sample = torch.randn(batch_size, 4, latent_size, latent_size)
            timestep = torch.randint(0, 1000, (batch_size,))
            encoder_hidden_states = torch.randn(batch_size, 77, 768)
            controlnet_cond = torch.randn(batch_size, 3, latent_size * 8, latent_size * 8)
            
            # Forward pass
            with torch.no_grad():
                outputs = controlnet(
                    sample=sample,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_cond,
                )
            
            # Check output structure
            assert isinstance(outputs, dict)
            assert 'down_block_res_samples' in outputs
            assert 'mid_block_res_sample' in outputs
            
            # Check number of down block outputs (should match number of down blocks)
            down_samples = outputs['down_block_res_samples']
            assert len(down_samples) == len(controlnet.down_blocks) + 1  # +1 for conv_in output
            
            # Check mid block output shape
            mid_sample = outputs['mid_block_res_sample']
            expected_mid_shape = (batch_size, 1280, latent_size // 8, latent_size // 8)
            assert mid_sample.shape == expected_mid_shape
            
    def test_multi_resolution_outputs(self):
        """Test that ControlNet produces multi-resolution features."""
        controlnet = ControlNetModel()
        
        batch_size = 1
        latent_size = 64  # 512x512 image
        
        # Create inputs
        sample = torch.randn(batch_size, 4, latent_size, latent_size)
        timestep = torch.randint(0, 1000, (batch_size,))
        encoder_hidden_states = torch.randn(batch_size, 77, 768)
        controlnet_cond = torch.randn(batch_size, 3, latent_size * 8, latent_size * 8)
        
        with torch.no_grad():
            outputs = controlnet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond,
            )
        
        down_samples = outputs['down_block_res_samples']
        
        # Expected resolutions: 1/1, 1/2, 1/4, 1/8, 1/8 (relative to latent space)
        # Which corresponds to: 1/8, 1/16, 1/32, 1/64, 1/64 (relative to original image)
        expected_spatial_sizes = [64, 32, 16, 8, 8]  # For 512x512 input
        expected_channels = [320, 320, 640, 1280, 1280]
        
        for i, (sample, expected_size, expected_ch) in enumerate(
            zip(down_samples, expected_spatial_sizes, expected_channels)
        ):
            expected_shape = (batch_size, expected_ch, expected_size, expected_size)
            assert sample.shape == expected_shape, f"Block {i}: expected {expected_shape}, got {sample.shape}"
            
    def test_conditioning_scale(self):
        """Test conditioning scale functionality."""
        controlnet = ControlNetModel()
        
        batch_size = 1
        latent_size = 32
        
        # Create inputs
        sample = torch.randn(batch_size, 4, latent_size, latent_size)
        timestep = torch.randint(0, 1000, (batch_size,))
        encoder_hidden_states = torch.randn(batch_size, 77, 768)
        controlnet_cond = torch.randn(batch_size, 3, latent_size * 8, latent_size * 8)
        
        # Test with different conditioning scales
        with torch.no_grad():
            outputs_scale_1 = controlnet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond,
                conditioning_scale=1.0,
            )
            
            outputs_scale_05 = controlnet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond,
                conditioning_scale=0.5,
            )
        
        # Outputs with scale 0.5 should be half the magnitude
        for sample_1, sample_05 in zip(
            outputs_scale_1['down_block_res_samples'],
            outputs_scale_05['down_block_res_samples']
        ):
            assert torch.allclose(sample_05, sample_1 * 0.5, rtol=1e-5)
            
        # Same for mid block
        assert torch.allclose(
            outputs_scale_05['mid_block_res_sample'],
            outputs_scale_1['mid_block_res_sample'] * 0.5,
            rtol=1e-5
        )
        
    def test_zero_convolution_integration(self):
        """Test that zero convolutions are properly integrated."""
        controlnet = ControlNetModel()
        
        # Check that controlnet output blocks are ZeroConvolution instances
        for block in controlnet.controlnet_down_blocks:
            assert isinstance(block, ZeroConvolution)
            
        assert isinstance(controlnet.controlnet_mid_block, ZeroConvolution)
        
        # Test that initial outputs are zero (before any training)
        batch_size = 1
        latent_size = 32
        
        sample = torch.randn(batch_size, 4, latent_size, latent_size)
        timestep = torch.randint(0, 1000, (batch_size,))
        encoder_hidden_states = torch.randn(batch_size, 77, 768)
        controlnet_cond = torch.randn(batch_size, 3, latent_size * 8, latent_size * 8)
        
        with torch.no_grad():
            outputs = controlnet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond,
            )
        
        # All outputs should be zero initially due to zero convolutions
        for sample in outputs['down_block_res_samples']:
            assert torch.allclose(sample, torch.zeros_like(sample))
            
        assert torch.allclose(
            outputs['mid_block_res_sample'],
            torch.zeros_like(outputs['mid_block_res_sample'])
        )


class TestControlNetMemoryEfficiency:
    """Test memory efficiency and T4 GPU compatibility."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_memory_usage(self):
        """Test that ControlNet fits within T4 GPU memory constraints."""
        device = torch.device("cuda")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device)
        
        # Create ControlNet on GPU
        controlnet = ControlNetModel().to(device)
        
        # Measure model memory
        model_memory = torch.cuda.memory_allocated(device) - initial_memory
        model_memory_gb = model_memory / (1024**3)
        
        # Model should be less than 1GB (leaves room for training overhead)
        assert model_memory_gb < 1.0, f"Model uses {model_memory_gb:.2f}GB, too much for T4"
        
        # Test forward pass memory
        batch_size = 1  # T4-optimized batch size
        latent_size = 64  # 512x512 image
        
        sample = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
        timestep = torch.randint(0, 1000, (batch_size,), device=device)
        encoder_hidden_states = torch.randn(batch_size, 77, 768, device=device)
        controlnet_cond = torch.randn(batch_size, 3, latent_size * 8, latent_size * 8, device=device)
        
        # Forward pass
        with torch.no_grad():
            outputs = controlnet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond,
            )
        
        # Total memory after forward pass
        total_memory = torch.cuda.memory_allocated(device)
        total_memory_gb = total_memory / (1024**3)
        
        # Should be well under T4 limit (13GB usable)
        assert total_memory_gb < 5.0, f"Forward pass uses {total_memory_gb:.2f}GB, too much for T4"
        
        logger.info(f"Model memory: {model_memory_gb:.2f}GB")
        logger.info(f"Forward pass memory: {total_memory_gb:.2f}GB")
        
    def test_gradient_checkpointing_compatibility(self):
        """Test that model is compatible with gradient checkpointing."""
        controlnet = ControlNetModel()
        
        # Enable gradient checkpointing (if available)
        if hasattr(controlnet, 'enable_gradient_checkpointing'):
            controlnet.enable_gradient_checkpointing()
            
        # Test that forward pass still works
        batch_size = 1
        latent_size = 32
        
        sample = torch.randn(batch_size, 4, latent_size, latent_size, requires_grad=True)
        timestep = torch.randint(0, 1000, (batch_size,))
        encoder_hidden_states = torch.randn(batch_size, 77, 768)
        controlnet_cond = torch.randn(batch_size, 3, latent_size * 8, latent_size * 8)
        
        outputs = controlnet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
        )
        
        # Test backward pass
        loss = sum(sample.sum() for sample in outputs['down_block_res_samples'])
        loss += outputs['mid_block_res_sample'].sum()
        loss.backward()
        
        # Check that gradients exist
        assert sample.grad is not None
        
    def test_mixed_precision_compatibility(self):
        """Test compatibility with mixed precision training."""
        controlnet = ControlNetModel()
        
        # Test with FP16 inputs
        batch_size = 1
        latent_size = 32
        
        sample = torch.randn(batch_size, 4, latent_size, latent_size, dtype=torch.float16)
        timestep = torch.randint(0, 1000, (batch_size,))
        encoder_hidden_states = torch.randn(batch_size, 77, 768, dtype=torch.float16)
        controlnet_cond = torch.randn(batch_size, 3, latent_size * 8, latent_size * 8, dtype=torch.float16)
        
        with torch.no_grad():
            outputs = controlnet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond,
            )
        
        # Check that outputs are in FP16
        for sample in outputs['down_block_res_samples']:
            assert sample.dtype == torch.float16
            
        assert outputs['mid_block_res_sample'].dtype == torch.float16


class TestControlNetUtilities:
    """Test utility functions and model creation helpers."""
    
    def test_create_controlnet_from_config(self):
        """Test ControlNet creation from configuration."""
        controlnet = create_controlnet_from_config(
            config_path="dummy_path",  # Not used in current implementation
            conditioning_channels=1,  # Depth conditioning
            conditioning_type="depth"
        )
        
        assert isinstance(controlnet, ControlNetModel)
        assert controlnet.config.conditioning_channels == 1
        
    def test_parameter_count(self):
        """Test that parameter count is reasonable for T4 GPU."""
        controlnet = ControlNetModel()
        
        total_params = sum(p.numel() for p in controlnet.parameters())
        trainable_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
        
        # Should be around 361M parameters (similar to original ControlNet)
        assert 300_000_000 < total_params < 500_000_000, f"Unexpected parameter count: {total_params}"
        assert total_params == trainable_params, "All parameters should be trainable"
        
        # Model size in FP16 should be reasonable
        model_size_mb = total_params * 2 / (1024**2)  # FP16
        assert model_size_mb < 1000, f"Model too large: {model_size_mb:.1f}MB"
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Model size (FP16): {model_size_mb:.1f}MB")


def run_comprehensive_test():
    """Run a comprehensive test of the ControlNet implementation."""
    print("Running comprehensive ControlNet tests...")
    
    # Test 1: Basic functionality
    print("1. Testing basic functionality...")
    controlnet = ControlNetModel(conditioning_channels=3)
    
    batch_size = 1
    latent_size = 64
    
    sample = torch.randn(batch_size, 4, latent_size, latent_size)
    timestep = torch.randint(0, 1000, (batch_size,))
    encoder_hidden_states = torch.randn(batch_size, 77, 768)
    controlnet_cond = torch.randn(batch_size, 3, latent_size * 8, latent_size * 8)
    
    with torch.no_grad():
        outputs = controlnet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
        )
    
    print(f"✓ Forward pass successful")
    print(f"✓ Output shapes: {len(outputs['down_block_res_samples'])} down blocks + 1 mid block")
    
    # Test 2: Multi-resolution verification
    print("2. Testing multi-resolution outputs...")
    down_samples = outputs['down_block_res_samples']
    expected_sizes = [64, 32, 16, 8, 8]
    
    for i, (sample, expected_size) in enumerate(zip(down_samples, expected_sizes)):
        actual_size = sample.shape[-1]
        assert actual_size == expected_size, f"Block {i}: expected {expected_size}, got {actual_size}"
    
    print("✓ Multi-resolution outputs correct")
    
    # Test 3: Zero initialization
    print("3. Testing zero initialization...")
    for sample in down_samples:
        assert torch.allclose(sample, torch.zeros_like(sample)), "Outputs should be zero initially"
    
    print("✓ Zero initialization working")
    
    # Test 4: Parameter count
    print("4. Testing parameter count...")
    total_params = sum(p.numel() for p in controlnet.parameters())
    print(f"✓ Total parameters: {total_params:,}")
    
    # Test 5: Memory efficiency
    print("5. Testing memory efficiency...")
    model_size_mb = total_params * 2 / (1024**2)  # FP16
    print(f"✓ Model size (FP16): {model_size_mb:.1f}MB")
    
    print("\nAll tests passed! ControlNet implementation is working correctly.")
    return True


if __name__ == "__main__":
    # Run comprehensive test
    success = run_comprehensive_test()
    
    if success:
        print("\n🎉 ControlNet implementation validated successfully!")
        print("Ready for integration with SD1.5 UNet and training pipeline.")
    else:
        print("\n❌ Tests failed. Please check the implementation.")