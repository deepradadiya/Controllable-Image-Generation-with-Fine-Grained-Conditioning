"""
Debug script to understand UNet structure and expected feature dimensions.
"""

import torch
from diffusers import UNet2DConditionModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_unet_structure():
    """Analyze the UNet structure to understand expected feature dimensions."""
    print("Analyzing UNet2DConditionModel structure...")
    
    # Create a standard SD1.5 UNet
    unet = UNet2DConditionModel(
        sample_size=64,  # 512x512 / 8 = 64x64 latent
        in_channels=4,
        out_channels=4,
        block_out_channels=(320, 640, 1280, 1280),
        cross_attention_dim=768,
    )
    
    print(f"\nUNet configuration:")
    print(f"  Sample size: {unet.config.sample_size}")
    print(f"  Block out channels: {unet.config.block_out_channels}")
    print(f"  Down block types: {unet.config.down_block_types}")
    print(f"  Up block types: {unet.config.up_block_types}")
    print(f"  Layers per block: {unet.config.layers_per_block}")
    
    # Test with actual forward pass to see where the error occurs
    batch_size = 1
    height, width = 64, 64
    
    sample = torch.randn(batch_size, 4, height, width)
    timestep = torch.randint(0, 1000, (batch_size,))
    encoder_hidden_states = torch.randn(batch_size, 77, 768)
    
    print(f"\nInput shapes:")
    print(f"  Sample: {sample.shape}")
    print(f"  Timestep: {timestep.shape}")
    print(f"  Encoder hidden states: {encoder_hidden_states.shape}")
    
    # Hook to capture intermediate shapes
    down_block_shapes = []
    
    def capture_down_block_output(module, input, output):
        if isinstance(output, tuple):
            # Down blocks return (hidden_states, res_samples)
            hidden_states, res_samples = output
            down_block_shapes.append({
                'hidden_states': hidden_states.shape if hidden_states is not None else None,
                'res_samples': [s.shape for s in res_samples] if res_samples else None
            })
        else:
            down_block_shapes.append({'output': output.shape})
    
    # Register hooks on down blocks
    hooks = []
    for i, down_block in enumerate(unet.down_blocks):
        hook = down_block.register_forward_hook(capture_down_block_output)
        hooks.append(hook)
    
    # Forward pass
    print(f"\nRunning forward pass...")
    with torch.no_grad():
        output = unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
        )
    
    print(f"Final output shape: {output.sample.shape}")
    
    # Print captured shapes
    print(f"\nDown block outputs:")
    for i, shapes in enumerate(down_block_shapes):
        print(f"  Down block {i}:")
        for key, value in shapes.items():
            print(f"    {key}: {value}")
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Now test with additional residuals to understand the expected format
    print(f"\nTesting with additional residuals...")
    
    # Create test residuals with different shapes to see what works
    test_residuals = []
    
    # Try different configurations
    configs_to_test = [
        # Config 1: Same resolution as input for all
        [(320, height, width), (320, height, width), (640, height, width), (1280, height, width), (1280, height, width)],
        
        # Config 2: Progressive downsampling
        [(320, height, width), (320, height, width), (640, height//2, width//2), (1280, height//4, width//4), (1280, height//8, width//8)],
        
        # Config 3: Different pattern
        [(320, height, width), (320, height//2, width//2), (640, height//4, width//4), (1280, height//8, width//8)],
    ]
    
    for config_idx, config in enumerate(configs_to_test):
        print(f"\nTesting configuration {config_idx + 1}:")
        test_residuals = []
        for channels, h, w in config:
            test_residuals.append(torch.randn(batch_size, channels, h, w))
            print(f"  Residual shape: {test_residuals[-1].shape}")
        
        try:
            with torch.no_grad():
                output = unet(
                    sample=sample,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=tuple(test_residuals),
                )
            print(f"  ✓ Configuration {config_idx + 1} works!")
            print(f"  Output shape: {output.sample.shape}")
            break
        except Exception as e:
            print(f"  ✗ Configuration {config_idx + 1} failed: {e}")
    
    return True


if __name__ == "__main__":
    analyze_unet_structure()