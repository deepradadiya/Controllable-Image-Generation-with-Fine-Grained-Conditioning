"""
Study the official diffusers ControlNet implementation to understand correct integration.
"""

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel as DiffusersControlNet
from diffusers import UNet2DConditionModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def study_official_controlnet():
    """Study how official diffusers ControlNet works."""
    print("Studying official diffusers ControlNet implementation...")
    
    # Create official ControlNet (this will be small for testing)
    print("\n1. Creating official ControlNet...")
    try:
        # Create a minimal ControlNet for testing
        official_controlnet = DiffusersControlNet(
            in_channels=4,
            conditioning_channels=3,
            block_out_channels=(320, 640, 1280, 1280),
            down_block_types=(
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
                "DownBlock2D",
            ),
            cross_attention_dim=768,
        )
        print(f"  Official ControlNet created successfully")
        print(f"  Parameters: {sum(p.numel() for p in official_controlnet.parameters()):,}")
    except Exception as e:
        print(f"  Failed to create official ControlNet: {e}")
        return False
    
    # Create UNet
    print("\n2. Creating UNet...")
    unet = UNet2DConditionModel(
        sample_size=64,
        in_channels=4,
        out_channels=4,
        block_out_channels=(320, 640, 1280, 1280),
        cross_attention_dim=768,
    )
    
    # Test inputs
    batch_size = 1
    height, width = 64, 64
    
    sample = torch.randn(batch_size, 4, height, width)
    timestep = torch.randint(0, 1000, (batch_size,))
    encoder_hidden_states = torch.randn(batch_size, 77, 768)
    condition_map = torch.randn(batch_size, 3, 512, 512)  # Full resolution
    
    print(f"\n3. Testing official ControlNet forward pass...")
    with torch.no_grad():
        controlnet_outputs = official_controlnet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=condition_map,
            return_dict=True,
        )
    
    print(f"  ControlNet output type: {type(controlnet_outputs)}")
    print(f"  ControlNet output keys: {controlnet_outputs.keys()}")
    
    if hasattr(controlnet_outputs, 'down_block_res_samples'):
        print(f"  Down block samples: {len(controlnet_outputs.down_block_res_samples)}")
        for i, sample_tensor in enumerate(controlnet_outputs.down_block_res_samples):
            print(f"    Block {i}: {sample_tensor.shape}")
    
    if hasattr(controlnet_outputs, 'mid_block_res_sample'):
        print(f"  Mid block sample: {controlnet_outputs.mid_block_res_sample.shape}")
    
    # Test UNet integration
    print(f"\n4. Testing UNet integration with official ControlNet...")
    try:
        with torch.no_grad():
            unet_output = unet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=controlnet_outputs.down_block_res_samples,
                mid_block_additional_residual=controlnet_outputs.mid_block_res_sample,
            )
        print(f"  ✓ Integration successful!")
        print(f"  UNet output shape: {unet_output.sample.shape}")
        
        # Print the exact format that works
        print(f"\n5. Successful integration format:")
        print(f"  down_block_additional_residuals type: {type(controlnet_outputs.down_block_res_samples)}")
        print(f"  down_block_additional_residuals length: {len(controlnet_outputs.down_block_res_samples)}")
        print(f"  mid_block_additional_residual type: {type(controlnet_outputs.mid_block_res_sample)}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_controlnet_outputs():
    """Compare our ControlNet with official one."""
    print("\n" + "="*60)
    print("Comparing ControlNet implementations")
    print("="*60)
    
    # Import our ControlNet
    from controlnet import ControlNetModel as OurControlNet
    
    # Create both ControlNets with same config
    config = {
        'conditioning_channels': 3,
        'in_channels': 4,
        'block_out_channels': (320, 640, 1280, 1280),
        'cross_attention_dim': 768,
    }
    
    print("\n1. Creating both ControlNet implementations...")
    
    # Official ControlNet
    official_controlnet = DiffusersControlNet(
        conditioning_channels=config['conditioning_channels'],
        in_channels=config['in_channels'],
        block_out_channels=config['block_out_channels'],
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        cross_attention_dim=config['cross_attention_dim'],
    )
    
    # Our ControlNet
    our_controlnet = OurControlNet(
        conditioning_channels=config['conditioning_channels'],
        in_channels=config['in_channels'],
        block_out_channels=config['block_out_channels'],
        cross_attention_dim=config['cross_attention_dim'],
    )
    
    print(f"  Official ControlNet params: {sum(p.numel() for p in official_controlnet.parameters()):,}")
    print(f"  Our ControlNet params: {sum(p.numel() for p in our_controlnet.parameters()):,}")
    
    # Test inputs
    batch_size = 1
    height, width = 64, 64
    
    sample = torch.randn(batch_size, 4, height, width)
    timestep = torch.randint(0, 1000, (batch_size,))
    encoder_hidden_states = torch.randn(batch_size, 77, 768)
    condition_map = torch.randn(batch_size, 3, 512, 512)
    
    print("\n2. Comparing outputs...")
    
    # Official output
    with torch.no_grad():
        official_output = official_controlnet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=condition_map,
        )
    
    # Our output
    with torch.no_grad():
        our_output = our_controlnet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=condition_map,
        )
    
    print(f"  Official output type: {type(official_output)}")
    print(f"  Our output type: {type(our_output)}")
    
    # Compare structures
    if hasattr(official_output, 'down_block_res_samples') and 'down_block_res_samples' in our_output:
        official_down = official_output.down_block_res_samples
        our_down = our_output['down_block_res_samples']
        
        print(f"  Official down samples: {len(official_down)}")
        print(f"  Our down samples: {len(our_down)}")
        
        for i, (off_sample, our_sample) in enumerate(zip(official_down, our_down)):
            print(f"    Block {i}: Official {off_sample.shape} vs Our {our_sample.shape}")
    
    return True


if __name__ == "__main__":
    print("Studying Official ControlNet Implementation")
    print("="*50)
    
    try:
        success1 = study_official_controlnet()
        success2 = compare_controlnet_outputs()
        
        if success1 and success2:
            print("\n✓ Study completed successfully!")
        else:
            print("\n✗ Study encountered issues")
            
    except Exception as e:
        print(f"\n❌ Study failed: {e}")
        import traceback
        traceback.print_exc()