# Implementation Plan: ControlNet Model Architecture & Training Loops

## Overview

This plan implements the ControlNet model architecture and all 3 training loops from scratch. The implementation covers `model/controlnet.py`, `model/pipeline.py`, `training/losses.py`, and the three training scripts (`training/train_depth.py`, `training/train_pose.py`, `training/train_edge.py`).

## Tasks

- [x] 1. ControlNet Architecture (`model/controlnet.py`)
  - [x] 1.1 Create `model/controlnet.py` with ASCII art architecture documentation
    - Add top-level comment with ASCII art showing: Condition Image → ControlNet Encoder (trainable copy) → Zero Convolutions → SD1.5 UNet (frozen) → Generated Image
    - Explain that ControlNet is a COPY of the SD1.5 encoder with an extra input channel
    - Explain zero convolutions are 1x1 conv layers initialized to zero — model starts identical to vanilla SD1.5
    - Explain SD1.5 UNet is 100% frozen, only ~360M adapter params are trained
    - Explain why this fits on T4: never backprop through the full UNet
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [x] 1.2 Implement the ControlNet class with condition embedding
    - Create `ControlNet(nn.Module)` class accepting `condition_channels=3`
    - Implement `condition_embedding` layer: small CNN that projects (B, 3, 512, 512) condition image to (B, 4, 64, 64) latent space
    - Copy encoder blocks from SD1.5 UNet (`down_blocks` and `mid_block`) using deep copy
    - Load pretrained weights into copied encoder blocks
    - Freeze all copied encoder block weights with `requires_grad_(False)`
    - _Requirements: 2.1, 2.2, 2.3, 2.5_

  - [x] 1.3 Implement zero convolution layers
    - Add trainable `nn.Conv2d(ch, ch, kernel_size=1)` zero_conv layers for each encoder block output
    - Initialize all zero_conv weights to zero using `nn.init.zeros_`
    - Initialize all zero_conv biases to zero
    - Verify zero_convs are the ONLY trainable layers (along with condition_embedding)
    - _Requirements: 2.4, 8.1, 8.2_

  - [x] 1.4 Implement forward pass and parameter counting
    - Implement `forward(noisy_latent, timestep, text_embedding, condition_image)` method
    - Forward pass: embed condition → add to noisy_latent → run through copied encoder → apply zero_convs → inject into frozen UNet decoder → return predicted noise (B, 4, 64, 64)
    - Print trainable vs frozen parameter counts at `__init__` time
    - _Requirements: 2.6, 2.7, 9.1, 9.2_

  - [x] 1.5 Write property tests for ControlNet architecture
    - **Property: Zero convolution outputs zeros at init** — for any random input, freshly initialized zero_conv produces all-zero output
    - **Property: Output shape consistency** — forward pass always returns (B, 4, 64, 64) for valid inputs
    - **Property: Condition embedding shape** — condition_embedding maps (B, 3, 512, 512) → (B, 4, 64, 64)
    - **Property: Frozen params have no gradients** — after backward, copied encoder params have grad=None
    - _Validates: Requirements 2.3, 2.5, 2.6, 8.1, 8.2_

- [x] 2. Diffusion Loss (`training/losses.py`)
  - [x] 2.1 Implement diffusion loss with documentation
    - Create `training/losses.py` with explanatory comment: noise is added to images, model predicts that noise, condition guides which image to reconstruct
    - Implement `compute_diffusion_loss(model_pred, noise, timesteps, step)` using `F.mse_loss`
    - Add loss logging every 10 steps via print statement
    - _Requirements: 4.1, 4.2, 4.3_

  - [x] 2.2 Write property tests for diffusion loss
    - **Property: Loss is non-negative** — MSE loss >= 0 for any inputs
    - **Property: Loss is zero when pred == target** — MSE(x, x) == 0
    - **Property: Loss is symmetric** — MSE(a, b) == MSE(b, a)
    - _Validates: Requirement 4.2_

- [x] 3. Inference Pipeline (`model/pipeline.py`)
  - [x] 3.1 Implement ControlNetPipeline class
    - Create `model/pipeline.py` with `ControlNetPipeline` class
    - Accept `controlnet`, `unet`, `vae`, `text_encoder`, `tokenizer`, `scheduler` in constructor
    - Implement `__call__` with parameters: `text_prompt`, `condition_image`, `condition_type`, `guidance_scale=7.5`, `num_inference_steps=20`, `seed`
    - _Requirements: 3.1, 3.4_

  - [x] 3.2 Implement classifier-free guidance and DDIM sampling
    - Add comment explaining CFG: compute conditional and unconditional predictions, interpolate with guidance_scale
    - Implement DDIM denoising loop: for each step, run UNet twice (with/without conditioning), apply CFG formula
    - Decode final latent with VAE decoder to produce PIL Image
    - _Requirements: 3.2, 3.3_

  - [x] 3.3 Implement condition overlay output
    - Implement `save_with_overlay` method that creates side-by-side composite: condition image | generated image
    - Label the composite with condition_type text
    - Save as PNG file suitable for visual inspection
    - _Requirements: 3.5, 11.1, 11.2, 11.3_

  - [x] 3.4 Write unit tests for inference pipeline
    - Test pipeline initialization with mock models
    - Test that output is a valid PIL Image of size 512x512
    - Test that condition_type validates against {"depth", "pose", "edge"}
    - Test overlay composite is saved with correct dimensions
    - _Validates: Requirements 3.1, 3.2, 3.4, 3.5_

- [x] 4. Depth Training Script (`training/train_depth.py`)
  - [x] 4.1 Implement model loading and optimizer setup
    - Load SD1.5 VAE, UNet, CLIP text encoder — ALL with `requires_grad_(False)`
    - Create trainable ControlNet adapter
    - Set up AdamW optimizer with lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-2 — only ControlNet params
    - Set up cosine LR schedule with warmup
    - Set up `torch.cuda.amp.GradScaler` for mixed precision
    - Add warning comment: "Estimated training time on T4: ~3 hours. Consider splitting across sessions."
    - _Requirements: 5.1, 5.2, 5.3, 5.15_

  - [x] 4.2 Implement the training loop
    - Encode image to latent with VAE
    - Sample random noise and timestep, add noise to latent
    - Encode text prompt with CLIP
    - Run ControlNet forward pass with condition image
    - Compute MSE loss using `compute_diffusion_loss`
    - Wrap forward pass in `torch.autocast("cuda", dtype=torch.float16)` with comment explaining: "FP16 halves VRAM by storing activations in float16"
    - Apply gradient clipping at max_norm=1.0 with comment: "Prevents exploding gradients that destabilize training"
    - Step optimizer, scaler, and LR scheduler
    - _Requirements: 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10, 5.11_

  - [x] 4.3 Implement logging and checkpointing
    - Initialize W&B run with project name and config
    - Log loss, learning_rate, and sample_images to W&B every 250 steps
    - Save checkpoint (model state_dict, optimizer state, step) to Google Drive every 250 steps
    - At training end, save adapter weights and upload to HuggingFace Hub at "{username}/controlnet-sd15-depth"
    - _Requirements: 5.12, 5.13, 5.14_

  - [x] 4.4 Write property tests for training loop
    - **Property: Frozen params never get gradients** — after any training step, UNet/VAE/CLIP params have grad=None
    - **Property: Gradient norm ≤ 1.0 after clipping** — total gradient norm is bounded
    - **Property: Loss decreases over multiple steps** — running average loss at step N < step 0 (with synthetic data)
    - _Validates: Requirements 5.9, 5.11_

- [x] 5. Pose Training Script (`training/train_pose.py`)
  - [x] 5.1 Implement pose training script
    - Copy structure from `train_depth.py` with `condition_type = "pose"`
    - Load pose condition maps from dataset
    - Set HuggingFace Hub repo to "{username}/controlnet-sd15-pose"
    - Include same FP16, gradient clipping, W&B logging, Drive checkpointing
    - _Requirements: 6.1, 6.2, 6.3_

- [x] 6. Edge Training Script (`training/train_edge.py`)
  - [x] 6.1 Implement edge training script
    - Copy structure from `train_depth.py` with `condition_type = "edge"`
    - Load edge condition maps from dataset
    - Set HuggingFace Hub repo to "{username}/controlnet-sd15-edge"
    - Include same FP16, gradient clipping, W&B logging, Drive checkpointing
    - _Requirements: 7.1, 7.2, 7.3_

- [x] 7. Shared Training Infrastructure
  - [x] 7.1 Extract shared training utilities
    - Create `training/utils.py` with shared functions: `setup_optimizer`, `setup_scheduler`, `save_checkpoint`, `load_checkpoint`, `log_to_wandb`, `upload_to_hub`
    - Ensure all three training scripts use identical optimizer config (AdamW, lr=1e-5, cosine)
    - Ensure all three scripts use identical checkpoint saving (every 250 steps to Drive)
    - Ensure all three scripts use identical W&B logging (loss, lr, samples every 250 steps)
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [x] 7.2 Write integration tests for shared infrastructure
    - Test checkpoint save and load produces identical model state
    - Test optimizer configuration matches spec (lr=1e-5, betas, weight_decay)
    - Test cosine LR schedule decreases over time
    - _Validates: Requirements 10.1, 10.2, 10.3_

- [x] 8. Final Integration
  - [x] 8.1 End-to-end integration test
    - Test: initialize ControlNet → run one training step → verify loss is finite and gradients flow correctly
    - Test: load trained ControlNet → run inference pipeline → verify output image is valid
    - Test: save checkpoint → reload → verify model produces identical output
    - _Validates: Requirements 2.6, 3.1, 5.9, 12.1, 12.4_

  - [x] 8.2 Verify T4 memory budget
    - Document expected memory usage: models (FP16) ~2.1GB + optimizer ~4.2GB + gradients ~2.1GB + activations ~3GB + buffer ~1.6GB = ~13GB
    - Add memory profiling code that prints peak GPU memory after first training step
    - Verify total fits within 15GB T4 VRAM
    - _Validates: Requirements 9.4_

## Notes

- All three training scripts share identical infrastructure — they differ ONLY in `condition_type` and HuggingFace repo name
- The ControlNet class handles the full forward pass including UNet integration, so training scripts just call `controlnet(noisy_latent, timestep, text_emb, condition)`
- Zero convolutions are the key insight: they ensure training starts from a stable SD1.5 baseline
- Property tests use synthetic/random data and don't require downloading SD1.5 weights (use mock UNet fixtures)

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1"] },
    { "id": 1, "tasks": ["1.2", "1.3"] },
    { "id": 2, "tasks": ["1.4", "1.5", "2.2"] },
    { "id": 3, "tasks": ["3.1", "3.2", "7.1"] },
    { "id": 4, "tasks": ["3.3", "3.4", "4.1"] },
    { "id": 5, "tasks": ["4.2", "4.3"] },
    { "id": 6, "tasks": ["4.4", "5.1", "6.1"] },
    { "id": 7, "tasks": ["7.2", "8.1", "8.2"] }
  ]
}
```
