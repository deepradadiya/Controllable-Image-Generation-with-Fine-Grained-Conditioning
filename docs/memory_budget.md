# T4 GPU Memory Budget for ControlNet Training

## Overview

This document details the expected VRAM usage during ControlNet training on a
Google Colab T4 GPU (15GB VRAM). The architecture is specifically designed to
fit within this constraint through three key optimizations:

1. **FP16 Mixed Precision** — halves activation and model memory
2. **No UNet Backpropagation** — the frozen UNet stores no gradients
3. **Gradient Checkpointing** — trades compute for memory on activations

## Memory Breakdown

The ControlNet architecture has two memory profiles depending on whether the
copied encoder blocks are trainable or frozen. Our implementation freezes the
copied encoder blocks (only condition_embedding + zero_convs are trainable),
which is more memory-efficient.

### Actual Implementation (Frozen Encoder Copy)

| Component | Size (GB) | Notes |
|-----------|-----------|-------|
| SD1.5 UNet (FP16, frozen) | ~1.7 | Inference only, no gradient storage needed |
| SD1.5 VAE (FP16, frozen) | ~0.2 | Encode/decode only, no gradients |
| CLIP Text Encoder (FP16, frozen) | ~0.2 | Text encoding only, no gradients |
| ControlNet Adapter (FP16, ~361M total) | ~0.7 | Includes frozen encoder copy + trainable layers |
| Optimizer States (FP32) | ~0.1 | Adam: 2 states for ~12M trainable params |
| Gradients (FP16) | ~0.02 | Only for trainable params (~12M) |
| Activations (FP16) | ~3.0 | Forward pass intermediate values for backprop |
| CUDA Overhead / Buffer | ~1.6 | Memory allocator fragmentation, CUDA context |
| **Total** | **~7.5 GB** | **Well within 15GB T4 VRAM** |

### Worst-Case Scenario (All ~360M Adapter Params Trainable)

If the full ControlNet adapter were trainable (as in the original paper):

| Component | Size (GB) | Notes |
|-----------|-----------|-------|
| SD1.5 UNet (FP16, frozen) | ~1.7 | Inference only, no gradient storage needed |
| SD1.5 VAE (FP16, frozen) | ~0.2 | Encode/decode only, no gradients |
| CLIP Text Encoder (FP16, frozen) | ~0.2 | Text encoding only, no gradients |
| ControlNet Adapter (FP16) | ~0.7 | ~360M params stored in FP16 (2 bytes/param) |
| Optimizer States (FP32) | ~2.8 | Adam maintains 2 states per param in FP32 |
| Gradients (FP16) | ~0.7 | Only computed for ControlNet params (~360M × 2 bytes) |
| Activations (FP16) | ~3.0 | Forward pass intermediate values for backprop |
| CUDA Overhead / Buffer | ~1.6 | Memory allocator fragmentation, CUDA context |
| **Total** | **~11 GB** | **Still within 15GB T4 VRAM** |

Both configurations fit comfortably within the T4's 15 GB VRAM budget.

## Detailed Calculations

### Model Parameters

| Model | Parameters | FP32 Size | FP16 Size |
|-------|-----------|-----------|-----------|
| SD1.5 UNet | ~860M | 3.4 GB | 1.7 GB |
| SD1.5 VAE | ~84M | 0.34 GB | 0.17 GB |
| CLIP Text Encoder | ~123M | 0.49 GB | 0.25 GB |
| ControlNet Adapter (total) | ~361M | 1.4 GB | 0.7 GB |
| — Trainable (condition_embedding + zero_convs) | ~12M | 0.05 GB | 0.02 GB |
| — Frozen (copied encoder blocks) | ~349M | 1.4 GB | 0.65 GB |

### Optimizer Memory (Adam/AdamW)

Adam maintains two state tensors per trainable parameter:
- First moment (mean): FP32, same size as parameter
- Second moment (variance): FP32, same size as parameter

With our implementation (~12M trainable params):
- ~12M params × 4 bytes/param × 2 states = **~0.09 GB**

Worst-case (all ~360M trainable):
- ~360M params × 4 bytes/param × 2 states = **~2.8 GB**

### Gradient Memory

Gradients are only computed for trainable parameters:
- Actual: ~12M params × 2 bytes (FP16) = **~0.02 GB**
- Worst-case: ~360M params × 2 bytes (FP16) = **~0.7 GB**

### Activation Memory

Forward pass stores intermediate activations for backpropagation.
With FP16 and batch_size=1 at 512×512 resolution:
- Encoder activations at multiple scales (64×64, 32×32, 16×16, 8×8)
- Estimated ~3 GB for the full forward/backward pass

### CUDA Overhead

- CUDA context initialization: ~0.5 GB
- Memory allocator fragmentation: ~0.5 GB
- Temporary buffers (convolution workspace, etc.): ~0.6 GB
- Total overhead: **~1.6 GB**

## Why This Fits on T4 (15GB VRAM)

Even in the worst-case scenario (all adapter params trainable), the total
estimated peak memory is **~11 GB**, leaving a comfortable **~4 GB** margin
below the T4's 15 GB limit. Our actual implementation uses only **~7.5 GB**,
leaving **~7.5 GB** of headroom. This margin accounts for:

- Occasional memory spikes during gradient accumulation
- W&B logging overhead
- Validation sample generation (inference mode, lower memory)
- Python/PyTorch runtime overhead

### Key Optimizations That Make This Possible

1. **Frozen UNet (no backprop)**: The SD1.5 UNet (~860M params) is completely
   frozen. We never compute or store gradients for it, saving ~3.4 GB that
   would otherwise be needed for UNet gradients + optimizer states.

2. **FP16 Mixed Precision**: Using `torch.autocast("cuda", dtype=torch.float16)`
   stores activations in half precision, cutting activation memory roughly in
   half compared to FP32 training.

3. **Gradient Checkpointing**: When enabled, intermediate activations are
   recomputed during the backward pass instead of stored, trading ~20% more
   compute time for significantly reduced activation memory.

4. **Batch Size 1 + Gradient Accumulation**: Using batch_size=1 with 8-step
   gradient accumulation keeps per-step memory minimal while achieving an
   effective batch size of 8.

## Memory Profiling

The training script (`training/train_depth.py`) includes built-in memory
profiling that prints peak GPU memory after the first training step:

```python
# Print peak GPU memory on first step (Requirement 9.5)
if step == 0 and torch.cuda.is_available():
    peak_mem = torch.cuda.max_memory_allocated() / (1024**2)
    logger.info(f"Peak GPU memory after first step: {peak_mem:.1f} MB")
```

This allows immediate verification that the training fits within the T4's
15 GB VRAM budget on the actual hardware.

## Comparison with Full Fine-Tuning

| Approach | Trainable Params | Estimated VRAM | Fits T4? |
|----------|-----------------|----------------|----------|
| Full UNet fine-tuning | ~860M | ~28 GB | ❌ No |
| LoRA (rank 64) | ~67M | ~8 GB | ✅ Yes |
| **ControlNet (ours)** | **~360M** | **~11 GB** | **✅ Yes** |
| Full UNet + ControlNet | ~1220M | ~35 GB | ❌ No |

ControlNet achieves a good balance: enough trainable parameters for strong
spatial conditioning, while staying well within T4 memory limits by never
backpropagating through the frozen UNet.

## Verification Script

Run `scripts/estimate_memory.py` to calculate the theoretical memory budget
based on actual model parameter counts without requiring GPU hardware or
pretrained weight downloads.
