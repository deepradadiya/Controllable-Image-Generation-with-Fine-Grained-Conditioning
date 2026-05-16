# Hyperparameter Tuning Guide

Recommendations and best practices for tuning ControlNet training hyperparameters to achieve optimal results within GPU memory constraints.

## Table of Contents

1. [Learning Rate Selection](#learning-rate-selection)
2. [Batch Size and Gradient Accumulation](#batch-size-and-gradient-accumulation)
3. [Conditioning Strength Tuning](#conditioning-strength-tuning)
4. [Training Duration](#training-duration)
5. [Memory Optimization Strategies](#memory-optimization-strategies)
6. [Condition-Specific Recommendations](#condition-specific-recommendations)
7. [Quick Reference Table](#quick-reference-table)

---

## Learning Rate Selection

### Recommended Values

| Scenario | Learning Rate | Notes |
|----------|--------------|-------|
| Default (recommended) | 1e-5 | Works well for most ControlNet training |
| Conservative start | 5e-6 | Use if training is unstable |
| Aggressive (faster convergence) | 2e-5 | Risk of instability, monitor closely |
| Fine-tuning existing ControlNet | 5e-6 | Lower LR for already-trained models |

### Why 1e-5?

ControlNet training is a fine-tuning task. The base SD1.5 UNet weights are frozen, and only the ControlNet adapter learns. A learning rate of 1e-5 is approximately 10x lower than typical vision model training, which prevents:
- Disrupting the zero-initialized convolution layers too quickly
- Causing training instability in the early steps
- Producing noisy gradients that lead to divergence

### Learning Rate Scheduling

```bash
# Recommended: Constant with warmup
--lr_scheduler="constant_with_warmup" \
--lr_warmup_steps=500

# Alternative: Cosine annealing (for longer training runs)
--lr_scheduler="cosine" \
--lr_warmup_steps=1000 \
--lr_num_cycles=1
```

**Warmup** is important for ControlNet because:
- Zero-initialized layers produce zero gradients initially
- Warmup allows the model to gradually learn meaningful features
- Prevents large parameter updates in early steps

**Schedule comparison**:

| Schedule | Best For | Behavior |
|----------|----------|----------|
| `constant_with_warmup` | Short runs (<10k steps) | Stable, predictable |
| `cosine` | Long runs (>10k steps) | Gradual decay, good final quality |
| `linear` | Medium runs | Steady decrease |
| `constant` | Debugging/testing | No decay, simple |

### Diagnosing Learning Rate Issues

**Learning rate too high**:
- Loss spikes or becomes NaN
- Generated images are noisy or distorted
- Gradient norms exceed 10.0 regularly

**Learning rate too low**:
- Loss decreases very slowly
- Model produces images identical to unconditioned SD1.5
- No visible conditioning effect after 5000+ steps

---

## Batch Size and Gradient Accumulation

### Understanding the Tradeoff

On a T4 GPU (15 GB VRAM), the physical batch size is limited to 1. Gradient accumulation simulates larger batches:

```
Effective batch size = physical_batch_size × gradient_accumulation_steps
```

### Recommended Configurations

| GPU | Physical Batch | Grad Accum Steps | Effective Batch |
|-----|---------------|------------------|-----------------|
| T4 (15 GB) | 1 | 8 | 8 |
| RTX 3060 (12 GB) | 1 | 8 | 8 |
| RTX 3080 (10 GB) | 1 | 8 | 8 |
| RTX 3090 (24 GB) | 2 | 4 | 8 |
| A100 (40 GB) | 4 | 2 | 8 |

### Why Effective Batch Size 8?

An effective batch size of 8 provides:
- Stable gradient estimates (reduces noise from single-sample batches)
- Reasonable training speed (not too many accumulation steps)
- Good convergence behavior for diffusion model fine-tuning

### Scaling the Learning Rate

When changing effective batch size, scale the learning rate proportionally:

```python
# Base: lr=1e-5, effective_batch=8
# If you increase to effective_batch=16:
scaled_lr = 1e-5 * (16 / 8)  # = 2e-5
```

Use the `--scale_lr` flag to do this automatically:

```bash
--scale_lr \
--train_batch_size=1 \
--gradient_accumulation_steps=16 \
--learning_rate=1e-5  # Will be scaled to 2e-5
```

### Impact on Training Speed

| Grad Accum Steps | Steps/Second (T4) | Time per Effective Batch |
|------------------|-------------------|--------------------------|
| 4 | ~1.2 | ~3.3s |
| 8 | ~1.2 | ~6.7s |
| 16 | ~1.2 | ~13.3s |

Higher accumulation steps don't slow down individual forward/backward passes, but each optimizer update takes more wall-clock time.

---

## Conditioning Strength Tuning

### Training-Time Conditioning Scale

During training, the conditioning scale controls how strongly ControlNet features influence the UNet:

```bash
--controlnet_conditioning_scale=1.0  # Default
```

| Scale | Effect |
|-------|--------|
| 0.5 | Weak conditioning, more creative freedom |
| 1.0 | Balanced (recommended for training) |
| 1.5 | Strong conditioning, strict adherence to condition map |

**Recommendation**: Train with scale 1.0, then adjust at inference time.

### Inference-Time Conditioning Scale

At inference, you can adjust conditioning strength without retraining:

```python
from src.inference.pipeline import ControlNetInferencePipeline, GenerationParams

params = GenerationParams(
    conditioning_scale=1.0,  # Adjust this
    guidance_scale=7.5,
    num_inference_steps=50,
)
```

**Tuning guide**:
- Start at 1.0 and evaluate results
- If images don't follow the condition map: increase to 1.2-1.5
- If images look over-constrained or artifacts appear: decrease to 0.7-0.9
- For artistic/creative results: use 0.3-0.5

### Classifier-Free Guidance Dropout

During training, randomly dropping the conditioning teaches the model to work with varying conditioning strengths:

```bash
--proportion_empty_prompts=0.1  # Drop 10% of prompts
```

This enables classifier-free guidance at inference time. Higher dropout (0.1-0.2) improves guidance quality but slightly slows convergence.

---

## Training Duration

### How Long to Train

| Dataset Size | Recommended Steps | Approximate Time (T4) |
|-------------|-------------------|------------------------|
| 1,000 images | 5,000 steps | ~2 hours |
| 5,000 images | 10,000 steps | ~4 hours |
| 10,000 images | 20,000 steps | ~8 hours |
| 50,000 images | 50,000 steps | ~20 hours |

### Epochs vs Steps

For ControlNet training, steps are more meaningful than epochs because:
- Dataset sizes vary significantly
- The model sees each sample multiple times with different noise levels
- Convergence depends on total gradient updates, not dataset passes

```bash
# Prefer max_train_steps over num_train_epochs
--max_train_steps=10000
```

### Early Stopping Indicators

Stop training when:
1. **Loss plateaus**: No improvement for 2000+ steps
2. **Visual quality saturates**: Validation images stop improving
3. **Overfitting**: Training loss decreases but validation quality degrades

### Checkpointing Strategy

```bash
--checkpointing_steps=1000 \      # Save every 1000 steps
--checkpoints_total_limit=3 \     # Keep only 3 most recent
--validation_steps=500            # Generate samples every 500 steps
```

Evaluate checkpoints at different stages to find the sweet spot between underfitting and overfitting.

---

## Memory Optimization Strategies

### Priority Order

Apply these optimizations in order until training fits in memory:

| Priority | Optimization | Memory Savings | Speed Impact |
|----------|-------------|----------------|--------------|
| 1 | Mixed precision (FP16) | ~50% | Slight speedup |
| 2 | Gradient checkpointing | ~30% | 10-20% slower |
| 3 | xformers attention | ~20% | Slight speedup |
| 4 | 8-bit Adam optimizer | ~25% optimizer states | Minimal |
| 5 | CPU offloading | Variable | Significant slowdown |

### Mixed Precision Training

```bash
--mixed_precision="fp16"
```

FP16 reduces memory for:
- Model weights (stored in FP16)
- Activations (computed in FP16)
- Gradients (accumulated in FP32 for stability)

**When to use BF16 instead**:
- Available on Ampere GPUs (A100, RTX 30xx/40xx)
- Better numerical stability than FP16
- No loss scaling needed

```bash
--mixed_precision="bf16"  # Requires Ampere GPU
```

### Gradient Checkpointing

```bash
--gradient_checkpointing
```

Trades computation for memory by recomputing activations during backward pass instead of storing them. Essential for T4 GPU training.

**Memory budget with gradient checkpointing on T4**:
- Models (FP16): ~2.1 GB
- Optimizer states: ~4.2 GB
- Gradients: ~2.1 GB
- Activations (batch=1): ~3.0 GB
- Buffer/overhead: ~1.6 GB
- **Total**: ~13 GB (fits in T4's 15 GB)

### xformers Memory-Efficient Attention

```bash
--enable_xformers_memory_efficient_attention
```

Replaces standard attention with memory-efficient implementation. Reduces memory usage and often improves speed.

**Installation**:
```bash
pip install xformers
```

### 8-bit Adam Optimizer

```bash
--use_8bit_adam
```

Reduces optimizer state memory by quantizing Adam's first and second moments to 8-bit. Requires `bitsandbytes`:

```bash
pip install bitsandbytes
```

### Memory Monitoring

Monitor GPU memory during training:

```python
import torch

# Check current usage
allocated = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.memory_reserved() / 1e9
print(f"Allocated: {allocated:.1f} GB, Reserved: {reserved:.1f} GB")

# Peak usage
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak: {peak:.1f} GB")
```

---

## Condition-Specific Recommendations

### Depth Conditioning

Depth maps provide strong structural guidance. Training tends to converge faster.

```bash
# Depth-specific settings
--learning_rate=1e-5 \
--max_train_steps=10000 \
--controlnet_conditioning_scale=1.0
```

**Tips**:
- Depth maps are single-channel (grayscale) — set `conditioning_channels=1`
- DPT-Large produces high-quality depth maps but is memory-intensive
- Normalize depth maps to [0, 1] range before training
- Depth conditioning works well even with fewer training samples

### Pose Conditioning

Pose conditioning is more challenging because skeletons are sparse signals.

```bash
# Pose-specific settings
--learning_rate=1e-5 \
--max_train_steps=15000 \
--controlnet_conditioning_scale=1.0 \
--gradient_accumulation_steps=8
```

**Tips**:
- Pose maps are 3-channel (RGB skeleton rendering) — set `conditioning_channels=3`
- Requires more training steps than depth (sparse signal is harder to learn)
- Filter dataset to images containing people for better results
- Consider using DWPose over MediaPipe for higher quality keypoints
- Increase conditioning scale to 1.2-1.5 at inference for stronger pose adherence

### Edge Conditioning

Edge maps provide outline-level control. Training is straightforward but sensitive to edge quality.

```bash
# Edge-specific settings
--learning_rate=1e-5 \
--max_train_steps=10000 \
--controlnet_conditioning_scale=1.0
```

**Tips**:
- Edge maps can be 1-channel or 3-channel depending on configuration
- Adaptive thresholding produces better edges than fixed thresholds
- Canny edge parameters (low/high threshold) significantly affect training quality
- Recommended Canny thresholds: low=100, high=200 (adjust per dataset)
- Edge conditioning is sensitive to line thickness — keep consistent

---

## Quick Reference Table

### Recommended Starting Configuration (T4 GPU)

```bash
python src/training/train_depth.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --output_dir="./models/trained/controlnet-depth" \
    --train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --learning_rate=1e-5 \
    --lr_scheduler="constant_with_warmup" \
    --lr_warmup_steps=500 \
    --max_train_steps=10000 \
    --mixed_precision="fp16" \
    --max_grad_norm=1.0 \
    --checkpointing_steps=1000 \
    --checkpoints_total_limit=3 \
    --validation_steps=500 \
    --report_to="wandb" \
    --seed=42
```

### Hyperparameter Summary

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| Learning rate | 1e-5 | 5e-6 to 2e-5 | Lower is safer |
| LR warmup steps | 500 | 100-1000 | Longer for larger datasets |
| Batch size (physical) | 1 | 1-4 | Limited by GPU memory |
| Gradient accumulation | 8 | 4-16 | Higher = more stable |
| Max gradient norm | 1.0 | 0.5-2.0 | Prevents exploding gradients |
| Conditioning scale | 1.0 | 0.5-1.5 | Adjust at inference |
| Guidance scale (inference) | 7.5 | 5.0-15.0 | Higher = more prompt adherence |
| Inference steps | 50 | 20-100 | More = higher quality |
| Mixed precision | fp16 | fp16/bf16/no | Always use fp16 on T4 |
| Conditioning dropout | 0.1 | 0.0-0.2 | Enables CFG at inference |

### Troubleshooting Quick Fixes

| Problem | First Try | Second Try |
|---------|-----------|------------|
| OOM error | Enable gradient checkpointing | Reduce grad accum to 4 |
| NaN loss | Reduce LR to 5e-6 | Enable gradient clipping |
| No convergence | Increase LR to 2e-5 | Check dataset quality |
| Slow training | Enable xformers | Reduce validation frequency |
| Poor quality | Train longer (2x steps) | Increase conditioning scale |
| Overfitting | Reduce training steps | Add conditioning dropout |

---

## Advanced: Experiment Tracking

### Systematic Hyperparameter Search

Use W&B Sweeps for automated hyperparameter search:

```python
# wandb_sweep.yaml
program: src/training/train_depth.py
method: bayes
metric:
  name: train/loss
  goal: minimize
parameters:
  learning_rate:
    min: 1e-6
    max: 5e-5
    distribution: log_uniform_values
  gradient_accumulation_steps:
    values: [4, 8, 16]
  lr_warmup_steps:
    values: [100, 500, 1000]
```

```bash
wandb sweep wandb_sweep.yaml
wandb agent your-entity/controlnet-training/sweep_id
```

### Comparing Runs

Key metrics to compare across experiments:
1. **Final training loss** — lower is generally better
2. **FID score** — measures generation quality (lower is better)
3. **Condition alignment** — measures how well images follow condition maps
4. **Training time** — practical consideration for iteration speed

---

## Next Steps

- See [Training Guide](training_guide.md) for step-by-step training instructions
- See [Deployment Guide](deployment_guide.md) for deploying trained models
- Run evaluation with `src/evaluation/compute_fid.py` to measure model quality

---

## Appendix: Best Practices Summary

### Do's

- **Start with defaults**: The recommended configuration works well for most cases. Only tune if you have a specific issue.
- **Monitor early**: Check W&B logs within the first 500 steps. If loss isn't decreasing, something is wrong.
- **Save frequently**: Checkpoints are cheap insurance against session disconnections and training failures.
- **Validate visually**: Numbers (FID, loss) are useful, but always look at generated images to assess quality.
- **Use version control**: Track your hyperparameter configurations alongside model checkpoints.

### Don'ts

- **Don't train too long**: More steps isn't always better. Overfitting degrades quality.
- **Don't skip warmup**: Zero-initialized layers need warmup to start learning properly.
- **Don't ignore gradient norms**: Consistently high gradient norms (>10) indicate instability.
- **Don't use large batch sizes on T4**: Physical batch size > 1 will likely OOM. Use gradient accumulation instead.
- **Don't change multiple hyperparameters at once**: Change one thing at a time so you can attribute improvements correctly.

### Recommended Workflow for New Projects

1. **Baseline run**: Use all default settings, train for 5000 steps
2. **Evaluate**: Check FID, condition alignment, and visual quality
3. **Identify bottleneck**: Is it quality? Speed? Memory?
4. **Targeted tuning**: Adjust the relevant hyperparameter(s)
5. **Compare**: Run evaluation again and compare to baseline
6. **Iterate**: Repeat steps 3-5 until satisfied
