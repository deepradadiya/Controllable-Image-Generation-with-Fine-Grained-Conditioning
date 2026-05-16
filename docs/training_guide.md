# ControlNet Training Guide

A step-by-step tutorial for training ControlNet models with depth, pose, and edge conditioning on Stable Diffusion 1.5. This guide is designed for beginners and walks you through the entire process from environment setup to a fully trained model.

## What You'll Build

By the end of this guide, you'll have a trained ControlNet model that can generate images conditioned on spatial control maps (depth, pose, or edge). The model takes a condition map and a text prompt as input, and produces an image that follows both the spatial structure and the text description.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Training a ControlNet Model](#training-a-controlnet-model)
5. [Monitoring Training with W&B](#monitoring-training-with-wb)
6. [Checkpoint Management](#checkpoint-management)
7. [Common Issues and Solutions](#common-issues-and-solutions)
8. [What's Next](#whats-next)

---

## Prerequisites

### Hardware Requirements

| Environment | GPU | VRAM | Notes |
|-------------|-----|------|-------|
| Google Colab (Free) | T4 | 15 GB | Recommended starting point |
| Google Colab Pro | A100 | 40 GB | Faster training, larger batches |
| Local | RTX 3060+ | 12+ GB | Requires CUDA 11.7+ |

### Software Requirements

- Python 3.9+
- PyTorch 2.0+ with CUDA support
- Git (for cloning the repository)

### Accounts

- [HuggingFace](https://huggingface.co/) account (for downloading SD1.5 and uploading models)
- [Weights & Biases](https://wandb.ai/) account (optional, for experiment tracking)

---

## Environment Setup

### Option A: Google Colab (Recommended for Beginners)

1. Open a new Colab notebook and select a T4 GPU runtime:
   - Runtime → Change runtime type → GPU → T4

2. Clone the repository and install dependencies:

```python
!git clone https://github.com/your-username/Controllable-Image-Generation-with-Fine-Grained-Conditioning.git
%cd Controllable-Image-Generation-with-Fine-Grained-Conditioning

!pip install -e .
!pip install -r requirements.txt
```

3. Authenticate with HuggingFace:

```python
from huggingface_hub import login
login(token="your_hf_token")
```

4. (Optional) Mount Google Drive for persistent storage:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Option B: Local Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/Controllable-Image-Generation-with-Fine-Grained-Conditioning.git
cd Controllable-Image-Generation-with-Fine-Grained-Conditioning
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

3. Install dependencies:

```bash
pip install -e .
pip install -r requirements.txt
```

4. Verify GPU availability:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

## Dataset Preparation

### Step 1: Download COCO Dataset

The pipeline uses a subset of COCO 2017 downloaded from HuggingFace datasets:

```python
from src.data.dataset_processor import COCODatasetProcessor

processor = COCODatasetProcessor(
    cache_dir="./data/cache",
    subset_size=10000,  # Start with 10k images
)

dataset = processor.download_and_process(split="train", streaming=False)
print(f"Downloaded {len(dataset)} samples")
```

For Colab with limited storage, use a smaller subset:

```python
processor = COCODatasetProcessor(
    cache_dir="/content/drive/MyDrive/ControlNet/data",
    subset_size=5000,
)
```

### Step 2: Extract Condition Maps

#### Depth Maps

```python
from src.data.extract_depth import DepthExtractor, DepthExtractionConfig

config = DepthExtractionConfig(
    device="auto",
    precision="fp16",
    target_size=(512, 512),
)
extractor = DepthExtractor(config)

# Process a single image
from PIL import Image
image = Image.open("path/to/image.jpg")
result = extractor.extract(image)

if result.success:
    depth_map = result.depth_map  # Shape: (H, W, 1), range [0, 1]
```

#### Pose Skeletons

```python
from src.data.extract_pose import PoseExtractor

extractor = PoseExtractor(
    prefer_dwpose=True,
    fallback_to_mediapipe=True,
    speed_critical=False,
)

pose_map = extractor.extract(image)  # Shape: (H, W, 3), RGB skeleton
```

#### Edge Maps

```python
from src.data.extract_edges import CannyEdgeExtractor, EdgeExtractionConfig

config = EdgeExtractionConfig(
    adaptive_threshold=True,
    output_channels=3,
    normalize_output=True,
)
extractor = CannyEdgeExtractor(config)

result = extractor.extract(image)
if result.success:
    edge_map = result.edge_map  # Shape: (H, W, 3), range [0, 1]
```

### Step 3: Batch Processing

Process the entire dataset at once:

```python
from pathlib import Path
import numpy as np

output_dir = Path("./data/condition_maps/depth")
output_dir.mkdir(parents=True, exist_ok=True)

for idx, sample in enumerate(dataset):
    image = sample['image']
    result = extractor.extract(image)
    
    if result.success:
        np.save(output_dir / f"{idx:06d}.npy", result.depth_map)
    
    if idx % 100 == 0:
        print(f"Processed {idx}/{len(dataset)} images")
```

### Step 4: Verify Dataset

```python
from src.data.verify_dataset import DatasetVerifier

verifier = DatasetVerifier(data_dir="./data")
report = verifier.validate_all()

print(f"Valid samples: {report.valid_count}/{report.total_count}")
print(f"Success rate: {report.success_rate:.1%}")
print(f"Common failures: {report.failure_modes}")
```

---

## Training a ControlNet Model

### Training Depth ControlNet

The simplest way to start training:

```bash
python src/training/train_depth.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --output_dir="./models/trained/controlnet-depth" \
    --train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing \
    --learning_rate=1e-5 \
    --lr_scheduler="constant_with_warmup" \
    --lr_warmup_steps=500 \
    --num_train_epochs=100 \
    --mixed_precision="fp16" \
    --checkpointing_steps=500 \
    --validation_steps=100 \
    --report_to="wandb" \
    --seed=42
```

### Training Pose ControlNet

```bash
python src/training/train_pose.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --output_dir="./models/trained/controlnet-pose" \
    --train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing \
    --learning_rate=1e-5 \
    --lr_scheduler="constant_with_warmup" \
    --lr_warmup_steps=500 \
    --num_train_epochs=100 \
    --mixed_precision="fp16" \
    --checkpointing_steps=500 \
    --report_to="wandb" \
    --seed=42
```

### Training Edge ControlNet

```bash
python src/training/train_edge.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --output_dir="./models/trained/controlnet-edge" \
    --train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing \
    --learning_rate=1e-5 \
    --lr_scheduler="constant_with_warmup" \
    --lr_warmup_steps=500 \
    --num_train_epochs=100 \
    --mixed_precision="fp16" \
    --checkpointing_steps=500 \
    --report_to="wandb" \
    --seed=42
```

### Training on Colab (Memory-Optimized)

For T4 GPU with 15 GB VRAM, use these settings:

```bash
python src/training/train_depth.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --output_dir="/content/drive/MyDrive/ControlNet/models/depth" \
    --train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing \
    --learning_rate=1e-5 \
    --mixed_precision="fp16" \
    --enable_xformers_memory_efficient_attention \
    --checkpointing_steps=500 \
    --max_train_steps=10000 \
    --report_to="wandb" \
    --seed=42
```

### Resuming Training

If your Colab session disconnects or you want to continue training:

```bash
python src/training/train_depth.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --output_dir="./models/trained/controlnet-depth" \
    --resume_from_checkpoint="latest" \
    --train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing \
    --learning_rate=1e-5 \
    --mixed_precision="fp16" \
    --report_to="wandb"
```

You can also specify a specific checkpoint:

```bash
--resume_from_checkpoint="./models/trained/controlnet-depth/checkpoint-5000"
```

---

## Monitoring Training with W&B

### Setup

1. Install and login to Weights & Biases:

```bash
pip install wandb
wandb login
```

2. Training scripts automatically log to W&B when `--report_to="wandb"` is set.

### What Gets Logged

- **Loss curves**: Training loss over steps/epochs
- **Learning rate**: Current learning rate schedule
- **GPU memory**: VRAM usage over time
- **Validation images**: Generated samples at regular intervals
- **Gradient norms**: For detecting training instability
- **Training speed**: Steps per second, samples per second

### Viewing Results

Visit [wandb.ai](https://wandb.ai) and navigate to your project (default: `controlnet-training`).

Key metrics to watch:
- **train/loss** should decrease steadily
- **train/grad_norm** should stay below 1.0 (with gradient clipping)
- **GPU memory** should stay below 13 GB on T4

### Custom Logging

```python
from src.utils.visualize import TrainingVisualizer

visualizer = TrainingVisualizer(log_dir="./logs")
visualizer.log_training_step(step=1000, loss=0.05, lr=1e-5)
visualizer.log_validation_images(step=1000, images=generated_images)
```

---

## Checkpoint Management

### Automatic Checkpointing

Checkpoints are saved automatically based on `--checkpointing_steps`. Each checkpoint contains:

```
checkpoint-5000/
├── controlnet/           # ControlNet weights
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── optimizer.bin         # Optimizer state
├── scheduler.bin         # LR scheduler state
├── random_states.pkl     # Random number generator states
└── training_state.json   # Step count, epoch, etc.
```

### Limiting Checkpoint Storage

To avoid running out of disk space:

```bash
--checkpoints_total_limit=3  # Keep only the 3 most recent checkpoints
```

### Saving to Google Drive (Colab)

```python
from src.utils.colab_helpers import ColabHelper

helper = ColabHelper(
    drive_project_path="MyDrive/ControlNet",
    checkpoint_interval_minutes=30,
)

# Automatic periodic saving during training
helper.start_checkpoint_timer(model_dir="./models/trained/controlnet-depth")
```

### Loading a Checkpoint for Inference

```python
from src.models.controlnet import ControlNetModel

model = ControlNetModel.from_pretrained(
    "./models/trained/controlnet-depth/checkpoint-5000/controlnet"
)
```

---

## Common Issues and Solutions

### Out of Memory (OOM) Errors

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions** (try in order):
1. Enable gradient checkpointing: `--gradient_checkpointing`
2. Use mixed precision: `--mixed_precision="fp16"`
3. Enable xformers: `--enable_xformers_memory_efficient_attention`
4. Reduce batch size to 1: `--train_batch_size=1`
5. Increase gradient accumulation: `--gradient_accumulation_steps=16`

### Training Loss Not Decreasing

**Possible causes**:
- Learning rate too high or too low
- Dataset quality issues
- Incorrect condition map format

**Solutions**:
1. Try learning rate 1e-5 (default, works well for most cases)
2. Verify condition maps are normalized to [0, 1] range
3. Check that image-condition pairs are correctly aligned
4. Run dataset verification: `python -m src.data.verify_dataset`

### NaN Loss Values

**Possible causes**:
- Learning rate too high
- Corrupted training samples
- Numerical instability in FP16

**Solutions**:
1. Reduce learning rate: `--learning_rate=5e-6`
2. Enable gradient clipping: `--max_grad_norm=1.0`
3. Try BF16 instead of FP16: `--mixed_precision="bf16"` (requires Ampere GPU)
4. Validate dataset for corrupted samples

### Colab Session Disconnection

**Prevention**:
- Save checkpoints frequently: `--checkpointing_steps=500`
- Use Google Drive for output: `--output_dir="/content/drive/MyDrive/ControlNet/models"`
- Enable session timer warnings in Colab helpers

**Recovery**:
- Use `--resume_from_checkpoint="latest"` to continue from the last saved checkpoint

### Slow Training Speed

**Optimizations**:
1. Enable TF32 (Ampere GPUs): `--allow_tf32`
2. Use xformers attention: `--enable_xformers_memory_efficient_attention`
3. Reduce validation frequency: `--validation_steps=500`
4. Use `--dataloader_num_workers=2` for faster data loading

### Model Produces Blurry or Low-Quality Images

**Possible causes**:
- Insufficient training (too few steps)
- Conditioning strength too low or too high
- Poor quality condition maps

**Solutions**:
1. Train for at least 5000-10000 steps before evaluating
2. Adjust conditioning scale during inference (try 0.5 to 1.5)
3. Verify condition map quality with visual inspection
4. Increase guidance scale during inference (7.5 to 12.0)

---

## What's Next

After training your model, you have several paths forward:

1. **Evaluate quality**: Use `src/evaluation/compute_fid.py` to measure FID scores and `src/evaluation/condition_alignment.py` to check how well your model follows condition maps.

2. **Deploy as a demo**: Follow the [Deployment Guide](deployment_guide.md) to create an interactive HuggingFace Space where anyone can try your model.

3. **Tune hyperparameters**: If results aren't satisfactory, consult the [Hyperparameter Guide](hyperparameter_guide.md) for optimization strategies.

4. **Train other condition types**: Once you've trained one model (e.g., depth), try training pose or edge models using the same workflow with different training scripts.

---

## Appendix: Understanding the Training Process

### How ControlNet Training Works

ControlNet training is a form of fine-tuning where:

1. **The base SD1.5 UNet weights are frozen** — they don't change during training
2. **Only the ControlNet adapter learns** — it learns to inject spatial information into the generation process
3. **Zero convolutions ensure stability** — the ControlNet starts by producing zero outputs, so it doesn't disrupt the pre-trained model initially

### Training Loop Overview

Each training step:
1. Load an image, its text caption, and the corresponding condition map
2. Add random noise to the image (simulating a diffusion timestep)
3. Pass the noisy image + text through the frozen UNet
4. Pass the condition map through the ControlNet to produce spatial features
5. Combine ControlNet features with UNet features
6. Predict the noise that was added
7. Compute the loss (difference between predicted and actual noise)
8. Update only the ControlNet weights

### Key Concepts for Beginners

- **Gradient checkpointing**: Saves GPU memory by recomputing intermediate values during backpropagation instead of storing them. Makes training ~15% slower but uses ~30% less memory.
- **Mixed precision (FP16)**: Uses 16-bit floating point numbers instead of 32-bit. Halves memory usage with minimal quality impact.
- **Gradient accumulation**: Simulates larger batch sizes by accumulating gradients over multiple forward passes before updating weights. Essential when GPU memory limits batch size to 1.
- **Conditioning scale**: Controls how strongly the condition map influences the generated image. Higher values = stricter adherence to the spatial structure.
