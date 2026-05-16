# Controllable Image Generation with Fine-Grained Conditioning

A production-grade ControlNet training pipeline for Stable Diffusion 1.5 that enables spatial conditioning using depth maps, pose skeletons, and edge maps. Implements the architecture from ["Adding Conditional Control to Text-to-Image Diffusion Models" (Zhang et al., 2023)](https://arxiv.org/abs/2302.05543) and is optimized for Google Colab T4 GPU (15GB VRAM).

## Features

- **Three conditioning types**: Depth maps (DPT), pose skeletons (DWPose/MediaPipe), and Canny edge maps
- **Memory-efficient training**: Gradient checkpointing, mixed precision (FP16), and gradient accumulation fit within T4 GPU constraints
- **Complete pipeline**: Dataset processing, training, evaluation (FID + condition alignment), inference, and web demo
- **Colab-optimized**: Session management, Google Drive checkpointing, automatic OOM recovery
- **HuggingFace integration**: Model serialization, Hub upload, and Gradio Space deployment
- **Experiment tracking**: Weights & Biases integration for loss curves, sample generations, and metrics

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Dataset Processing](#dataset-processing)
  - [Condition Map Extraction](#condition-map-extraction)
  - [Training](#training)
  - [Inference](#inference)
  - [Evaluation](#evaluation)
  - [Web Demo](#web-demo)
- [Visual Examples](#visual-examples)
- [HuggingFace Space Deployment](#huggingface-space-deployment)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [References](#references)

## Installation

### Google Colab (Recommended)

```python
# Clone the repository
!git clone https://github.com/your-username/Controllable-Image-Generation-with-Fine-Grained-Conditioning.git
%cd Controllable-Image-Generation-with-Fine-Grained-Conditioning

# Install dependencies (optimized for Colab T4 GPU with CUDA 11.8)
!pip install -r requirements.txt

# Install the package in development mode
!pip install -e .

# Verify installation
from src import BaseConfig, print_environment_info
print_environment_info()
```

### Local Development

```bash
# Clone the repository
git clone https://github.com/your-username/Controllable-Image-Generation-with-Fine-Grained-Conditioning.git
cd Controllable-Image-Generation-with-Fine-Grained-Conditioning

# Create a virtual environment (Python 3.8-3.11)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Verify Installation

```python
from src import BaseConfig

config = BaseConfig()
print(f"Device: {config.device}")
print(f"Environment: Colab={config.is_colab}, Local={config.is_local}")
print(f"Memory config: {config.get_memory_config()}")
```

## Quick Start

### Generate an image with edge conditioning in 5 lines:

```python
from src.inference.pipeline import ControlNetInferencePipeline, InferenceConfig, GenerationParams
from PIL import Image

# Load a source image and set up the pipeline
config = InferenceConfig(condition_type="edge", enable_memory_optimization=True)
pipeline = ControlNetInferencePipeline(config)

# Generate
source_image = Image.open("your_image.png")
result = pipeline.generate(
    prompt="a beautiful painting in watercolor style",
    condition_image=source_image,
    params=GenerationParams(num_inference_steps=30, conditioning_scale=1.0)
)
result.images[0].save("output.png")
```

### Train a ControlNet model:

```python
from configs.base_config import get_colab_config
from src.training.trainer import ControlNetTrainer

config = get_colab_config()
trainer = ControlNetTrainer(
    condition_type="depth",
    batch_size=1,
    gradient_accumulation_steps=8,
    mixed_precision=True
)
trainer.train()
```

## Project Structure

```
├── configs/                        # Configuration management
│   ├── base_config.py              # Centralized hyperparameters and settings
│   └── example_config.yaml         # Example YAML configuration
├── src/                            # Source code
│   ├── data/                       # Dataset processing
│   │   ├── dataset_processor.py    # COCO dataset download and preprocessing
│   │   ├── extract_depth.py        # DPT depth map extraction
│   │   ├── extract_pose.py         # DWPose/MediaPipe pose extraction
│   │   ├── extract_edges.py        # Canny edge map extraction
│   │   └── verify_dataset.py       # Dataset validation and QA
│   ├── models/                     # Model architecture
│   │   ├── controlnet.py           # ControlNet implementation (zero convolutions)
│   │   ├── unet_wrapper.py         # SD1.5 UNet with ControlNet integration
│   │   └── config.py               # Model configuration and serialization
│   ├── training/                   # Training system
│   │   ├── trainer.py              # Memory-optimized training orchestrator
│   │   ├── losses.py               # Diffusion loss with conditioning
│   │   ├── train_depth.py          # Depth conditioning training
│   │   ├── train_pose.py           # Pose conditioning training
│   │   └── train_edge.py           # Edge conditioning training
│   ├── inference/                  # Inference pipeline
│   │   ├── pipeline.py             # End-to-end generation with DDIM sampling
│   │   ├── controls.py             # Conditioning strength controls
│   │   └── model_loader.py         # Model loading and compatibility checks
│   ├── evaluation/                 # Evaluation metrics
│   │   ├── compute_fid.py          # FID score computation
│   │   ├── condition_alignment.py  # Condition following metrics
│   │   └── visual_grid.py          # Side-by-side comparison grids
│   ├── utils/                      # Utilities
│   │   ├── colab_helpers.py        # Colab session management and Drive integration
│   │   ├── memory_utils.py         # GPU memory monitoring and optimization
│   │   └── visualize.py            # Training visualization and W&B integration
│   └── app/                        # Web demo
│       ├── gradio_app.py           # Gradio interface for HuggingFace Spaces
│       ├── controls.py             # Interactive generation controls
│       └── model_manager.py        # Model caching and lazy loading
├── examples/                       # Usage examples and demos
├── docs/                           # Additional documentation
├── data/                           # Dataset storage (auto-created)
├── models/                         # Model checkpoints (auto-created)
├── outputs/                        # Generated outputs
├── logs/                           # Training logs
├── requirements.txt                # Pinned dependencies for Colab T4
└── setup.py                        # Package installation
```

## Usage

### Dataset Processing

Download and preprocess the COCO 2017 dataset with condition map extraction:

```python
from src.data.dataset_processor import DatasetProcessor
from configs.base_config import BaseConfig

config = BaseConfig()
processor = DatasetProcessor(config)

# Download COCO subset (streaming for memory efficiency)
dataset_path = processor.download_coco_subset(subset_size=10000)

# Validate dataset integrity
report = processor.validate_dataset_integrity()
print(f"Valid samples: {report.valid_samples}")
print(f"Failed samples: {report.failed_samples}")
```

### Condition Map Extraction

#### Depth Maps (DPT)

```python
from src.data.extract_depth import DepthExtractor, DepthExtractionConfig
from PIL import Image

config = DepthExtractionConfig(
    device="auto",
    precision="fp16",
    target_size=(512, 512)
)
extractor = DepthExtractor(config)

image = Image.open("input.jpg")
result = extractor.extract(image)

if result.success:
    # result.depth_map: numpy array (H, W, 1), range [0, 1]
    depth_display = (result.depth_map[:, :, 0] * 255).astype("uint8")
    Image.fromarray(depth_display).save("depth_map.png")
```

#### Pose Skeletons (DWPose)

```python
from src.data.extract_pose import PoseExtractor
from PIL import Image

extractor = PoseExtractor(
    prefer_dwpose=True,
    fallback_to_mediapipe=True
)

image = Image.open("person.jpg")
pose_map = extractor.extract(image)
# pose_map: numpy array (H, W, 3) with rendered skeleton
Image.fromarray(pose_map).save("pose_skeleton.png")
```

#### Edge Maps (Canny)

```python
from src.data.extract_edges import CannyEdgeExtractor, EdgeExtractionConfig
from PIL import Image

config = EdgeExtractionConfig(
    adaptive_threshold=True,
    output_channels=3,
    normalize_output=True
)
extractor = CannyEdgeExtractor(config)

image = Image.open("input.jpg")
result = extractor.extract(image)

if result.success:
    # result.edge_map: numpy array (H, W, 3), range [0, 1]
    edge_display = (result.edge_map * 255).astype("uint8")
    Image.fromarray(edge_display).save("edges.png")
```

### Training

#### Depth-Conditioned ControlNet

```python
from src.training.train_depth import train_depth_controlnet
from configs.base_config import get_depth_config

config = get_depth_config()

# Adjust for your environment
config.training.num_train_epochs = 50
config.training.learning_rate = 1e-5
config.training.gradient_accumulation_steps = 8
config.dataset.subset_size = 5000

train_depth_controlnet(config)
```

#### Pose-Conditioned ControlNet

```python
from src.training.train_pose import train_pose_controlnet
from configs.base_config import get_pose_config

config = get_pose_config()
train_pose_controlnet(config)
```

#### Edge-Conditioned ControlNet

```python
from src.training.train_edge import train_edge_controlnet
from configs.base_config import get_edge_config

config = get_edge_config()
train_edge_controlnet(config)
```

#### Training with Colab Optimizations

```python
from configs.base_config import get_colab_config
from src.utils.colab_helpers import setup_drive_checkpointing

config = get_colab_config()

# Enable Google Drive checkpointing for session persistence
setup_drive_checkpointing(config)

# Training will automatically:
# - Save checkpoints every 30 minutes
# - Warn before 12-hour session timeout
# - Resume from latest checkpoint on reconnection
```

#### Monitoring with Weights & Biases

```python
import wandb

wandb.init(project="controlnet-training", name="depth-experiment-1")

# Training automatically logs:
# - Loss curves
# - Sample generations at regular intervals
# - GPU memory usage
# - Learning rate schedule
```

### Inference

#### Single Image Generation

```python
from src.inference.pipeline import (
    ControlNetInferencePipeline,
    InferenceConfig,
    GenerationParams
)
from PIL import Image

# Configure pipeline
config = InferenceConfig(
    pretrained_model_path="runwayml/stable-diffusion-v1-5",
    controlnet_model_path="./models/trained/controlnet_depth",
    condition_type="depth",
    enable_memory_optimization=True
)

pipeline = ControlNetInferencePipeline(config)

# Generate with a depth map
depth_map = Image.open("depth_map.png")
result = pipeline.generate(
    prompt="a serene mountain landscape at sunset, photorealistic",
    condition_image=depth_map,
    params=GenerationParams(
        num_inference_steps=50,
        guidance_scale=7.5,
        conditioning_scale=1.0,
        seed=42
    )
)

result.images[0].save("generated.png")
print(f"Generation time: {result.generation_time_seconds:.1f}s")
```

#### Batch Generation

```python
params = GenerationParams(
    num_inference_steps=30,
    guidance_scale=7.5,
    conditioning_scale=1.0,
    num_images=4  # Generate 4 variations
)

result = pipeline.generate(
    prompt="a cozy cabin in the woods",
    condition_image=edge_map,
    params=params
)

for i, img in enumerate(result.images):
    img.save(f"output_{i}.png")
```

#### Adjusting Conditioning Strength

```python
# Low strength: condition map provides loose guidance
result_loose = pipeline.generate(
    prompt="abstract art",
    condition_image=edge_map,
    params=GenerationParams(conditioning_scale=0.3)
)

# High strength: condition map is followed closely
result_strict = pipeline.generate(
    prompt="abstract art",
    condition_image=edge_map,
    params=GenerationParams(conditioning_scale=1.5)
)
```

### Evaluation

#### FID Score Computation

```python
from src.evaluation.compute_fid import FIDCalculator

calculator = FIDCalculator(batch_size=16)

fid_score = calculator.compute_fid_score(
    generated_images=generated_images,
    reference_images=reference_images
)
print(f"FID Score: {fid_score:.2f}")
```

#### Condition Alignment Metrics

```python
from src.evaluation.condition_alignment import ConditionAlignmentEvaluator

evaluator = ConditionAlignmentEvaluator()

alignment_score = evaluator.compute_condition_alignment(
    generated_images=generated_images,
    condition_maps=condition_maps
)
print(f"Condition Alignment: {alignment_score:.4f}")
```

#### Visual Comparison Grids

```python
from src.evaluation.visual_grid import VisualGridGenerator

grid_gen = VisualGridGenerator()
comparison_grid = grid_gen.generate_visual_comparison(
    condition_maps=condition_maps,
    generated_images=generated_images,
    reference_images=reference_images
)
comparison_grid.save("evaluation_grid.png")
```

### Web Demo

#### Launch Locally

```python
from src.app.gradio_app import create_gradio_app

app = create_gradio_app()
app.launch(share=False, server_port=7860)
```

#### Launch with Public URL (Colab)

```python
from src.app.gradio_app import create_gradio_app

app = create_gradio_app()
app.launch(share=True)  # Creates a public Gradio link
```

## Visual Examples

### Condition Map Types

The pipeline supports three types of spatial conditioning. Below are examples from the edge extraction demo (see `outputs/edge_demo/`):

| Original Image | Edge Map (Adaptive) | Edge Map (High Sensitivity) |
|:-:|:-:|:-:|
| `outputs/edge_demo/original_image.png` | `outputs/edge_demo/edges_adaptive.png` | `outputs/edge_demo/edges_high_sensitivity.png` |

| Edge Map (Fixed Threshold) | Edge Map (Low Sensitivity, Inverted) | Realistic Edges |
|:-:|:-:|:-:|
| `outputs/edge_demo/edges_fixed.png` | `outputs/edge_demo/edges_low_sensitivity_inverted.png` | `outputs/edge_demo/realistic_edges.png` |

### Batch Processing Results

The `outputs/edge_demo/` directory contains batch processing results showing multiple images processed through the edge extraction pipeline:

- `outputs/edge_demo/batch_result_0.png` - Batch processing example 1
- `outputs/edge_demo/batch_result_1.png` - Batch processing example 2
- `outputs/edge_demo/batch_result_2.png` - Batch processing example 3

### Pipeline Workflow

```
Source Image → Condition Extractor → Condition Map → ControlNet + SD1.5 → Generated Image
                  (depth/pose/edge)                    (DDIM sampling)
```

**Depth conditioning**: Preserves spatial structure and relative distances. Best for landscapes, architecture, and scenes with clear depth layering.

**Pose conditioning**: Maintains human body positioning and proportions. Best for character generation and pose transfer.

**Edge conditioning**: Preserves outlines and structural boundaries. Best for style transfer while maintaining composition.

## HuggingFace Space Deployment

### Deploy to HuggingFace Spaces

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)

2. Upload your trained model to HuggingFace Hub:
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="./models/trained/controlnet_depth",
    repo_id="your-username/controlnet-depth",
    repo_type="model"
)
```

3. Create an `app.py` for the Space:
```python
from src.app.gradio_app import create_gradio_app

app = create_gradio_app()
app.launch()
```

4. Push to the Space repository:
```bash
git add .
git commit -m "Deploy ControlNet demo"
git push
```

### Space Requirements

The Space `requirements.txt` should include:
```
torch>=2.0.0
diffusers>=0.21.0
transformers>=4.35.0
gradio>=3.50.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
```

## Configuration

### Using YAML Configuration Files

```yaml
# my_experiment.yaml
experiment_name: "depth_controlnet_v2"
seed: 42

dataset:
  subset_size: 10000
  batch_size: 1
  condition_types: ["depth"]

model:
  conditioning_channels: 1
  mixed_precision: "fp16"
  enable_gradient_checkpointing: true
  enable_xformers: true

training:
  num_train_epochs: 100
  learning_rate: 1e-5
  gradient_accumulation_steps: 8
  lr_scheduler: "cosine"
  lr_warmup_steps: 1000
  save_steps: 5000
```

```python
from configs.base_config import BaseConfig

config = BaseConfig.load_config("my_experiment.yaml")
```

### Environment-Specific Configs

```python
from configs.base_config import get_colab_config, get_local_config

# Colab: batch_size=1, gradient_accumulation=8, Drive checkpointing
config = get_colab_config()

# Local: higher batch size, more workers, debug logging
config = get_local_config()
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-5 | Conservative for fine-tuning diffusion models |
| `batch_size` | 1 | T4 GPU memory constraint |
| `gradient_accumulation_steps` | 8 | Effective batch size = 8 |
| `num_train_epochs` | 100 | Full training run |
| `mixed_precision` | fp16 | Reduces memory by ~50% |
| `conditioning_scale` | 1.0 | ControlNet influence strength |
| `guidance_scale` | 7.5 | Classifier-free guidance |
| `num_inference_steps` | 50 | DDIM sampling steps |

## Troubleshooting

### Out of Memory (OOM) Errors

**Symptoms**: `CUDA out of memory`, `RuntimeError: CUDA error`

**Solutions**:

1. **Reduce batch size** (already at 1 for T4):
   ```python
   config.dataset.batch_size = 1
   ```

2. **Enable gradient checkpointing** (trades compute for memory):
   ```python
   config.model.enable_gradient_checkpointing = True
   ```

3. **Use mixed precision training**:
   ```python
   config.model.mixed_precision = "fp16"
   ```

4. **Clear GPU cache before training**:
   ```python
   import torch
   torch.cuda.empty_cache()
   import gc
   gc.collect()
   ```

5. **Reduce image resolution** for initial experiments:
   ```python
   config.dataset.image_size = 256  # Instead of 512
   ```

6. **Use the memory utilities**:
   ```python
   from src.utils.memory_utils import get_gpu_memory_info, optimize_memory
   
   info = get_gpu_memory_info()
   print(f"Used: {info['used_gb']:.1f}GB / {info['total_gb']:.1f}GB")
   optimize_memory()
   ```

### Google Colab Session Disconnects

**Symptoms**: Runtime disconnects after ~12 hours, losing training progress

**Solutions**:

1. **Enable automatic checkpointing**:
   ```python
   from src.utils.colab_helpers import setup_drive_checkpointing
   
   # Saves to Google Drive every 30 minutes
   setup_drive_checkpointing(config)
   ```

2. **Resume from checkpoint after reconnection**:
   ```python
   from src.utils.colab_helpers import restore_latest_checkpoint
   
   checkpoint = restore_latest_checkpoint(config)
   trainer.resume_from_checkpoint(checkpoint)
   ```

3. **Use session timer warnings**:
   ```python
   from src.utils.colab_helpers import ColabSessionManager
   
   session = ColabSessionManager()
   # Warns 60 minutes before timeout
   session.start_monitoring()
   ```

4. **Keep the session alive** (browser-based):
   - Open browser console (F12) and run:
   ```javascript
   function KeepAlive() { 
     document.querySelector("colab-connect-button").click(); 
   }
   setInterval(KeepAlive, 60000);
   ```

### Model Loading Issues

**Symptoms**: `FileNotFoundError`, `OSError: Can't load model`, architecture mismatch errors

**Solutions**:

1. **Verify model path exists**:
   ```python
   from pathlib import Path
   model_path = Path("./models/trained/controlnet_depth")
   print(f"Model exists: {model_path.exists()}")
   print(f"Files: {list(model_path.glob('*'))}")
   ```

2. **Check model compatibility**:
   ```python
   from src.inference.model_loader import verify_model_compatibility
   
   is_compatible, message = verify_model_compatibility(model_path)
   print(message)
   ```

3. **Download from HuggingFace Hub**:
   ```python
   from huggingface_hub import snapshot_download
   
   model_path = snapshot_download(
       repo_id="your-username/controlnet-depth",
       local_dir="./models/trained/controlnet_depth"
   )
   ```

4. **Handle missing pretrained model**:
   ```python
   # The pipeline will download SD1.5 automatically on first use
   config = InferenceConfig(
       pretrained_model_path="runwayml/stable-diffusion-v1-5"
   )
   ```

### Training Divergence

**Symptoms**: Loss becomes NaN or increases rapidly, generated samples are noise

**Solutions**:

1. **Reduce learning rate**:
   ```python
   config.training.learning_rate = 1e-6  # 10x lower
   ```

2. **Increase warmup steps**:
   ```python
   config.training.lr_warmup_steps = 2000
   ```

3. **Enable gradient clipping** (already default):
   ```python
   config.training.max_grad_norm = 1.0
   ```

4. **Check data quality**:
   ```python
   from src.data.verify_dataset import DatasetVerifier
   
   verifier = DatasetVerifier()
   report = verifier.verify_all()
   print(f"Corrupted samples: {report.corrupted_count}")
   ```

### Condition Map Extraction Failures

**Symptoms**: Blank condition maps, extraction errors, model download failures

**Solutions**:

1. **Depth extraction fails**:
   ```python
   # Ensure timm is installed for DPT model
   !pip install timm>=0.9.0
   
   # Use CPU fallback if GPU memory is insufficient
   config = DepthExtractionConfig(device="cpu", precision="fp32")
   ```

2. **Pose extraction fails** (no person detected):
   ```python
   # Enable MediaPipe fallback
   extractor = PoseExtractor(
       prefer_dwpose=True,
       fallback_to_mediapipe=True
   )
   ```

3. **Edge extraction produces poor results**:
   ```python
   # Adjust thresholds for your images
   config = EdgeExtractionConfig(
       adaptive_threshold=True,  # Auto-adjusts per image
       low_threshold=50,         # Lower = more edges
       high_threshold=150        # Higher = fewer edges
   )
   ```

### Import Errors

**Symptoms**: `ModuleNotFoundError`, `ImportError`

**Solutions**:

1. **Ensure package is installed**:
   ```bash
   pip install -e .
   ```

2. **Add project root to path** (Colab):
   ```python
   import sys
   sys.path.insert(0, '/content/Controllable-Image-Generation-with-Fine-Grained-Conditioning')
   ```

3. **Install missing dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Slow Training Speed

**Solutions**:

1. **Enable xFormers** for memory-efficient attention:
   ```python
   config.model.enable_xformers = True
   ```

2. **Enable TF32** on Ampere GPUs:
   ```python
   import torch
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32 = True
   ```

3. **Use persistent data workers**:
   ```python
   config.dataset.persistent_workers = True
   config.dataset.pin_memory = True
   ```

## References

- [Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) - Zhang et al., 2023
- [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) - Runway ML
- [Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502) - Song et al., 2020
- [DPT: Dense Prediction Transformers](https://arxiv.org/abs/2103.13413) - Ranftl et al., 2021
- [DWPose: Effective Whole-body Pose Estimation](https://arxiv.org/abs/2307.15880) - Yang et al., 2023
- [HuggingFace Diffusers](https://github.com/huggingface/diffusers)
- [Gradio](https://gradio.app/)

## License

Apache License 2.0

## Acknowledgments

This project builds upon the work of the ControlNet authors and the HuggingFace community. Training is designed to be accessible on Google Colab free tier, making ControlNet experimentation available to researchers and practitioners without expensive hardware.
