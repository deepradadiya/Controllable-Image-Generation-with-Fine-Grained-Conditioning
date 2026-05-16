# HuggingFace Spaces Deployment Guide

A complete guide for deploying your trained ControlNet models as an interactive web demo on HuggingFace Spaces.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Uploading Models to HuggingFace Hub](#uploading-models-to-huggingface-hub)
4. [Creating a HuggingFace Space](#creating-a-huggingface-space)
5. [Configuring the Gradio App](#configuring-the-gradio-app)
6. [Space Configuration Files](#space-configuration-files)
7. [Deploying the Space](#deploying-the-space)
8. [Monitoring and Maintenance](#monitoring-and-maintenance)
9. [Troubleshooting](#troubleshooting)

---

## Overview

HuggingFace Spaces provides free hosting for ML demos using Gradio or Streamlit. Our ControlNet demo uses Gradio Blocks to provide:

- Image upload with automatic condition map extraction
- Selection between depth, pose, and edge conditioning
- Text prompt input for guided generation
- Adjustable generation parameters (steps, guidance scale, conditioning strength)
- Side-by-side display of condition maps and generated images

### Quick Start Summary

If you're familiar with HuggingFace Spaces, here's the condensed workflow:

1. Upload your trained model to HuggingFace Hub
2. Create a new Gradio Space
3. Copy the project source files into the Space repo
4. Add `app.py` as the entry point (calls `create_gradio_app()`)
5. Push and wait for the build to complete

For detailed instructions, continue reading below.

---

## Prerequisites

- A trained ControlNet model (see [Training Guide](training_guide.md))
- A [HuggingFace account](https://huggingface.co/join)
- HuggingFace CLI installed: `pip install huggingface_hub`
- Git LFS installed: `git lfs install`

---

## Uploading Models to HuggingFace Hub

### Step 1: Authenticate

```bash
huggingface-cli login
# Enter your HuggingFace token when prompted
```

Or programmatically:

```python
from huggingface_hub import login
login(token="hf_your_token_here")
```

### Step 2: Create a Model Repository

```python
from huggingface_hub import create_repo

repo_id = "your-username/controlnet-depth-v1"
create_repo(repo_id, repo_type="model", private=False)
```

### Step 3: Upload Model Weights

#### Option A: Using the Hub API

```python
from huggingface_hub import upload_folder

upload_folder(
    folder_path="./models/trained/controlnet-depth/final_model",
    repo_id="your-username/controlnet-depth-v1",
    repo_type="model",
    commit_message="Upload trained ControlNet depth model",
)
```

#### Option B: Using Git LFS

```bash
git clone https://huggingface.co/your-username/controlnet-depth-v1
cd controlnet-depth-v1

# Copy model files
cp -r /path/to/models/trained/controlnet-depth/final_model/* .

# Push to Hub
git add .
git commit -m "Upload trained ControlNet depth model"
git push
```

### Step 4: Add a Model Card

Create a `README.md` in your model repository:

```markdown
---
license: apache-2.0
tags:
  - controlnet
  - stable-diffusion
  - depth
  - image-generation
pipeline_tag: text-to-image
---

# ControlNet Depth Model

A ControlNet model trained for depth-conditioned image generation with Stable Diffusion 1.5.

## Usage

```python
from diffusers import ControlNetModel
model = ControlNetModel.from_pretrained("your-username/controlnet-depth-v1")
```

## Training Details

- Base model: runwayml/stable-diffusion-v1-5
- Condition type: Depth maps (DPT-Large)
- Dataset: COCO 2017 subset (10,000 images)
- Training steps: 10,000
- Learning rate: 1e-5
```

---

## Creating a HuggingFace Space

### Step 1: Create the Space

Via the HuggingFace website:
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose a name (e.g., `controlnet-demo`)
3. Select **Gradio** as the SDK
4. Choose hardware: **CPU Basic** (free) or **T4 Small** (for GPU inference)
5. Set visibility to Public or Private

Or via CLI:

```python
from huggingface_hub import create_repo

create_repo(
    "your-username/controlnet-demo",
    repo_type="space",
    space_sdk="gradio",
    space_hardware="cpu-basic",  # or "t4-small" for GPU
)
```

### Step 2: Clone the Space Repository

```bash
git clone https://huggingface.co/spaces/your-username/controlnet-demo
cd controlnet-demo
```

---

## Configuring the Gradio App

### App Entry Point

Create `app.py` in the Space root (this is the entry point HuggingFace Spaces looks for):

```python
"""
ControlNet Image Generation Demo - HuggingFace Space Entry Point

This app provides an interactive interface for generating images
conditioned on depth maps, pose skeletons, or edge maps.
"""

import sys
from pathlib import Path

# Add source to path
sys.path.insert(0, str(Path(__file__).parent))

from src.app.gradio_app import create_gradio_app

# Create and launch the app
app = create_gradio_app()
app.launch()
```

### Customizing the Interface

The Gradio app in `src/app/gradio_app.py` provides the full interface. It uses Gradio Blocks with:
- Image upload component with automatic condition map extraction on upload
- Dropdown for condition type selection (depth, pose, edge)
- Text prompt input
- Accordion with generation parameter sliders (inference steps, guidance scale, conditioning strength)
- Side-by-side display of condition map and generated image
- Status messages for user feedback

Key customization points:

```python
# In src/app/gradio_app.py - adjust default parameters for the demo
DEFAULT_NUM_STEPS = 30       # Fewer steps for faster demo response
DEFAULT_GUIDANCE_SCALE = 7.5
DEFAULT_CONDITIONING_STRENGTH = 1.0

# Condition types available in the dropdown
CONDITION_TYPES = ["depth", "pose", "edge"]
```

The app automatically extracts condition maps when an image is uploaded or the condition type changes, providing immediate visual feedback before generation.

### Model Loading Configuration

For the Space, configure model loading from HuggingFace Hub:

```python
# In your app configuration
MODEL_REPOS = {
    "depth": "your-username/controlnet-depth-v1",
    "pose": "your-username/controlnet-pose-v1",
    "edge": "your-username/controlnet-edge-v1",
}
```

---

## Space Configuration Files

### requirements.txt

Create a `requirements.txt` for the Space (separate from the project's main requirements):

```text
torch>=2.0.0
torchvision>=0.15.0
diffusers>=0.21.0
transformers>=4.30.0
accelerate>=0.20.0
safetensors>=0.3.0
gradio>=4.0.0
Pillow>=9.0.0
numpy>=1.23.0
opencv-python-headless>=4.7.0
huggingface_hub>=0.16.0
xformers>=0.0.20
mediapipe>=0.10.0
```

### README.md (Space Card)

Create a `README.md` with Space metadata:

```markdown
---
title: ControlNet Image Generation
emoji: 🎨
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: apache-2.0
tags:
  - controlnet
  - stable-diffusion
  - image-generation
  - depth
  - pose
  - edge
---

# ControlNet Image Generation Demo

Generate images with spatial conditioning using depth maps, pose skeletons, or edge maps.

## How to Use

1. Upload a source image
2. Select a condition type (depth, pose, or edge)
3. Enter a text prompt describing the desired output
4. Adjust generation parameters if needed
5. Click "Generate" to create your image

## Models

This demo uses ControlNet models trained on COCO 2017 with Stable Diffusion 1.5.
```

### .gitattributes

For handling large files:

```
*.safetensors filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
```

---

## Deploying the Space

### Step 1: Organize Files

Your Space repository should have this structure:

```
controlnet-demo/
├── app.py                    # Entry point
├── requirements.txt          # Dependencies
├── README.md                 # Space card
├── .gitattributes           # LFS configuration
├── src/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── gradio_app.py    # Main Gradio interface
│   │   └── controlnet_handler.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── pipeline.py      # Inference pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── controlnet.py    # ControlNet architecture
│   │   └── unet_wrapper.py  # UNet integration
│   ├── data/
│   │   ├── __init__.py
│   │   ├── extract_depth.py
│   │   ├── extract_pose.py
│   │   └── extract_edges.py
│   └── utils/
│       └── __init__.py
└── configs/
    └── base_config.py
```

### Step 2: Push to the Space

```bash
cd controlnet-demo

# Add all files
git add .
git commit -m "Initial deployment of ControlNet demo"
git push
```

### Step 3: Verify Deployment

1. Visit your Space URL: `https://huggingface.co/spaces/your-username/controlnet-demo`
2. Wait for the build to complete (check the "Building" status)
3. Test the interface with a sample image

### Step 4: Upgrade Hardware (Optional)

For GPU inference (much faster generation):

1. Go to Space Settings
2. Under "Space hardware", select "T4 small" or "A10G small"
3. Note: GPU hardware requires a HuggingFace Pro subscription or community GPU grants

---

## Monitoring and Maintenance

### Checking Space Status

```python
from huggingface_hub import HfApi

api = HfApi()
space_info = api.space_info("your-username/controlnet-demo")
print(f"Status: {space_info.runtime.stage}")
print(f"Hardware: {space_info.runtime.hardware}")
```

### Viewing Logs

Access logs from the Space page:
1. Click the "..." menu on your Space
2. Select "See logs"
3. Check for errors or performance issues

Or via API:

```python
from huggingface_hub import HfApi

api = HfApi()
logs = api.get_space_runtime("your-username/controlnet-demo")
```

### Updating the Space

To push updates:

```bash
cd controlnet-demo
# Make changes...
git add .
git commit -m "Update: improved error handling"
git push
```

The Space will automatically rebuild after each push.

### Restarting the Space

If the Space becomes unresponsive:

```python
from huggingface_hub import restart_space

restart_space("your-username/controlnet-demo")
```

### Setting Environment Variables

For API keys or configuration:

1. Go to Space Settings → "Repository secrets"
2. Add secrets (e.g., `WANDB_API_KEY`, `HF_TOKEN`)

These are available as environment variables in your app:

```python
import os
hf_token = os.environ.get("HF_TOKEN")
```

---

## Troubleshooting

### Space Fails to Build

**Check requirements.txt**: Ensure all packages are compatible and versions are pinned.

```bash
# Test locally first
pip install -r requirements.txt
python app.py
```

**Common fixes**:
- Use `opencv-python-headless` instead of `opencv-python` (avoids GUI dependencies)
- Pin specific versions to avoid conflicts
- Check that all imports in `app.py` resolve correctly

### Out of Memory on CPU Spaces

CPU Spaces have limited RAM (16 GB). Solutions:
- Use model quantization (INT8 or FP16 on CPU)
- Load models lazily (only when needed)
- Use smaller model variants if available
- Upgrade to GPU hardware

### Slow Inference

On CPU Spaces, generation can take 2-5 minutes. Solutions:
- Reduce default inference steps to 20
- Use DDIM scheduler (faster than DDPM)
- Upgrade to T4 GPU hardware
- Add a loading indicator in the UI

### Model Loading Failures

```python
# Add robust error handling for model loading
try:
    from src.inference.pipeline import ControlNetInferencePipeline
    pipeline = ControlNetInferencePipeline(config)
except Exception as e:
    # Provide a helpful error message in the UI
    print(f"Model loading failed: {e}")
    print("Ensure model weights are uploaded to HuggingFace Hub")
```

### CORS or Network Issues

If the Space can't download models:
- Ensure model repositories are public (or use `HF_TOKEN` secret)
- Check that the model repo IDs are correct
- Verify network connectivity in Space logs

---

## Advanced: Custom Domain

To use a custom domain for your Space:

1. Go to Space Settings → "Custom domain"
2. Add your domain (e.g., `controlnet.yourdomain.com`)
3. Configure DNS CNAME record pointing to `your-username-controlnet-demo.hf.space`

---

## Next Steps

- See [Training Guide](training_guide.md) for training new models
- See [Hyperparameter Guide](hyperparameter_guide.md) for optimizing model quality
- Explore the evaluation module for measuring deployed model performance

---

## Appendix: Production Considerations

### Scaling for Multiple Users

If your Space receives significant traffic:
- **Queue system**: Gradio automatically queues requests. Set `app.launch(max_threads=2)` to limit concurrent processing.
- **Caching**: The inference pipeline caches loaded models. First request is slow; subsequent requests are faster.
- **Timeout handling**: Set reasonable timeouts for generation (30-60s on GPU, 2-5min on CPU).

### Cost Management

| Hardware | Cost | Use Case |
|----------|------|----------|
| CPU Basic | Free | Demo/testing (slow inference) |
| T4 Small | ~$0.60/hr | Production demos |
| A10G Small | ~$1.05/hr | High-quality, fast inference |

Tips for managing costs:
- Use "sleep after" settings to pause the Space when idle
- Start with CPU for testing, upgrade to GPU only when ready for users
- Monitor usage through the Space analytics dashboard

### Security Best Practices

- Never hardcode API tokens in source files
- Use HuggingFace Space Secrets for sensitive configuration
- Set model repositories to public to avoid authentication issues in the Space
- Validate and sanitize user inputs (image size limits, prompt length limits)
