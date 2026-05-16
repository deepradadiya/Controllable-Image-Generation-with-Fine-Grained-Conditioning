---
title: ControlNet Image Generation
emoji: 🎨
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
license: apache-2.0
hardware: gpu-basic
tags:
  - controlnet
  - stable-diffusion
  - image-generation
  - depth-map
  - pose-estimation
  - edge-detection
---

# ControlNet Image Generation Demo

Generate images conditioned on spatial control maps using ControlNet with Stable Diffusion 1.5.

## Features

- **Depth Conditioning**: Control image structure using depth maps
- **Pose Conditioning**: Guide character positioning with pose skeletons
- **Edge Conditioning**: Define outlines using Canny edge maps

## Usage

1. Upload a source image
2. Select a condition type (depth, pose, or edge)
3. Enter a text prompt describing the desired output
4. Adjust generation parameters (optional)
5. Click "Generate" to create a conditioned image

## Model Information

This demo uses ControlNet adapters trained on COCO 2017 dataset with Stable Diffusion 1.5 as the base model. The implementation follows the architecture from "Adding Conditional Control to Text-to-Image Diffusion Models" (Zhang et al., 2023).

## Technical Details

- **Base Model**: Stable Diffusion 1.5
- **Architecture**: ControlNet with zero convolution initialization
- **Conditioning Types**: Depth (DPT), Pose (DWPose), Edge (Canny)
- **Inference**: DDIM sampling with adjustable conditioning strength
