# Requirements Document

## Introduction

This document specifies the requirements for building the ControlNet model architecture and all three training loops from scratch. The system implements the ControlNet adapter architecture from "Adding Conditional Control to Text-to-Image Diffusion Models" (Zhang et al., 2023) on top of Stable Diffusion 1.5, designed to run on a Google Colab T4 GPU. The implementation covers the core model (`model/controlnet.py`), inference pipeline (`model/pipeline.py`), diffusion loss computation (`training/losses.py`), and three condition-specific training scripts (`training/train_depth.py`, `training/train_pose.py`, `training/train_edge.py`).

## Glossary

- **ControlNet_Adapter**: A trainable copy of the SD1.5 encoder with an extra input channel for the condition image, connected to the frozen UNet via zero convolutions (~360M trainable parameters)
- **SD1_5_UNet**: The Stable Diffusion 1.5 UNet backbone that remains 100% frozen during training — weights never change
- **Zero_Convolution**: A 1x1 convolution layer initialized to zero weights and zero bias — at training start these output nothing, so the model starts identical to vanilla SD1.5 and gradually learns conditioning
- **Condition_Image**: An RGB spatial control signal (depth map, pose skeleton, or Canny edge map) that guides image generation
- **Noisy_Latent**: The latent-space representation of an image with added Gaussian noise at a given diffusion timestep
- **Diffusion_Loss**: The MSE loss between predicted noise and actual noise added to the latent, conditioned on text and spatial control
- **Inference_Pipeline**: The end-to-end system that takes a text prompt and condition image and produces a generated image using DDIM sampling with classifier-free guidance
- **Classifier_Free_Guidance**: A technique that amplifies the effect of conditioning by computing both conditional and unconditional predictions and interpolating between them using a guidance scale
- **Mixed_Precision_Training**: Using float16 for forward/backward passes while keeping optimizer states in float32, halving VRAM usage
- **Gradient_Clipping**: Capping gradient norms at a threshold (1.0) to prevent exploding gradients during training
- **Cosine_LR_Schedule**: A learning rate schedule that decays following a cosine curve from the initial value to near zero
- **VAE_Encoder**: The Stable Diffusion VAE that encodes images from pixel space to latent space (64x64 for 512x512 images)
- **CLIP_Text_Encoder**: The text encoder that converts text prompts into embedding vectors for conditioning the UNet
- **W_and_B**: Weights & Biases experiment tracking platform for logging metrics and sample images
- **HuggingFace_Hub**: Model hosting platform where trained adapter weights are uploaded after training
- **Condition_Embedding_Layer**: A convolutional layer that projects the condition image into the same spatial dimensions as the noisy latent

## Requirements

### Requirement 1: ControlNet Architecture with ASCII Art Documentation

**User Story:** As a researcher, I want a clearly documented ControlNet architecture implementation in `model/controlnet.py`, so that I can understand the data flow and train spatial conditioning adapters for SD1.5.

#### Acceptance Criteria

1. THE `model/controlnet.py` SHALL contain a top-level comment with an ASCII art diagram that uses box-drawing or standard ASCII characters (e.g., `+`, `-`, `|`, `>`) to depict labeled components connected by directional arrows in the following flow: Condition Image → ControlNet Encoder (trainable copy) → Zero Convolutions → SD1.5 UNet (frozen) → Generated Image, with each component labeled and trainable components visually distinguished from frozen components (e.g., annotated with "(trainable)" or "(frozen)")
2. THE top-level comment SHALL explain that ControlNet_Adapter is a COPY of the SD1.5 encoder with an extra input channel for the condition
3. THE top-level comment SHALL explain that Zero_Convolution layers are 1x1 conv layers initialized to zero so the model starts identical to vanilla SD1.5 and gradually learns conditioning
4. THE top-level comment SHALL explain that SD1_5_UNet is 100% frozen and only the adapter parameters (between 350M and 370M parameters) are trained
5. THE top-level comment SHALL explain why this architecture fits on a T4 GPU: backpropagation never flows through the full UNet, so only the adapter's memory footprint is required for gradient storage

### Requirement 2: ControlNet Class Implementation

**User Story:** As a developer, I want a complete ControlNet class that accepts condition images and produces modified UNet outputs, so that I can train spatial conditioning for any condition type.

#### Acceptance Criteria

1. THE ControlNet_Adapter class SHALL accept a `condition_channels` parameter with a default value of 3, supporting integer values of 1 or 3 to handle grayscale (depth, edge) or RGB (pose) condition images
2. WHEN initialized, THE ControlNet_Adapter SHALL create encoder blocks matching the SD1.5 UNet encoder structure (4 down blocks with channel dimensions 320, 640, 1280, 1280) and load pretrained weights from the SD1.5 UNet checkpoint into them
3. WHEN initialized, THE ControlNet_Adapter SHALL freeze all copied encoder block weights by setting `requires_grad=False` on every parameter in the copied blocks, so that no gradients are computed for those parameters during training
4. THE ControlNet_Adapter SHALL contain trainable Zero_Convolution layers (one per encoder resolution scale plus one for the mid block) initialized with all weights and biases set to zero, ensuring the ControlNet output is zero at the start of training
5. THE ControlNet_Adapter SHALL contain a Condition_Embedding_Layer that progressively downsamples the condition image from full resolution (H × W) to the noisy latent spatial resolution (H/8 × W/8) with an output channel dimension of 320
6. WHEN the forward method is called with (noisy_latent of shape B×4×H×W, timestep, text_embedding of shape B×77×768, condition_image of shape B×C_cond×8H×8W), THE ControlNet_Adapter SHALL return a dictionary containing `down_block_res_samples` (a list of feature tensors, one per resolution scale) and `mid_block_res_sample` (a single feature tensor from the mid block)
7. WHEN initialized, THE ControlNet_Adapter SHALL print the count of trainable parameters versus frozen parameters to the console
8. IF the pretrained SD1.5 UNet weights cannot be loaded during initialization, THEN THE ControlNet_Adapter SHALL raise an error indicating that the pretrained model path is invalid or inaccessible
9. THE ControlNet_Adapter forward method SHALL accept a `conditioning_scale` float parameter (default 1.0, range 0.0 to 2.0) that multiplies all output feature tensors, allowing runtime control of conditioning strength

### Requirement 3: Inference Pipeline

**User Story:** As a user, I want a full inference pipeline in `model/pipeline.py` that takes a text prompt and condition image and produces a generated image, so that I can generate conditioned images after training.

#### Acceptance Criteria

1. THE Inference_Pipeline SHALL accept a `text_prompt` (string, maximum 77 tokens after CLIP tokenization), a `condition_image` (PIL Image, resized internally to 512×512 pixels), and a `condition_type` argument restricted to one of {"depth", "pose", "edge"}
2. THE Inference_Pipeline SHALL implement Classifier_Free_Guidance with a configurable `guidance_scale` parameter that defaults to 7.5 and accepts values in the range 1.0 to 20.0
3. THE Inference_Pipeline SHALL include a comment explaining Classifier_Free_Guidance: computing conditional and unconditional predictions and interpolating between them
4. THE Inference_Pipeline SHALL use `num_inference_steps=20` as default for fast demo inference
5. WHEN inference completes, THE Inference_Pipeline SHALL save a side-by-side composite image (condition image on the left, generated image on the right) as a PNG file for visual comparison
6. IF `condition_type` is not one of {"depth", "pose", "edge"}, THEN THE Inference_Pipeline SHALL raise a ValueError with an error message indicating the invalid condition type and listing the supported types
7. WHEN inference completes, THE Inference_Pipeline SHALL return a generated image as a PIL Image of size 512×512 pixels in RGB mode

### Requirement 4: Diffusion Training Loss

**User Story:** As a practitioner, I want a clearly documented diffusion loss implementation in `training/losses.py`, so that I can understand and verify the training objective.

#### Acceptance Criteria

1. THE `training/losses.py` SHALL contain a top-level comment explaining the diffusion training loss: random noise is added to images at a random timestep, then the model is trained to predict that noise, with the text embedding and condition image guiding which image to reconstruct
2. THE Diffusion_Loss function SHALL accept predicted noise and actual noise tensors of identical shape and return a scalar MSE loss value suitable for backpropagation
3. WHEN the current training step is a multiple of 10, THE Diffusion_Loss function SHALL print the current step number and loss value to standard output
4. IF the predicted noise and actual noise tensors have mismatched shapes, THEN THE Diffusion_Loss function SHALL raise an error indicating the shape mismatch

### Requirement 5: Depth Conditioning Training Script

**User Story:** As a practitioner, I want a complete training script in `training/train_depth.py` for depth-conditioned ControlNet, so that I can train the adapter end-to-end on a T4 GPU.

#### Acceptance Criteria

1. WHEN training begins, THE `training/train_depth.py` SHALL load the SD1.5 VAE, UNet, and CLIP text encoder with ALL weights frozen (requires_grad set to False for every parameter)
2. WHEN training begins, THE `training/train_depth.py` SHALL load the trainable ControlNet_Adapter as the only module with requires_grad set to True
3. THE `training/train_depth.py` SHALL use AdamW optimizer with learning_rate=1e-5, betas=(0.9, 0.999), weight_decay=1e-2, and a cosine learning rate schedule with 500 warmup steps, applied only to ControlNet_Adapter parameters
4. WHEN a training step executes, THE `training/train_depth.py` SHALL encode the target image to latent space using VAE_Encoder with a batch size of 1 and gradient accumulation over 8 steps to simulate an effective batch size of 8
5. WHEN a training step executes, THE `training/train_depth.py` SHALL sample a random timestep in the range [0, 999] and add noise to the latent using the DDPM noise scheduler
6. WHEN a training step executes, THE `training/train_depth.py` SHALL encode the text prompt using CLIP_Text_Encoder with a maximum token length of 77
7. WHEN a training step executes, THE `training/train_depth.py` SHALL encode the depth condition image (shape: 3×512×512, normalized to [0, 1]) using the ControlNet_Adapter encoder
8. WHEN a training step executes, THE `training/train_depth.py` SHALL predict noise using the ControlNet-modified UNet and compute MSE loss between predicted noise and the actual noise added in step 5
9. WHEN a training step executes, THE `training/train_depth.py` SHALL backpropagate only through ControlNet_Adapter parameters (frozen layers shall receive no gradients and remain unchanged)
10. THE `training/train_depth.py` SHALL use Mixed_Precision_Training with torch.autocast (dtype=float16) and a GradScaler, and include a comment explaining why: halves VRAM usage by storing activations in float16
11. THE `training/train_depth.py` SHALL apply Gradient_Clipping at max_norm=1.0 before each optimizer step and include a comment explaining why: prevents exploding gradients that destabilize training
12. THE `training/train_depth.py` SHALL log to W_and_B every 250 steps: the current training loss value, the current learning rate, and 4 sample images generated from fixed validation prompts and condition maps
13. THE `training/train_depth.py` SHALL save a checkpoint to Google Drive every 250 steps containing the ControlNet_Adapter state_dict, optimizer state, LR scheduler state, and current step number, retaining a maximum of 3 most recent checkpoints
14. WHEN training completes after a maximum of 10000 training steps, THE `training/train_depth.py` SHALL save adapter weights in safetensors format and upload to HuggingFace_Hub at "{username}/controlnet-sd15-depth"
15. THE `training/train_depth.py` SHALL include a warning comment that estimated training time on T4 is approximately 3 hours and suggest splitting across sessions using the checkpoint resume mechanism
16. IF a checkpoint exists at the configured output directory, THEN THE `training/train_depth.py` SHALL support resuming training from the latest checkpoint by restoring model weights, optimizer state, LR scheduler state, and step counter

### Requirement 6: Pose Conditioning Training Script

**User Story:** As a practitioner, I want a complete training script in `training/train_pose.py` for pose-conditioned ControlNet, so that I can train a pose adapter with the same architecture.

#### Acceptance Criteria

1. THE `training/train_pose.py` SHALL implement the same training loop as `training/train_depth.py` with `condition_type` set to pose and `conditioning_channels` set to 3 (RGB pose skeletons)
2. WHEN training completes, THE `training/train_pose.py` SHALL save adapter weights to HuggingFace_Hub at "{username}/controlnet-sd15-pose"
3. THE `training/train_pose.py` SHALL include the same Mixed_Precision_Training, Gradient_Clipping, W_and_B logging, and checkpoint saving as the depth training script
4. THE `training/train_pose.py` SHALL accept the same command-line arguments as `training/train_depth.py`, with `--output_dir` defaulting to "./controlnet-pose-model" and `--hub_model_id` defaulting to "{username}/controlnet-sd15-pose"

### Requirement 7: Edge Conditioning Training Script

**User Story:** As a practitioner, I want a complete training script in `training/train_edge.py` for edge-conditioned ControlNet, so that I can train an edge adapter with the same architecture.

#### Acceptance Criteria

1. THE `training/train_edge.py` SHALL implement a training loop structurally identical to `training/train_depth.py`, with `condition_type` set to "edge" and `conditioning_channels` set to 1 (grayscale Canny edge maps)
2. WHEN training completes successfully, THE `training/train_edge.py` SHALL save adapter weights to HuggingFace Hub at "{username}/controlnet-sd15-edge" using the `push_to_hub` flag
3. THE `training/train_edge.py` SHALL use fp16 mixed precision training, gradient clipping with `max_grad_norm` of 1.0, Weights & Biases logging via the `report_to` argument defaulting to "wandb", and checkpoint saving every 250 training steps
4. IF training is interrupted or fails before completion, THEN THE `training/train_edge.py` SHALL preserve the most recent checkpoint so that training can be resumed via the `resume_from_checkpoint` argument
5. THE `training/train_edge.py` SHALL accept the same command-line arguments as `training/train_depth.py`, with `--output_dir` defaulting to "./controlnet-edge-model" and `--hub_model_id` defaulting to "{username}/controlnet-sd15-edge"

### Requirement 8: Zero Convolution Initialization

**User Story:** As a researcher, I want zero convolutions properly initialized so that training starts from a stable baseline identical to vanilla SD1.5.

#### Acceptance Criteria

1. THE Zero_Convolution layers SHALL be 1x1 convolution layers (`nn.Conv2d` with `kernel_size=1`) with both weights and biases initialized to exactly 0.0 using `nn.init.zeros_`
2. WHEN training starts, THE Zero_Convolution layers SHALL output zero tensors for any input, so that the combined SD1_5_UNet output is numerically identical (within floating-point tolerance of 1e-5) to the SD1_5_UNet output without the ControlNet_Adapter connected
3. THE Zero_Convolution layers SHALL have input and output channel dimensions matching the encoder block output they connect to, with one Zero_Convolution layer per skip connection between the ControlNet_Adapter and the SD1_5_UNet decoder
4. WHEN a forward pass is executed after initialization and before any optimizer step, THE Zero_Convolution layers SHALL produce all-zero output tensors verifiable by asserting that the absolute maximum value of each output tensor equals 0.0
5. WHILE training is in progress, THE Zero_Convolution layers SHALL have `requires_grad=True` so that they accumulate gradients and update their weights via the optimizer

### Requirement 9: Parameter Counting and Memory Efficiency

**User Story:** As a Colab user, I want to verify that only adapter parameters are trainable and the model fits in T4 VRAM, so that I can train without running out of memory.

#### Acceptance Criteria

1. WHEN the ControlNet_Adapter is initialized, THE system SHALL print the total count of trainable parameters and the total count of frozen parameters, each as an integer number of parameters
2. WHEN the ControlNet_Adapter is initialized, THE system SHALL verify that the trainable parameter count is between 300M and 420M parameters
3. WHEN a backward pass completes during training, THE system SHALL ensure that all parameters in SD1_5_UNet, VAE_Encoder, and CLIP_Text_Encoder have no accumulated gradients (grad remains None)
4. WHILE training with a batch size of 1 and input resolution of 512x512, THE training scripts SHALL maintain peak GPU VRAM usage at or below 15GB on a T4 GPU using Mixed_Precision_Training and gradient checkpointing
5. WHEN the first training step completes, THE system SHALL print the peak GPU memory allocated in megabytes for verification

### Requirement 10: Training Loop Shared Infrastructure

**User Story:** As a developer, I want shared training infrastructure across all three condition types, so that the training scripts are consistent and maintainable.

#### Acceptance Criteria

1. THE training scripts SHALL share the same optimizer configuration: AdamW with lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-2, epsilon=1e-8
2. THE training scripts SHALL share the same Cosine_LR_Schedule configuration with lr_warmup_steps=500 and num_cycles=0.5, decaying from lr=1e-5 to near zero
3. THE training scripts SHALL share the same checkpoint saving logic to Google Drive every 250 steps, saving ControlNet_Adapter weights, optimizer state, scheduler state, random states, and current step count, retaining a maximum of 3 most recent checkpoints
4. THE training scripts SHALL share the same W_and_B logging configuration: loss, learning_rate, and 4 sample_images every 250 steps
5. THE training scripts SHALL share the same Mixed_Precision_Training configuration using torch.autocast with dtype=float16
6. THE training scripts SHALL share the same Gradient_Clipping configuration at max_norm=1.0
7. THE training scripts SHALL import shared infrastructure from a common module rather than duplicating configuration code across scripts
8. IF a checkpoint exists at the configured Google Drive path, THEN THE training scripts SHALL support resuming training from that checkpoint by restoring model weights, optimizer state, scheduler state, and step count

### Requirement 11: Condition Image Overlay Output

**User Story:** As a user evaluating results, I want generated images saved with condition overlays, so that I can visually compare how well the model follows the conditioning.

#### Acceptance Criteria

1. WHEN the Inference_Pipeline saves output images, THE system SHALL create a side-by-side composite placing the condition image on the left and the generated image on the right, both scaled to equal height
2. THE composite output SHALL render a text label indicating the condition type used ("depth", "pose", or "edge") positioned above or below the condition image without overlapping image content
3. THE composite output SHALL be saved as a lossless PNG file at the combined resolution of both images (condition width + generated width by shared height)
4. IF the condition image dimensions differ from the generated image dimensions, THEN THE system SHALL resize the condition image to match the generated image height while preserving aspect ratio before compositing
5. WHEN saving the composite, THE system SHALL use a filename that includes the condition type and a timestamp or sequential index to distinguish between multiple inference runs

### Requirement 12: HuggingFace Hub Model Upload

**User Story:** As a practitioner, I want trained models automatically uploaded to HuggingFace Hub, so that I can share and reuse them easily.

#### Acceptance Criteria

1. WHEN training completes for depth conditioning, THE system SHALL upload weights to "{username}/controlnet-sd15-depth" on HuggingFace_Hub
2. WHEN training completes for pose conditioning, THE system SHALL upload weights to "{username}/controlnet-sd15-pose" on HuggingFace_Hub
3. WHEN training completes for edge conditioning, THE system SHALL upload weights to "{username}/controlnet-sd15-edge" on HuggingFace_Hub
4. THE uploaded model SHALL include a model card containing: the base model identifier ("runwayml/stable-diffusion-v1-5"), the condition type (depth, pose, or edge), the learning rate, the number of training steps completed, the dataset name, and the training precision setting
5. IF the HuggingFace_Hub upload fails due to network error or authentication failure, THEN THE system SHALL save the model weights locally to the configured output directory and log an error message indicating the upload failure reason
6. IF no valid HuggingFace_Hub token is configured, THEN THE system SHALL skip the upload, save the model weights locally, and log a warning message indicating that upload was skipped due to missing authentication
