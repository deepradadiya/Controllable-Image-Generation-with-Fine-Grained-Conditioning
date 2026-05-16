#!/usr/bin/env python3
"""
Depth Conditioning Training Script for ControlNet

This script implements the complete training pipeline for depth-conditioned
ControlNet models. It includes data loading, model setup, training loop,
and evaluation with Weights & Biases integration.

Requirements satisfied: 4.1, 4.5, 6.4
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DConditionModel
from diffusers.utils import check_min_version
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import our components
from src.models.controlnet import ControlNetModel
from src.models.unet_wrapper import ControlNetUNet2DConditionModel
from src.models.config import ControlNetConfig, TrainingConfig, ModelMetadata, ControlNetModelManager
from src.training.trainer import ControlNetTrainer, create_trainer_from_configs
from src.training.losses import create_loss_function
from src.data.dataset_processor import DatasetProcessor
from src.data.extract_depth import DepthExtractor

# Check minimum diffusers version
check_min_version("0.21.0")

# Configure logging
logger = get_logger(__name__, log_level="INFO")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ControlNet for depth conditioning")
    
    # Model configuration
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained ControlNet model. If not specified, initializes a new ControlNet model.",
    )
    
    # Dataset configuration
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="coco2017",
        help="The name of the Dataset to train on.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data.",
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this value.",
    )
    
    # Training configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./controlnet-depth-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="The resolution for input images, all the images in the train/validation dataset will be resized to this resolution.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoints to store.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Whether training should be resumed from a previous checkpoint.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help="Whether or not to allow TF32 on Ampere GPUs.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help="The integration to report the results and logs to.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Whether to use mixed precision.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help="Save more memory by using setting grads to None instead of zero.",
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings.",
    )
    
    # ControlNet specific arguments
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.0,
        help="The conditioning scale for ControlNet.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        help="Path to validation image for inference during training.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="a beautiful landscape",
        help="Validation prompt for inference during training.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to generate during validation.",
    )
    
    args = parser.parse_args()
    
    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")
    
    return args


def create_depth_dataset(args, accelerator):
    """Create depth conditioning dataset."""
    logger.info("Creating depth conditioning dataset...")
    
    # Initialize dataset processor
    dataset_processor = DatasetProcessor(
        cache_dir=args.cache_dir,
    )
    
    # Download and process dataset
    dataset = dataset_processor.download_and_process(
        split="train",
        streaming=False,  # For training, we need the full dataset
    )
    
    # Initialize depth extractor
    depth_extractor = DepthExtractor(
        model_name="intel/dpt-large",
        device=accelerator.device,
    )
    
    logger.info(f"Dataset created with {len(dataset)} samples")
    return dataset, depth_extractor


def create_dataloader(dataset, depth_extractor, args, accelerator):
    """Create training dataloader with depth conditioning."""
    from torch.utils.data import Dataset
    import torchvision.transforms as transforms
    from PIL import Image
    
    class DepthConditioningDataset(Dataset):
        def __init__(self, dataset, depth_extractor, resolution=512):
            self.dataset = dataset
            self.depth_extractor = depth_extractor
            self.resolution = resolution
            
            # Image transforms
            self.image_transforms = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            
            # Depth transforms
            self.depth_transforms = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.ToTensor(),
            ])
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            item = self.dataset[idx]
            
            # Get image and caption
            image = item['image']
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif not isinstance(image, Image.Image):
                image = Image.fromarray(image).convert('RGB')
            
            caption = item.get('caption', item.get('text', ''))
            
            # Extract depth map
            depth_map = self.depth_extractor.extract_depth(image)
            
            # Apply transforms
            pixel_values = self.image_transforms(image)
            conditioning_pixel_values = self.depth_transforms(depth_map)
            
            return {
                'pixel_values': pixel_values,
                'conditioning_pixel_values': conditioning_pixel_values,
                'input_ids': caption,  # Will be tokenized later
            }
    
    # Create dataset
    train_dataset = DepthConditioningDataset(
        dataset=dataset,
        depth_extractor=depth_extractor,
        resolution=args.resolution,
    )
    
    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=lambda x: x,  # Custom collate function
    )
    
    return train_dataloader


def main():
    """Main training function."""
    args = parse_args()
    
    # Initialize accelerator
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=args.logging_dir,
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    # Make one log on every process with the configuration for debugging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    # Set seed for reproducibility
    if args.seed is not None:
        set_seed(args.seed)
    
    # Handle output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model configurations
    logger.info("Creating model configurations...")
    
    model_config = ControlNetConfig(
        condition_type="depth",
        conditioning_channels=1,  # Depth maps are grayscale
        in_channels=4,
        block_out_channels=(320, 640, 1280, 1280),
        cross_attention_dim=768,
        attention_head_dim=8,
        use_linear_projection=False,
    )
    
    training_config = TrainingConfig(
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_train_steps=args.max_train_steps,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_num_cycles=args.lr_num_cycles,
        lr_power=args.lr_power,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_weight_decay=args.adam_weight_decay,
        adam_epsilon=args.adam_epsilon,
        max_grad_norm=args.max_grad_norm,
        mixed_precision=args.mixed_precision,
        gradient_checkpointing=args.gradient_checkpointing,
        enable_xformers_memory_efficient_attention=args.enable_xformers_memory_efficient_attention,
        dataloader_num_workers=args.dataloader_num_workers,
        validation_steps=args.validation_steps,
        checkpointing_steps=args.checkpointing_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
        proportion_empty_prompts=args.proportion_empty_prompts,
        logging_dir=args.logging_dir,
        report_to=args.report_to,
        seed=args.seed,
    )
    
    metadata = ModelMetadata(
        model_name="controlnet_depth",
        model_version="1.0.0",
        condition_type="depth",
        base_model=args.pretrained_model_name_or_path,
        training_dataset=args.dataset_name or "custom",
        description="ControlNet model for depth conditioning trained on COCO dataset",
        tags=["depth", "controlnet", "stable-diffusion", "image-generation"],
    )
    
    # Save configurations
    if accelerator.is_main_process:
        config_dir = Path(args.output_dir) / "configs"
        config_dir.mkdir(exist_ok=True)
        
        model_config.save_json(config_dir / "model_config.json")
        training_config.save_json(config_dir / "training_config.json")
        metadata.save_json(config_dir / "metadata.json")
        
        # Save args
        with open(config_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
    
    # Create dataset and dataloader
    logger.info("Setting up dataset...")
    dataset, depth_extractor = create_depth_dataset(args, accelerator)
    train_dataloader = create_dataloader(dataset, depth_extractor, args, accelerator)
    
    # Create models
    logger.info("Creating models...")
    
    # ControlNet
    if args.controlnet_model_name_or_path:
        logger.info(f"Loading ControlNet from {args.controlnet_model_name_or_path}")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing new ControlNet model")
        controlnet = ControlNetModel(
            conditioning_channels=model_config.conditioning_channels,
            in_channels=model_config.in_channels,
            block_out_channels=model_config.block_out_channels,
            cross_attention_dim=model_config.cross_attention_dim,
            attention_head_dim=model_config.attention_head_dim,
            use_linear_projection=model_config.use_linear_projection,
        )
    
    # UNet wrapper
    logger.info(f"Loading UNet from {args.pretrained_model_name_or_path}")
    unet = ControlNetUNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        controlnet_conditioning_scale=training_config.controlnet_conditioning_scale,
    )
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler"
    )
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = ControlNetTrainer(
        controlnet=controlnet,
        unet=unet,
        noise_scheduler=noise_scheduler,
        training_config=training_config,
        model_config=model_config,
        output_dir=args.output_dir,
        device=accelerator.device,
        enable_wandb=(args.report_to == "wandb"),
    )
    
    # Prepare for training with accelerator
    controlnet, unet, train_dataloader, trainer.optimizer = accelerator.prepare(
        controlnet, unet, train_dataloader, trainer.optimizer
    )
    
    # Update trainer with prepared models
    trainer.controlnet = controlnet
    trainer.unet = unet
    
    # Training
    logger.info("Starting training...")
    
    try:
        training_results = trainer.train(
            train_dataloader=train_dataloader,
            validation_dataloader=None,  # No validation for now
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Training results: {training_results}")
        
        # Save final model
        if accelerator.is_main_process:
            logger.info("Saving final model...")
            
            # Update metadata with training results
            metadata.training_steps = training_results['total_steps']
            metadata.training_epochs = training_results['total_epochs']
            metadata.final_loss = training_results['final_loss']
            
            # Save model using model manager
            model_manager = ControlNetModelManager(base_path=args.output_dir)
            
            final_model_path = model_manager.save_model(
                model=accelerator.unwrap_model(controlnet),
                model_config=model_config,
                training_config=training_config,
                metadata=metadata,
                save_directory=Path(args.output_dir) / "final_model",
                push_to_hub=args.push_to_hub,
                hub_model_id=args.hub_model_id,
            )
            
            logger.info(f"Final model saved to: {final_model_path}")
            
            if args.push_to_hub:
                logger.info(f"Model pushed to Hub: {args.hub_model_id}")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise e
    
    finally:
        # Cleanup
        if accelerator.is_main_process:
            logger.info("Training script completed")


if __name__ == "__main__":
    main()