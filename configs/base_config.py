"""
Base Configuration for ControlNet Training Pipeline

This module provides centralized configuration management for all components
of the ControlNet training pipeline. Configurations are optimized for Google
Colab T4 GPU constraints (15GB VRAM, ~13GB usable).

The configuration system uses dataclasses for type safety and OmegaConf
for flexible configuration loading from YAML files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Import torch with fallback for environments where it's not installed
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


@dataclass
class DatasetConfig:
    """Configuration for dataset processing and loading"""
    
    # Dataset source and size
    dataset_name: str = "COCO"
    dataset_version: str = "2017"
    subset_size: int = 10000  # Reduced for Colab constraints
    validation_split: float = 0.1
    
    # Data paths
    data_root: str = "./data"
    cache_dir: str = "./data/cache"
    processed_dir: str = "./data/processed"
    
    # Image processing
    image_size: int = 512  # Standard SD1.5 resolution
    condition_map_size: int = 512
    max_prompt_length: int = 77  # CLIP text encoder limit
    
    # Data loading
    batch_size: int = 1  # T4 GPU memory constraint
    num_workers: int = 2  # Colab CPU cores
    pin_memory: bool = True
    persistent_workers: bool = True
    
    # Condition types
    condition_types: List[str] = field(default_factory=lambda: ["depth", "pose", "edge"])
    
    # Quality thresholds
    min_image_size: int = 256
    max_aspect_ratio: float = 2.0
    min_prompt_length: int = 5


@dataclass
class ModelConfig:
    """Configuration for ControlNet and UNet models"""
    
    # Base model
    pretrained_model_name: str = "runwayml/stable-diffusion-v1-5"
    revision: str = "main"
    torch_dtype: str = "float16"  # Memory optimization
    
    # ControlNet architecture
    conditioning_channels: int = 3  # RGB condition maps
    block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280)
    attention_head_dim: int = 8
    cross_attention_dim: int = 768
    use_linear_projection: bool = False
    
    # Memory optimizations
    enable_xformers: bool = True  # Memory efficient attention
    enable_gradient_checkpointing: bool = True
    enable_cpu_offload: bool = False  # Keep on GPU for speed
    
    # Model precision
    mixed_precision: str = "fp16"  # "no", "fp16", "bf16"
    
    # Zero convolution initialization
    zero_init_type: str = "zero"  # "zero", "kaiming", "xavier"


@dataclass
class TrainingConfig:
    """Configuration for training process"""
    
    # Training duration
    num_train_epochs: int = 100
    max_train_steps: Optional[int] = None
    gradient_accumulation_steps: int = 8  # Simulate larger batch size
    
    # Learning rate and optimization
    learning_rate: float = 1e-5  # Conservative for fine-tuning
    lr_scheduler: str = "cosine"  # "linear", "cosine", "constant"
    lr_warmup_steps: int = 1000
    lr_num_cycles: float = 0.5
    
    # Optimizer settings
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Noise scheduling
    noise_scheduler: str = "ddpm"  # "ddpm", "ddim", "pndm"
    num_train_timesteps: int = 1000
    beta_start: float = 0.00085
    beta_end: float = 0.012
    beta_schedule: str = "scaled_linear"
    
    # Conditioning
    conditioning_dropout_prob: float = 0.1  # Classifier-free guidance
    conditioning_scale_range: Tuple[float, float] = (0.5, 1.5)
    
    # Checkpointing and logging
    save_steps: int = 5000
    logging_steps: int = 100
    eval_steps: int = 1000
    checkpointing_steps: int = 1000
    max_checkpoints: int = 3
    
    # Validation during training
    validation_epochs: int = 10
    num_validation_images: int = 4
    validation_guidance_scale: float = 7.5
    validation_num_inference_steps: int = 20


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    
    # FID computation
    fid_batch_size: int = 16
    fid_num_samples: int = 1000
    inception_dims: int = 2048
    
    # Condition alignment metrics
    alignment_batch_size: int = 8
    alignment_num_samples: int = 500
    
    # Visual evaluation
    visual_grid_size: Tuple[int, int] = (4, 4)
    comparison_image_size: int = 256
    
    # Inference parameters for evaluation
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    conditioning_scale: float = 1.0


@dataclass
class ColabConfig:
    """Configuration for Google Colab environment"""
    
    # Session management
    session_timeout_hours: int = 12
    checkpoint_interval_minutes: int = 30
    warning_before_timeout_minutes: int = 60
    
    # Google Drive integration
    use_drive_storage: bool = True
    drive_mount_path: str = "/content/drive"
    drive_project_path: str = "MyDrive/ControlNet"
    
    # Memory management
    gpu_memory_threshold: float = 0.9  # Trigger cleanup at 90%
    enable_memory_monitoring: bool = True
    memory_check_interval: int = 100  # steps
    
    # Performance optimization
    enable_tf32: bool = True  # Faster training on Ampere GPUs
    dataloader_num_workers: int = 2
    pin_memory: bool = True


@dataclass
class InferenceConfig:
    """Configuration for inference pipeline"""
    
    # Generation parameters
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    conditioning_scale: float = 1.0
    negative_prompt: str = "blurry, bad quality, distorted, deformed"
    
    # Scheduler settings
    scheduler_type: str = "ddim"  # "ddim", "ddpm", "pndm", "euler"
    
    # Batch processing
    batch_size: int = 1
    max_batch_size: int = 4  # T4 memory limit
    
    # Output settings
    output_format: str = "png"
    output_quality: int = 95
    return_latents: bool = False


@dataclass
class WebDemoConfig:
    """Configuration for Gradio web demo"""
    
    # Interface settings
    title: str = "ControlNet Image Generation"
    description: str = "Generate images with spatial conditioning using depth, pose, or edge maps"
    
    # Generation parameters (user-adjustable)
    default_num_steps: int = 20  # Faster for demo
    min_steps: int = 10
    max_steps: int = 50
    
    default_guidance_scale: float = 7.5
    min_guidance_scale: float = 1.0
    max_guidance_scale: float = 20.0
    
    default_conditioning_scale: float = 1.0
    min_conditioning_scale: float = 0.0
    max_conditioning_scale: float = 2.0
    
    # Interface limits
    max_image_size: int = 768
    max_prompt_length: int = 200
    
    # Performance
    enable_queue: bool = True
    max_queue_size: int = 10
    timeout_seconds: int = 300


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring"""
    
    # Logging levels
    log_level: str = "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
    console_log_level: str = "INFO"
    file_log_level: str = "DEBUG"
    
    # Log files
    log_dir: str = "./logs"
    log_filename: str = "controlnet_training.log"
    max_log_size_mb: int = 100
    backup_count: int = 5
    
    # Weights & Biases
    use_wandb: bool = True
    wandb_project: str = "controlnet-training"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=lambda: ["controlnet", "stable-diffusion"])
    
    # Monitoring
    log_system_stats: bool = True
    log_gpu_stats: bool = True
    stats_interval: int = 100  # steps


@dataclass
class PathConfig:
    """Configuration for file paths and directories"""
    
    # Root directories
    project_root: str = "."
    data_dir: str = "./data"
    models_dir: str = "./models"
    outputs_dir: str = "./outputs"
    logs_dir: str = "./logs"
    cache_dir: str = "./cache"
    
    # Model paths
    pretrained_models_dir: str = "./models/pretrained"
    trained_models_dir: str = "./models/trained"
    checkpoints_dir: str = "./models/checkpoints"
    
    # Data paths
    raw_data_dir: str = "./data/raw"
    processed_data_dir: str = "./data/processed"
    condition_maps_dir: str = "./data/condition_maps"
    
    # Output paths
    generated_images_dir: str = "./outputs/generated"
    evaluation_results_dir: str = "./outputs/evaluation"
    visualizations_dir: str = "./outputs/visualizations"


@dataclass
class BaseConfig:
    """Main configuration class combining all sub-configurations"""
    
    # Sub-configurations
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    colab: ColabConfig = field(default_factory=ColabConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    web_demo: WebDemoConfig = field(default_factory=WebDemoConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Global settings
    seed: int = 42
    device: str = "auto"  # "auto", "cuda", "cpu"
    experiment_name: str = "controlnet_experiment"
    
    # Environment detection
    is_colab: bool = field(init=False)
    is_kaggle: bool = field(init=False)
    is_local: bool = field(init=False)
    
    def __post_init__(self):
        """Initialize environment detection and device settings"""
        self._detect_environment()
        self._setup_device()
        self._create_directories()
    
    def _detect_environment(self):
        """Detect the current execution environment"""
        try:
            import google.colab
            self.is_colab = True
        except ImportError:
            self.is_colab = False
        
        try:
            import kaggle
            self.is_kaggle = True
        except ImportError:
            self.is_kaggle = False
        
        self.is_local = not (self.is_colab or self.is_kaggle)
    
    def _setup_device(self):
        """Setup device configuration based on environment"""
        if self.device == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
                # Enable optimizations for T4 GPU
                if self.is_colab:
                    torch.backends.cuda.matmul.allow_tf32 = self.colab.enable_tf32
                    torch.backends.cudnn.allow_tf32 = self.colab.enable_tf32
            else:
                self.device = "cpu"
                # Adjust settings for CPU-only execution
                self.model.mixed_precision = "no"
                self.model.enable_xformers = False
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.paths.data_dir,
            self.paths.models_dir,
            self.paths.outputs_dir,
            self.paths.logs_dir,
            self.paths.cache_dir,
            self.paths.pretrained_models_dir,
            self.paths.trained_models_dir,
            self.paths.checkpoints_dir,
            self.paths.raw_data_dir,
            self.paths.processed_data_dir,
            self.paths.condition_maps_dir,
            self.paths.generated_images_dir,
            self.paths.evaluation_results_dir,
            self.paths.visualizations_dir,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get_condition_config(self, condition_type: str) -> Dict:
        """Get condition-specific configuration"""
        condition_configs = {
            "depth": {
                "model_name": "Intel/dpt-large",
                "conditioning_channels": 1,
                "preprocessing": "normalize_depth",
                "color_mode": "grayscale"
            },
            "pose": {
                "model_name": "dwpose",
                "conditioning_channels": 3,
                "preprocessing": "render_skeleton",
                "color_mode": "rgb",
                "keypoint_threshold": 0.3
            },
            "edge": {
                "model_name": "canny",
                "conditioning_channels": 1,
                "preprocessing": "canny_edge",
                "color_mode": "grayscale",
                "low_threshold": 100,
                "high_threshold": 200
            }
        }
        
        if condition_type not in condition_configs:
            raise ValueError(f"Unknown condition type: {condition_type}")
        
        return condition_configs[condition_type]
    
    def get_memory_config(self) -> Dict:
        """Get memory optimization configuration based on available GPU memory"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {"batch_size": 1, "gradient_accumulation_steps": 1}
        
        # Get GPU memory in GB
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        if gpu_memory_gb >= 24:  # RTX 3090/4090
            return {
                "batch_size": 4,
                "gradient_accumulation_steps": 2,
                "enable_gradient_checkpointing": False
            }
        elif gpu_memory_gb >= 16:  # RTX 3080/4080
            return {
                "batch_size": 2,
                "gradient_accumulation_steps": 4,
                "enable_gradient_checkpointing": True
            }
        else:  # T4, RTX 3060, etc.
            return {
                "batch_size": 1,
                "gradient_accumulation_steps": 8,
                "enable_gradient_checkpointing": True
            }
    
    def save_config(self, path: Union[str, Path]):
        """Save configuration to YAML file"""
        from omegaconf import OmegaConf
        
        # Convert to OmegaConf for easy serialization
        conf = OmegaConf.structured(self)
        OmegaConf.save(conf, path)
    
    @classmethod
    def load_config(cls, path: Union[str, Path]) -> 'BaseConfig':
        """Load configuration from YAML file"""
        from omegaconf import OmegaConf
        
        # Load YAML and merge with default config
        conf = OmegaConf.load(path)
        default_conf = OmegaConf.structured(cls())
        merged_conf = OmegaConf.merge(default_conf, conf)
        
        return OmegaConf.to_object(merged_conf)


# Default configuration instance
default_config = BaseConfig()


def get_config(config_path: Optional[Union[str, Path]] = None) -> BaseConfig:
    """
    Get configuration instance
    
    Args:
        config_path: Optional path to custom configuration file
        
    Returns:
        BaseConfig instance
    """
    if config_path is None:
        return default_config
    else:
        return BaseConfig.load_config(config_path)


# Condition-specific configuration helpers
def get_depth_config() -> BaseConfig:
    """Get configuration optimized for depth conditioning"""
    config = BaseConfig()
    config.model.conditioning_channels = 1
    config.dataset.condition_types = ["depth"]
    return config


def get_pose_config() -> BaseConfig:
    """Get configuration optimized for pose conditioning"""
    config = BaseConfig()
    config.model.conditioning_channels = 3
    config.dataset.condition_types = ["pose"]
    return config


def get_edge_config() -> BaseConfig:
    """Get configuration optimized for edge conditioning"""
    config = BaseConfig()
    config.model.conditioning_channels = 1
    config.dataset.condition_types = ["edge"]
    return config


# Environment-specific configuration helpers
def get_colab_config() -> BaseConfig:
    """Get configuration optimized for Google Colab"""
    config = BaseConfig()
    
    # Colab-specific optimizations
    config.dataset.batch_size = 1
    config.training.gradient_accumulation_steps = 8
    config.model.enable_gradient_checkpointing = True
    config.model.mixed_precision = "fp16"
    config.colab.use_drive_storage = True
    
    # Reduce dataset size for faster iteration
    config.dataset.subset_size = 5000
    config.training.num_train_epochs = 50
    
    return config


def get_local_config() -> BaseConfig:
    """Get configuration optimized for local development"""
    config = BaseConfig()
    
    # Local development optimizations
    config.colab.use_drive_storage = False
    config.dataset.num_workers = 4
    config.logging.log_level = "DEBUG"
    
    return config


if __name__ == "__main__":
    # Example usage and configuration validation
    config = BaseConfig()
    print(f"Environment: Colab={config.is_colab}, Kaggle={config.is_kaggle}, Local={config.is_local}")
    print(f"Device: {config.device}")
    print(f"Memory config: {config.get_memory_config()}")
    
    # Save example configuration
    config.save_config("example_config.yaml")
    print("Example configuration saved to example_config.yaml")