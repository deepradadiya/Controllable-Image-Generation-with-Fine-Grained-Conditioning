# ControlNet Training Pipeline - Setup Guide

This guide covers the dependency management and configuration system implemented for the ControlNet training pipeline.

## 📁 Project Structure

The project follows a modular structure optimized for Google Colab and local development:

```
controlnet-training-pipeline/
├── configs/                    # Configuration management
│   ├── __init__.py
│   ├── base_config.py         # Centralized configuration system
│   └── example_config.yaml    # Example configuration file
├── src/                       # Source code package
│   └── __init__.py
├── data/                      # Dataset storage (auto-created)
├── models/                    # Model storage (auto-created)
├── outputs/                   # Generated outputs (auto-created)
├── logs/                      # Training logs (auto-created)
├── cache/                     # Cache directory (auto-created)
├── requirements.txt           # Pinned dependencies for Colab T4
├── setup.py                   # Package installation script
└── validate_setup.py          # Setup validation script
```

## 🔧 Dependency Management

### Requirements.txt

The `requirements.txt` file contains all dependencies with pinned versions optimized for Google Colab T4 GPU:

- **Core ML Framework**: PyTorch 2.0.1 with CUDA 11.8 support
- **Diffusion Models**: Diffusers, Transformers, Accelerate from HuggingFace
- **Dataset Processing**: Datasets, OpenCV, NumPy, Pillow
- **Condition Extraction**: Timm (DPT), MediaPipe, ControlNet-aux
- **Training Optimization**: xFormers, BitsAndBytes, Weights & Biases
- **Evaluation**: SciPy, Scikit-image, LPIPS
- **Web Demo**: Gradio, FastAPI
- **Development**: Pytest, Black, Flake8, MyPy

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Validate installation
python validate_setup.py
```

### Colab Installation

```python
# In Google Colab
!pip install -r requirements.txt
!pip install -e .

# Test installation
from controlnet_pipeline import BaseConfig
config = BaseConfig()
print(f"Device: {config.device}")
```

## ⚙️ Configuration System

### BaseConfig Class

The configuration system provides centralized management of all hyperparameters and settings:

```python
from controlnet_pipeline import BaseConfig, get_config

# Create default configuration
config = BaseConfig()

# Access sub-configurations
print(f"Batch size: {config.dataset.batch_size}")
print(f"Learning rate: {config.training.learning_rate}")
print(f"Device: {config.device}")
```

### Configuration Categories

1. **DatasetConfig**: Dataset processing and loading settings
2. **ModelConfig**: ControlNet and UNet architecture settings
3. **TrainingConfig**: Training loop and optimization settings
4. **EvaluationConfig**: Evaluation metrics and validation settings
5. **ColabConfig**: Google Colab specific optimizations
6. **InferenceConfig**: Image generation pipeline settings
7. **WebDemoConfig**: Gradio web interface settings
8. **LoggingConfig**: Logging and monitoring settings
9. **PathConfig**: File paths and directory structure

### Environment Detection

The configuration system automatically detects the execution environment:

```python
config = BaseConfig()
print(f"Colab: {config.is_colab}")
print(f"Kaggle: {config.is_kaggle}")
print(f"Local: {config.is_local}")
```

### Memory Optimization

Automatic memory configuration based on available GPU:

```python
config = BaseConfig()
memory_config = config.get_memory_config()
print(memory_config)
# Output: {'batch_size': 1, 'gradient_accumulation_steps': 8, 'enable_gradient_checkpointing': True}
```

### Condition-Specific Configurations

Pre-configured settings for different conditioning types:

```python
from controlnet_pipeline import get_depth_config, get_pose_config, get_edge_config

depth_config = get_depth_config()    # 1-channel grayscale depth maps
pose_config = get_pose_config()      # 3-channel RGB pose skeletons  
edge_config = get_edge_config()      # 1-channel grayscale edge maps
```

### YAML Configuration Files

Load custom configurations from YAML files:

```python
# Save configuration
config = BaseConfig()
config.save_config("my_config.yaml")

# Load configuration
config = BaseConfig.load_config("my_config.yaml")
```

Example YAML configuration:

```yaml
experiment_name: "controlnet_depth_experiment"
seed: 42

dataset:
  subset_size: 5000
  batch_size: 1
  condition_types: ["depth"]

model:
  conditioning_channels: 1
  mixed_precision: "fp16"
  enable_gradient_checkpointing: true

training:
  num_train_epochs: 50
  learning_rate: 1e-5
  gradient_accumulation_steps: 8
```

## 🚀 Quick Start

1. **Clone and setup**:
   ```bash
   git clone <repository>
   cd controlnet-training-pipeline
   python validate_setup.py
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Test configuration**:
   ```python
   from controlnet_pipeline import BaseConfig
   config = BaseConfig()
   print(f"Ready for training on {config.device}")
   ```

4. **Customize configuration**:
   ```python
   # For depth conditioning
   from controlnet_pipeline import get_depth_config
   config = get_depth_config()
   
   # For Colab optimization
   from controlnet_pipeline import get_colab_config
   config = get_colab_config()
   ```

## 🔍 Validation

Run the validation script to ensure everything is set up correctly:

```bash
python validate_setup.py
```

This will test:
- ✅ Project structure
- ✅ Requirements file format
- ✅ Configuration system functionality
- ✅ Setup.py package metadata
- ✅ YAML configuration parsing

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**: Ensure you've installed requirements and the package
2. **CUDA not available**: Configuration automatically falls back to CPU mode
3. **Memory errors**: Configuration automatically optimizes for available GPU memory
4. **Missing directories**: Directories are auto-created by the configuration system

### Environment-Specific Notes

- **Google Colab**: Optimized for T4 GPU with 15GB VRAM
- **Local Development**: Supports various GPU configurations
- **CPU-only**: Automatically disables GPU-specific optimizations

## 📚 Next Steps

After completing the setup:

1. **Dataset Processing**: Implement COCO dataset downloader and condition map extractors
2. **Model Architecture**: Implement ControlNet and UNet wrapper classes
3. **Training System**: Implement memory-optimized training loop
4. **Evaluation**: Implement FID scores and condition alignment metrics
5. **Web Demo**: Create Gradio interface for HuggingFace Spaces

The configuration system is now ready to support all these components with centralized, environment-aware settings management.