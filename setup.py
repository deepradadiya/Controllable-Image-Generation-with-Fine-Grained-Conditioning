"""
Setup script for ControlNet Training Pipeline

This setup script enables installation of the ControlNet training pipeline
as a Python package, making it easy to import and use across different
environments including Google Colab, Kaggle, and local development.

Installation:
    pip install -e .  # Development installation
    pip install .     # Standard installation

Usage:
    from controlnet_pipeline import BaseConfig, ControlNetTrainer
    from controlnet_pipeline.models import ControlNet
    from controlnet_pipeline.data import DatasetProcessor
"""

from setuptools import setup, find_packages
from pathlib import Path
import re

# Read version from __init__.py
def get_version():
    """Extract version from package __init__.py"""
    init_file = Path("src") / "__init__.py"
    if init_file.exists():
        with open(init_file, "r", encoding="utf-8") as f:
            content = f.read()
            version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", content, re.M)
            if version_match:
                return version_match.group(1)
    return "0.1.0"  # Default version

# Read long description from README
def get_long_description():
    """Get long description from README file"""
    readme_file = Path("README.md")
    if readme_file.exists():
        with open(readme_file, "r", encoding="utf-8") as f:
            return f.read()
    return "ControlNet Training Pipeline for Stable Diffusion 1.5"

# Read requirements from requirements.txt
def get_requirements():
    """Parse requirements from requirements.txt"""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        return []
    
    requirements = []
    with open(requirements_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Skip comments, empty lines, and extra index URLs
            if line and not line.startswith("#") and not line.startswith("--"):
                # Handle version specifiers with +cu118 suffix
                if "+cu" in line:
                    # Extract base package name and version
                    package = line.split("+")[0]
                    requirements.append(package)
                else:
                    requirements.append(line)
    
    return requirements

# Development requirements
dev_requirements = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.0.0",
    "flake8>=6.0.0",
    "mypy>=1.5.0",
    "pre-commit>=3.0.0",
    "jupyter>=1.0.0",
    "notebook>=6.5.0",
]

# Documentation requirements
docs_requirements = [
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.2.0",
    "myst-parser>=1.0.0",
    "sphinx-autodoc-typehints>=1.19.0",
]

# All extra requirements
all_requirements = dev_requirements + docs_requirements

setup(
    name="controlnet-training-pipeline",
    version=get_version(),
    author="ControlNet Pipeline Team",
    author_email="contact@controlnet-pipeline.com",
    description="Production-grade ControlNet training pipeline for Stable Diffusion 1.5",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/controlnet-pipeline/controlnet-training-pipeline",
    project_urls={
        "Bug Reports": "https://github.com/controlnet-pipeline/controlnet-training-pipeline/issues",
        "Source": "https://github.com/controlnet-pipeline/controlnet-training-pipeline",
        "Documentation": "https://controlnet-pipeline.readthedocs.io/",
        "HuggingFace Space": "https://huggingface.co/spaces/controlnet-pipeline/demo",
    },
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "controlnet_pipeline": [
            "configs/*.yaml",
            "configs/*.json",
            "assets/*.png",
            "assets/*.jpg",
        ],
    },
    
    # Requirements
    python_requires=">=3.8,<3.12",
    install_requires=get_requirements(),
    extras_require={
        "dev": dev_requirements,
        "docs": docs_requirements,
        "all": all_requirements,
    },
    
    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            "controlnet-train=controlnet_pipeline.cli.train:main",
            "controlnet-infer=controlnet_pipeline.cli.infer:main",
            "controlnet-eval=controlnet_pipeline.cli.eval:main",
            "controlnet-demo=controlnet_pipeline.cli.demo:main",
            "controlnet-process-data=controlnet_pipeline.cli.process_data:main",
        ],
    },
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
    ],
    
    # Keywords for PyPI search
    keywords=[
        "controlnet",
        "stable-diffusion",
        "diffusion-models",
        "image-generation",
        "deep-learning",
        "pytorch",
        "huggingface",
        "computer-vision",
        "ai",
        "machine-learning",
        "colab",
        "training-pipeline",
    ],
    
    # License
    license="Apache License 2.0",
    
    # Zip safety
    zip_safe=False,
    
    # Additional metadata
    platforms=["any"],
    
    # Options for different installation scenarios
    options={
        "bdist_wheel": {
            "universal": False,  # Not universal due to CUDA dependencies
        },
    },
)


# Post-installation setup for Colab environment
def setup_colab_environment():
    """Setup additional configurations for Google Colab"""
    try:
        import google.colab
        print("Google Colab detected. Setting up environment...")
        
        # Install additional Colab-specific packages
        import subprocess
        import sys
        
        colab_packages = [
            "google-colab",
            "google-auth",
            "google-auth-oauthlib",
            "google-auth-httplib2",
        ]
        
        for package in colab_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {package}")
        
        # Setup Google Drive integration
        try:
            from google.colab import drive
            print("✓ Google Drive integration available")
        except ImportError:
            print("✗ Google Drive integration not available")
        
        print("Colab environment setup complete!")
        
    except ImportError:
        # Not in Colab, skip setup
        pass


def setup_cuda_environment():
    """Verify and setup CUDA environment"""
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            
            print(f"✓ CUDA available: {cuda_version}")
            print(f"✓ GPU devices: {device_count}")
            print(f"✓ Primary GPU: {device_name}")
            
            # Check for T4 GPU and provide optimization suggestions
            if "T4" in device_name:
                print("🚀 T4 GPU detected - optimizations enabled:")
                print("  - Mixed precision training (FP16)")
                print("  - Gradient checkpointing")
                print("  - Memory-efficient attention (xFormers)")
                
        else:
            print("⚠️  CUDA not available - using CPU mode")
            print("   Training will be significantly slower")
            
    except ImportError:
        print("⚠️  PyTorch not installed - please install requirements first")


def verify_installation():
    """Verify that the installation was successful"""
    try:
        # Test basic imports
        from controlnet_pipeline.configs import BaseConfig
        print("✓ Configuration system imported successfully")
        
        # Test configuration creation
        config = BaseConfig()
        print(f"✓ Configuration created - Device: {config.device}")
        
        # Test directory creation
        import os
        required_dirs = ["data", "models", "outputs", "logs", "cache"]
        for dir_name in required_dirs:
            if os.path.exists(dir_name):
                print(f"✓ Directory created: {dir_name}")
            else:
                print(f"✗ Directory missing: {dir_name}")
        
        print("\n🎉 Installation verification complete!")
        print("You can now use: from controlnet_pipeline import BaseConfig")
        
    except ImportError as e:
        print(f"✗ Installation verification failed: {e}")
        print("Please check your installation and requirements")


if __name__ == "__main__":
    # Run post-installation setup when called directly
    print("Running post-installation setup...")
    setup_colab_environment()
    setup_cuda_environment()
    verify_installation()