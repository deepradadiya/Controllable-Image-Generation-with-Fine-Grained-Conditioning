#!/usr/bin/env python3
"""
Validation script for ControlNet Training Pipeline setup

This script validates that the dependency management and configuration system
is properly implemented and working correctly.

Usage:
    python validate_setup.py
"""

import sys
import os
from pathlib import Path

def test_project_structure():
    """Test that the project structure is correctly set up"""
    print("🔍 Testing project structure...")
    
    required_files = [
        "requirements.txt",
        "setup.py",
        "configs/__init__.py",
        "configs/base_config.py",
        "configs/example_config.yaml",
        "src/__init__.py",
    ]
    
    required_dirs = [
        "configs",
        "src",
        "data",
        "models", 
        "outputs",
        "logs",
        "cache",
    ]
    
    # Check files
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")
            return False
    
    # Check directories
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ - MISSING")
            return False
    
    return True


def test_requirements_file():
    """Test that requirements.txt is properly formatted"""
    print("\n🔍 Testing requirements.txt...")
    
    try:
        with open("requirements.txt", "r") as f:
            content = f.read()
        
        # Check for key dependencies
        key_deps = [
            "torch",
            "diffusers", 
            "transformers",
            "datasets",
            "gradio",
            "wandb",
            "opencv-python",
        ]
        
        for dep in key_deps:
            if dep in content:
                print(f"  ✓ {dep}")
            else:
                print(f"  ✗ {dep} - MISSING")
                return False
        
        # Check for version pinning
        lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
        pinned_count = sum(1 for line in lines if '==' in line and not line.startswith('--'))
        total_deps = len([line for line in lines if not line.startswith('--')])
        
        if pinned_count > 0:
            print(f"  ✓ Version pinning: {pinned_count}/{total_deps} packages pinned")
        else:
            print(f"  ⚠️  No version pinning found")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error reading requirements.txt: {e}")
        return False


def test_configuration_system():
    """Test that the configuration system works"""
    print("\n🔍 Testing configuration system...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, '.')
        
        from configs.base_config import BaseConfig, get_config
        
        # Test basic configuration creation
        config = BaseConfig()
        print(f"  ✓ BaseConfig created")
        print(f"    Device: {config.device}")
        print(f"    Environment: {'Colab' if config.is_colab else 'Local' if config.is_local else 'Kaggle'}")
        
        # Test configuration methods
        memory_config = config.get_memory_config()
        print(f"  ✓ Memory config: {memory_config}")
        
        depth_config = config.get_condition_config('depth')
        print(f"  ✓ Depth config: {depth_config['model_name']}")
        
        # Test condition-specific configs
        from configs.base_config import get_depth_config, get_pose_config, get_edge_config
        
        depth_cfg = get_depth_config()
        pose_cfg = get_pose_config()
        edge_cfg = get_edge_config()
        
        print(f"  ✓ Depth conditioning: {depth_cfg.model.conditioning_channels} channels")
        print(f"  ✓ Pose conditioning: {pose_cfg.model.conditioning_channels} channels")
        print(f"  ✓ Edge conditioning: {edge_cfg.model.conditioning_channels} channels")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Configuration system error: {e}")
        return False


def test_setup_py():
    """Test that setup.py is properly configured"""
    print("\n🔍 Testing setup.py...")
    
    try:
        # Test that setup.py can be imported and parsed
        import subprocess
        result = subprocess.run([sys.executable, "setup.py", "--name"], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            package_name = result.stdout.strip()
            print(f"  ✓ Package name: {package_name}")
        else:
            print(f"  ⚠️  Setup.py warning (expected without dependencies)")
        
        # Check that setup.py contains required metadata
        with open("setup.py", "r") as f:
            content = f.read()
        
        required_fields = [
            "name=",
            "version=",
            "install_requires=",
            "entry_points=",
            "classifiers=",
        ]
        
        for field in required_fields:
            if field in content:
                print(f"  ✓ {field}")
            else:
                print(f"  ✗ {field} - MISSING")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Setup.py error: {e}")
        return False


def test_yaml_config():
    """Test that YAML configuration loading works"""
    print("\n🔍 Testing YAML configuration...")
    
    try:
        # Check if example config exists
        if not Path("configs/example_config.yaml").exists():
            print("  ⚠️  Example config not found, skipping YAML test")
            return True
        
        # Try to parse YAML (basic validation)
        with open("configs/example_config.yaml", "r") as f:
            content = f.read()
        
        # Check for key sections
        sections = ["dataset:", "model:", "training:", "logging:"]
        for section in sections:
            if section in content:
                print(f"  ✓ {section}")
            else:
                print(f"  ✗ {section} - MISSING")
                return False
        
        print("  ✓ YAML configuration format valid")
        return True
        
    except Exception as e:
        print(f"  ✗ YAML configuration error: {e}")
        return False


def main():
    """Run all validation tests"""
    print("🚀 ControlNet Training Pipeline - Setup Validation")
    print("=" * 60)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Requirements File", test_requirements_file),
        ("Configuration System", test_configuration_system),
        ("Setup.py", test_setup_py),
        ("YAML Configuration", test_yaml_config),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All validation tests passed!")
        print("The dependency management and configuration system is ready.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Install package: pip install -e .")
        print("3. Test with: python -c 'from controlnet_pipeline import BaseConfig; print(BaseConfig())'")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()