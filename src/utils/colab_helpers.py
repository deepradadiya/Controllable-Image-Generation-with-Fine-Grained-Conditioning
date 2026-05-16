"""
Google Colab Utilities and Helpers

This module provides utilities specifically designed for Google Colab environments,
including Google Drive integration, session management, GPU monitoring, and
automatic checkpoint saving with recovery suggestions.

Requirements satisfied: 6.1, 6.2, 6.3, 6.5
"""

import os
import time
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import warnings

import torch
import psutil

logger = logging.getLogger(__name__)


class ColabEnvironmentDetector:
    """Detect if running in Google Colab environment."""
    
    @staticmethod
    def is_colab() -> bool:
        """Check if running in Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    @staticmethod
    def is_kaggle() -> bool:
        """Check if running in Kaggle environment."""
        return os.path.exists('/kaggle')
    
    @staticmethod
    def get_environment() -> str:
        """Get the current environment type."""
        if ColabEnvironmentDetector.is_colab():
            return "colab"
        elif ColabEnvironmentDetector.is_kaggle():
            return "kaggle"
        else:
            return "local"


class GoogleDriveManager:
    """Manage Google Drive integration for persistent storage."""
    
    def __init__(self, mount_point: str = "/content/drive"):
        """
        Initialize Google Drive manager.
        
        Args:
            mount_point: Mount point for Google Drive
        """
        self.mount_point = mount_point
        self.is_mounted = False
        
    def mount_drive(self, force_remount: bool = False) -> bool:
        """
        Mount Google Drive in Colab.
        
        Args:
            force_remount: Force remount even if already mounted
            
        Returns:
            True if successful, False otherwise
        """
        if not ColabEnvironmentDetector.is_colab():
            logger.warning("Not in Colab environment, skipping Drive mount")
            return False
        
        if self.is_mounted and not force_remount:
            logger.info("Google Drive already mounted")
            return True
        
        try:
            from google.colab import drive
            
            logger.info(f"Mounting Google Drive at {self.mount_point}...")
            drive.mount(self.mount_point, force_remount=force_remount)
            
            # Verify mount
            if os.path.exists(self.mount_point):
                self.is_mounted = True
                logger.info("Google Drive mounted successfully")
                return True
            else:
                logger.error("Google Drive mount failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to mount Google Drive: {e}")
            return False
    
    def get_drive_path(self, relative_path: str) -> Path:
        """
        Get full path in Google Drive.
        
        Args:
            relative_path: Relative path from Drive root
            
        Returns:
            Full path to file/directory in Drive
        """
        if not self.is_mounted:
            raise RuntimeError("Google Drive not mounted")
        
        return Path(self.mount_point) / "MyDrive" / relative_path
    
    def create_project_folder(self, project_name: str) -> Path:
        """
        Create project folder in Google Drive.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Path to created project folder
        """
        project_path = self.get_drive_path(f"ControlNet_Projects/{project_name}")
        project_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Project folder created: {project_path}")
        return project_path
    
    def sync_to_drive(self, local_path: Path, drive_path: Path) -> bool:
        """
        Sync local files to Google Drive.
        
        Args:
            local_path: Local file or directory path
            drive_path: Target path in Google Drive
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if local_path.is_file():
                # Copy single file
                drive_path.parent.mkdir(parents=True, exist_ok=True)
                import shutil
                shutil.copy2(local_path, drive_path)
                logger.info(f"Synced file: {local_path} -> {drive_path}")
                
            elif local_path.is_dir():
                # Copy directory
                import shutil
                if drive_path.exists():
                    shutil.rmtree(drive_path)
                shutil.copytree(local_path, drive_path)
                logger.info(f"Synced directory: {local_path} -> {drive_path}")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to sync to Drive: {e}")
            return False


class ColabSessionManager:
    """Manage Colab session lifecycle and warnings."""
    
    def __init__(self, session_limit_hours: float = 12.0):
        """
        Initialize session manager.
        
        Args:
            session_limit_hours: Maximum session duration in hours
        """
        self.session_limit_hours = session_limit_hours
        self.session_start_time = datetime.now()
        self.warning_intervals = [0.5, 1.0, 2.0]  # Hours before limit to warn
        self.last_warning_time = None
        
    def get_session_duration(self) -> timedelta:
        """Get current session duration."""
        return datetime.now() - self.session_start_time
    
    def get_remaining_time(self) -> timedelta:
        """Get remaining session time."""
        elapsed = self.get_session_duration()
        limit = timedelta(hours=self.session_limit_hours)
        return limit - elapsed
    
    def should_warn(self) -> bool:
        """Check if should issue session warning."""
        remaining = self.get_remaining_time()
        
        for warning_hours in self.warning_intervals:
            warning_threshold = timedelta(hours=warning_hours)
            
            if remaining <= warning_threshold:
                # Check if we haven't warned for this threshold recently
                if (self.last_warning_time is None or 
                    datetime.now() - self.last_warning_time > timedelta(minutes=30)):
                    return True
        
        return False
    
    def issue_warning(self) -> str:
        """Issue session warning and return message."""
        remaining = self.get_remaining_time()
        self.last_warning_time = datetime.now()
        
        if remaining.total_seconds() <= 0:
            message = "⚠️ CRITICAL: Colab session time exceeded! Save your work immediately!"
        elif remaining.total_seconds() < 3600:  # Less than 1 hour
            minutes = int(remaining.total_seconds() / 60)
            message = f"⚠️ WARNING: Only {minutes} minutes remaining in Colab session!"
        else:
            hours = remaining.total_seconds() / 3600
            message = f"⚠️ NOTICE: {hours:.1f} hours remaining in Colab session"
        
        logger.warning(message)
        
        # Display in Colab if available
        if ColabEnvironmentDetector.is_colab():
            try:
                from IPython.display import display, HTML
                display(HTML(f'<div style="color: orange; font-weight: bold;">{message}</div>'))
            except ImportError:
                pass
        
        return message
    
    def check_and_warn(self) -> Optional[str]:
        """Check session time and warn if necessary."""
        if self.should_warn():
            return self.issue_warning()
        return None


class GPUMonitor:
    """Monitor GPU usage and provide OOM handling."""
    
    def __init__(self):
        """Initialize GPU monitor."""
        self.gpu_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.gpu_available else 0
        
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get comprehensive GPU information."""
        if not self.gpu_available:
            return {"available": False, "message": "No GPU available"}
        
        info = {"available": True, "devices": []}
        
        for i in range(self.device_count):
            device_props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_reserved = torch.cuda.memory_reserved(i)
            memory_total = device_props.total_memory
            
            device_info = {
                "device_id": i,
                "name": device_props.name,
                "compute_capability": f"{device_props.major}.{device_props.minor}",
                "total_memory_gb": memory_total / (1024**3),
                "allocated_memory_gb": memory_allocated / (1024**3),
                "reserved_memory_gb": memory_reserved / (1024**3),
                "free_memory_gb": (memory_total - memory_reserved) / (1024**3),
                "utilization_percent": (memory_reserved / memory_total) * 100,
            }
            
            info["devices"].append(device_info)
        
        return info
    
    def get_memory_summary(self) -> str:
        """Get formatted memory summary."""
        if not self.gpu_available:
            return "No GPU available"
        
        info = self.get_gpu_info()
        summary_lines = []
        
        for device in info["devices"]:
            summary_lines.append(
                f"GPU {device['device_id']} ({device['name']}): "
                f"{device['allocated_memory_gb']:.1f}GB / {device['total_memory_gb']:.1f}GB "
                f"({device['utilization_percent']:.1f}% used)"
            )
        
        return "\n".join(summary_lines)
    
    def check_memory_pressure(self, threshold_percent: float = 85.0) -> bool:
        """Check if GPU memory pressure is high."""
        if not self.gpu_available:
            return False
        
        info = self.get_gpu_info()
        
        for device in info["devices"]:
            if device["utilization_percent"] > threshold_percent:
                return True
        
        return False
    
    def suggest_memory_optimization(self) -> List[str]:
        """Suggest memory optimization strategies."""
        suggestions = []
        
        if not self.gpu_available:
            suggestions.append("Consider using CPU training with smaller models")
            return suggestions
        
        info = self.get_gpu_info()
        
        for device in info["devices"]:
            if device["utilization_percent"] > 90:
                suggestions.extend([
                    "🔥 CRITICAL: GPU memory almost full!",
                    "• Reduce batch size immediately",
                    "• Enable gradient checkpointing",
                    "• Use mixed precision (FP16)",
                    "• Clear GPU cache: torch.cuda.empty_cache()",
                ])
            elif device["utilization_percent"] > 75:
                suggestions.extend([
                    "⚠️ HIGH: GPU memory pressure detected",
                    "• Consider reducing batch size",
                    "• Enable gradient accumulation",
                    "• Monitor memory usage closely",
                ])
            elif device["utilization_percent"] > 50:
                suggestions.extend([
                    "✓ MODERATE: GPU memory usage is reasonable",
                    "• Current usage is sustainable",
                    "• Consider slight batch size increase if needed",
                ])
            else:
                suggestions.extend([
                    "✓ LOW: GPU memory usage is low",
                    "• You can increase batch size",
                    "• Consider using larger models",
                ])
        
        return suggestions


class AutoCheckpointManager:
    """Automatic checkpoint management for Colab sessions."""
    
    def __init__(
        self,
        checkpoint_dir: Path,
        drive_manager: Optional[GoogleDriveManager] = None,
        checkpoint_interval_minutes: int = 30,
        max_checkpoints: int = 5,
    ):
        """
        Initialize auto checkpoint manager.
        
        Args:
            checkpoint_dir: Local checkpoint directory
            drive_manager: Google Drive manager for backup
            checkpoint_interval_minutes: Interval between checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.drive_manager = drive_manager
        self.checkpoint_interval = timedelta(minutes=checkpoint_interval_minutes)
        self.max_checkpoints = max_checkpoints
        self.last_checkpoint_time = datetime.now()
        
    def should_checkpoint(self) -> bool:
        """Check if it's time for a checkpoint."""
        return datetime.now() - self.last_checkpoint_time >= self.checkpoint_interval
    
    def create_checkpoint(
        self,
        model_state: Dict[str, Any],
        step: int,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Create a checkpoint.
        
        Args:
            model_state: Model state dictionary
            step: Current training step
            additional_data: Additional data to save
            
        Returns:
            Path to created checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_step_{step}_{timestamp}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        # Prepare checkpoint data
        checkpoint_data = {
            "step": step,
            "timestamp": timestamp,
            "model_state": model_state,
        }
        
        if additional_data:
            checkpoint_data.update(additional_data)
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        self.last_checkpoint_time = datetime.now()
        
        logger.info(f"Checkpoint created: {checkpoint_path}")
        
        # Backup to Google Drive if available
        if self.drive_manager and self.drive_manager.is_mounted:
            drive_checkpoint_path = self.drive_manager.get_drive_path(
                f"ControlNet_Checkpoints/{checkpoint_name}"
            )
            self.drive_manager.sync_to_drive(checkpoint_path, drive_checkpoint_path)
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save space."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Keep only the most recent checkpoints
        for old_checkpoint in checkpoints[self.max_checkpoints:]:
            old_checkpoint.unlink()
            logger.info(f"Removed old checkpoint: {old_checkpoint}")
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        
        if not checkpoints:
            return None
        
        # Sort by modification time, most recent first
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return checkpoints[0]
    
    def load_checkpoint(self, checkpoint_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint data.
        
        Args:
            checkpoint_path: Specific checkpoint to load (latest if None)
            
        Returns:
            Checkpoint data or None if not found
        """
        if checkpoint_path is None:
            checkpoint_path = self.find_latest_checkpoint()
        
        if checkpoint_path is None or not checkpoint_path.exists():
            logger.warning("No checkpoint found")
            return None
        
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None


class ColabTrainingHelper:
    """Comprehensive helper for ControlNet training in Colab."""
    
    def __init__(
        self,
        project_name: str = "controlnet_training",
        auto_mount_drive: bool = True,
        checkpoint_interval_minutes: int = 30,
    ):
        """
        Initialize Colab training helper.
        
        Args:
            project_name: Name of the training project
            auto_mount_drive: Whether to automatically mount Google Drive
            checkpoint_interval_minutes: Checkpoint interval in minutes
        """
        self.project_name = project_name
        self.environment = ColabEnvironmentDetector.get_environment()
        
        # Initialize components
        self.drive_manager = GoogleDriveManager()
        self.session_manager = ColabSessionManager()
        self.gpu_monitor = GPUMonitor()
        
        # Setup project directories
        if self.environment == "colab":
            self.local_project_dir = Path(f"/content/{project_name}")
        else:
            # Use current directory for local testing
            self.local_project_dir = Path(f"./{project_name}")
        self.local_project_dir.mkdir(exist_ok=True)
        
        # Mount Google Drive if requested and in Colab
        if auto_mount_drive and self.environment == "colab":
            self.drive_manager.mount_drive()
            if self.drive_manager.is_mounted:
                self.drive_project_dir = self.drive_manager.create_project_folder(project_name)
            else:
                self.drive_project_dir = None
        else:
            self.drive_project_dir = None
        
        # Setup checkpoint manager
        checkpoint_dir = self.local_project_dir / "checkpoints"
        self.checkpoint_manager = AutoCheckpointManager(
            checkpoint_dir=checkpoint_dir,
            drive_manager=self.drive_manager if self.drive_manager.is_mounted else None,
            checkpoint_interval_minutes=checkpoint_interval_minutes,
        )
        
        logger.info(f"Colab training helper initialized for {project_name}")
        logger.info(f"Environment: {self.environment}")
        logger.info(f"Local project dir: {self.local_project_dir}")
        if self.drive_project_dir:
            logger.info(f"Drive project dir: {self.drive_project_dir}")
    
    def setup_training_environment(self) -> Dict[str, Any]:
        """Setup complete training environment."""
        logger.info("Setting up Colab training environment...")
        
        # Check GPU
        gpu_info = self.gpu_monitor.get_gpu_info()
        logger.info("GPU Information:")
        logger.info(self.gpu_monitor.get_memory_summary())
        
        # Check session time
        session_warning = self.session_manager.check_and_warn()
        
        # Setup directories
        dirs_created = {
            "models": self.local_project_dir / "models",
            "logs": self.local_project_dir / "logs",
            "outputs": self.local_project_dir / "outputs",
            "checkpoints": self.local_project_dir / "checkpoints",
        }
        
        for name, path in dirs_created.items():
            path.mkdir(exist_ok=True)
        
        # Environment summary
        setup_info = {
            "environment": self.environment,
            "gpu_available": gpu_info["available"],
            "gpu_devices": len(gpu_info.get("devices", [])),
            "drive_mounted": self.drive_manager.is_mounted,
            "local_project_dir": str(self.local_project_dir),
            "drive_project_dir": str(self.drive_project_dir) if self.drive_project_dir else None,
            "session_warning": session_warning,
            "directories_created": {k: str(v) for k, v in dirs_created.items()},
        }
        
        logger.info("Training environment setup complete")
        return setup_info
    
    def monitor_training(self, step: int, loss: float) -> Dict[str, Any]:
        """Monitor training progress and handle warnings."""
        monitoring_info = {
            "step": step,
            "loss": loss,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Check session time
        session_warning = self.session_manager.check_and_warn()
        if session_warning:
            monitoring_info["session_warning"] = session_warning
        
        # Check GPU memory
        if self.gpu_monitor.check_memory_pressure():
            memory_suggestions = self.gpu_monitor.suggest_memory_optimization()
            monitoring_info["memory_warning"] = memory_suggestions
            logger.warning("GPU memory pressure detected!")
            for suggestion in memory_suggestions[:3]:  # Show top 3 suggestions
                logger.warning(suggestion)
        
        # Check if checkpoint needed
        if self.checkpoint_manager.should_checkpoint():
            monitoring_info["checkpoint_needed"] = True
        
        return monitoring_info
    
    def save_training_checkpoint(
        self,
        model_state: Dict[str, Any],
        step: int,
        loss: float,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save training checkpoint with metadata."""
        checkpoint_data = {
            "loss": loss,
            "gpu_info": self.gpu_monitor.get_gpu_info(),
            "session_duration": str(self.session_manager.get_session_duration()),
        }
        
        if additional_data:
            checkpoint_data.update(additional_data)
        
        return self.checkpoint_manager.create_checkpoint(
            model_state=model_state,
            step=step,
            additional_data=checkpoint_data,
        )
    
    def get_recovery_suggestions(self) -> List[str]:
        """Get recovery suggestions for common Colab issues."""
        suggestions = []
        
        # Session time suggestions
        remaining_time = self.session_manager.get_remaining_time()
        if remaining_time.total_seconds() < 3600:  # Less than 1 hour
            suggestions.extend([
                "🕐 Session Time Critical:",
                "• Save checkpoint immediately",
                "• Backup to Google Drive",
                "• Prepare to restart session",
            ])
        
        # GPU memory suggestions
        if self.gpu_monitor.check_memory_pressure():
            suggestions.extend(self.gpu_monitor.suggest_memory_optimization())
        
        # Drive backup suggestions
        if not self.drive_manager.is_mounted:
            suggestions.extend([
                "💾 Backup Recommendations:",
                "• Mount Google Drive for persistent storage",
                "• Enable automatic checkpoint backup",
            ])
        
        return suggestions


def main():
    """Test Colab utilities."""
    print("Testing Colab Utilities")
    print("=" * 30)
    
    # Test environment detection
    env = ColabEnvironmentDetector.get_environment()
    print(f"Environment: {env}")
    
    # Test GPU monitoring
    gpu_monitor = GPUMonitor()
    print(f"GPU Info:")
    print(gpu_monitor.get_memory_summary())
    
    # Test session management
    session_manager = ColabSessionManager(session_limit_hours=0.1)  # 6 minutes for testing
    print(f"Session duration: {session_manager.get_session_duration()}")
    print(f"Remaining time: {session_manager.get_remaining_time()}")
    
    # Test training helper
    helper = ColabTrainingHelper(
        project_name="test_controlnet",
        auto_mount_drive=False,  # Don't mount for testing
    )
    
    setup_info = helper.setup_training_environment()
    print(f"Setup info: {setup_info}")
    
    # Test monitoring
    monitoring_info = helper.monitor_training(step=100, loss=0.5)
    print(f"Monitoring info: {monitoring_info}")
    
    print("\n✅ Colab utilities test completed!")
    print("\nKey features implemented:")
    print("✓ Environment detection (Colab/Kaggle/Local)")
    print("✓ Google Drive integration and mounting")
    print("✓ Session time monitoring and warnings")
    print("✓ GPU memory monitoring and OOM suggestions")
    print("✓ Automatic checkpoint management")
    print("✓ Training environment setup")
    print("✓ Recovery suggestions")
    
    print(f"\n📋 Task 6.1 Implementation Complete!")
    print(f"Requirements satisfied: 6.1, 6.2, 6.3, 6.5")


if __name__ == "__main__":
    main()