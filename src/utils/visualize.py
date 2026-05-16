"""
Training Visualization and Monitoring

This module provides comprehensive training visualization and monitoring utilities
for the ControlNet training pipeline. It includes loss curve plotting, sample
generation logging, Weights & Biases integration, GPU memory tracking, and
training speed metrics.

Requirements satisfied: 12.1, 12.2, 12.3, 12.4
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server/Colab
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics at a single step."""

    step: int
    loss: float
    learning_rate: float
    epoch: Optional[int] = None
    validation_loss: Optional[float] = None
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_reserved_mb: float = 0.0
    steps_per_second: float = 0.0
    samples_per_second: float = 0.0
    grad_norm: Optional[float] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        d = {
            "step": self.step,
            "train/loss": self.loss,
            "train/learning_rate": self.learning_rate,
            "performance/steps_per_second": self.steps_per_second,
            "performance/samples_per_second": self.samples_per_second,
            "memory/gpu_allocated_mb": self.gpu_memory_allocated_mb,
            "memory/gpu_reserved_mb": self.gpu_memory_reserved_mb,
        }
        if self.epoch is not None:
            d["train/epoch"] = self.epoch
        if self.validation_loss is not None:
            d["eval/validation_loss"] = self.validation_loss
        if self.grad_norm is not None:
            d["train/grad_norm"] = self.grad_norm
        return d


class SpeedTracker:
    """Track training speed metrics (steps/sec, samples/sec).

    Maintains a rolling window of step timestamps to compute
    smoothed throughput metrics.
    """

    def __init__(self, window_size: int = 50):
        """
        Initialize speed tracker.

        Args:
            window_size: Number of recent steps to use for speed calculation
        """
        self.window_size = window_size
        self._timestamps: deque = deque(maxlen=window_size)
        self._batch_sizes: deque = deque(maxlen=window_size)
        self._start_time: Optional[float] = None
        self._total_steps: int = 0
        self._total_samples: int = 0

    def step(self, batch_size: int = 1) -> None:
        """
        Record a training step.

        Args:
            batch_size: Number of samples in this step's batch
        """
        now = time.time()
        if self._start_time is None:
            self._start_time = now

        self._timestamps.append(now)
        self._batch_sizes.append(batch_size)
        self._total_steps += 1
        self._total_samples += batch_size

    def get_steps_per_second(self) -> float:
        """
        Get smoothed steps per second over the rolling window.

        Returns:
            Steps per second (0.0 if insufficient data)
        """
        if len(self._timestamps) < 2:
            return 0.0

        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0

        return (len(self._timestamps) - 1) / elapsed

    def get_samples_per_second(self) -> float:
        """
        Get smoothed samples per second over the rolling window.

        Returns:
            Samples per second (0.0 if insufficient data)
        """
        if len(self._timestamps) < 2:
            return 0.0

        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0

        total_samples_in_window = sum(self._batch_sizes) - self._batch_sizes[0]
        return total_samples_in_window / elapsed

    def get_total_elapsed(self) -> float:
        """Get total elapsed time since first step in seconds."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def get_eta(self, total_steps: int) -> float:
        """
        Estimate time remaining based on current speed.

        Args:
            total_steps: Total number of steps planned

        Returns:
            Estimated seconds remaining
        """
        sps = self.get_steps_per_second()
        if sps <= 0:
            return float('inf')

        remaining_steps = total_steps - self._total_steps
        return remaining_steps / sps


class GPUMetricsTracker:
    """Track GPU memory usage and utilization over time.

    Records GPU memory snapshots and provides summary statistics
    for monitoring training resource usage.
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize GPU metrics tracker.

        Args:
            device: CUDA device to monitor (default: current device)
        """
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.gpu_available = torch.cuda.is_available()
        self._memory_history: List[Dict[str, float]] = []

    def snapshot(self) -> Dict[str, float]:
        """
        Take a GPU memory snapshot.

        Returns:
            Dictionary with memory stats in MB
        """
        if not self.gpu_available:
            return {
                "allocated_mb": 0.0,
                "reserved_mb": 0.0,
                "free_mb": 0.0,
                "total_mb": 0.0,
                "utilization_percent": 0.0,
            }

        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
        total = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 2)
        free = total - reserved

        snapshot_data = {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "free_mb": free,
            "total_mb": total,
            "utilization_percent": (reserved / total) * 100 if total > 0 else 0.0,
        }

        self._memory_history.append(snapshot_data)
        return snapshot_data

    def get_peak_memory_mb(self) -> float:
        """Get peak allocated GPU memory in MB."""
        if not self.gpu_available:
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)

    def get_memory_summary(self) -> Dict[str, float]:
        """
        Get summary statistics of GPU memory usage.

        Returns:
            Dictionary with min, max, mean, and current memory stats
        """
        if not self._memory_history:
            return {"status": "no_data"}

        allocated_values = [s["allocated_mb"] for s in self._memory_history]

        return {
            "current_allocated_mb": allocated_values[-1],
            "peak_allocated_mb": max(allocated_values),
            "mean_allocated_mb": np.mean(allocated_values),
            "min_allocated_mb": min(allocated_values),
            "num_snapshots": len(self._memory_history),
        }



class TrainingVisualizer:
    """
    Comprehensive training visualization and monitoring system.

    Provides loss curve plotting with smoothing, sample generation logging,
    Weights & Biases integration, GPU memory tracking, and training speed
    metrics. Supports both local file saving and W&B cloud logging.

    Requirements:
        - 12.1: Plot training loss curves with proper axis labels and legends
        - 12.2: Log sample generations at regular intervals during training
        - 12.3: Integrate with Weights & Biases for experiment tracking
        - 12.4: Log GPU memory usage and training speed metrics
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        experiment_name: str = "controlnet_training",
        use_wandb: bool = True,
        wandb_project: str = "controlnet-training",
        wandb_config: Optional[Dict[str, Any]] = None,
        sample_log_interval: int = 500,
        plot_interval: int = 100,
    ):
        """
        Initialize the training visualizer.

        Args:
            output_dir: Directory for saving plots and sample images
            experiment_name: Name of the experiment for logging
            use_wandb: Whether to enable Weights & Biases logging
            wandb_project: W&B project name
            wandb_config: Configuration dict to log to W&B
            sample_log_interval: Steps between sample generation logs
            plot_interval: Steps between loss curve plot updates
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        self.samples_dir = self.output_dir / "samples"
        self.samples_dir.mkdir(exist_ok=True)

        self.experiment_name = experiment_name
        self.sample_log_interval = sample_log_interval
        self.plot_interval = plot_interval

        # Metrics storage
        self._loss_history: List[float] = []
        self._val_loss_history: List[float] = []
        self._lr_history: List[float] = []
        self._steps: List[int] = []
        self._val_steps: List[int] = []
        self._metrics_history: List[TrainingMetrics] = []

        # Sub-trackers
        self.speed_tracker = SpeedTracker()
        self.gpu_tracker = GPUMetricsTracker()

        # W&B integration
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self._wandb_initialized = False
        if self.use_wandb:
            self._init_wandb(wandb_project, wandb_config)

        logger.info(
            f"TrainingVisualizer initialized: output_dir={self.output_dir}, "
            f"wandb={'enabled' if self._wandb_initialized else 'disabled'}"
        )

    def _init_wandb(
        self,
        project: str,
        config: Optional[Dict[str, Any]],
    ) -> None:
        """
        Initialize Weights & Biases experiment tracking.

        Args:
            project: W&B project name
            config: Configuration to log
        """
        if not WANDB_AVAILABLE:
            logger.warning("wandb not installed. Install with: pip install wandb")
            self.use_wandb = False
            return

        try:
            wandb.init(
                project=project,
                name=self.experiment_name,
                config=config or {},
                dir=str(self.output_dir),
                reinit=True,
            )
            self._wandb_initialized = True
            logger.info(f"W&B initialized: project={project}, run={self.experiment_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}. Continuing without W&B.")
            self.use_wandb = False
            self._wandb_initialized = False

    def log_step(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        batch_size: int = 1,
        epoch: Optional[int] = None,
        validation_loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> TrainingMetrics:
        """
        Log metrics for a single training step.

        This is the primary method to call at each training step. It records
        loss, learning rate, speed, and GPU metrics, then logs to W&B if enabled.

        Args:
            step: Current global training step
            loss: Training loss value
            learning_rate: Current learning rate
            batch_size: Batch size for this step
            epoch: Current epoch number
            validation_loss: Validation loss (if computed this step)
            grad_norm: Gradient norm (if computed this step)
            extra_metrics: Additional metrics to log

        Returns:
            TrainingMetrics object with all recorded metrics
        """
        # Track speed
        self.speed_tracker.step(batch_size)

        # Track GPU memory
        gpu_snapshot = self.gpu_tracker.snapshot()

        # Build metrics object
        metrics = TrainingMetrics(
            step=step,
            loss=loss,
            learning_rate=learning_rate,
            epoch=epoch,
            validation_loss=validation_loss,
            gpu_memory_allocated_mb=gpu_snapshot["allocated_mb"],
            gpu_memory_reserved_mb=gpu_snapshot["reserved_mb"],
            steps_per_second=self.speed_tracker.get_steps_per_second(),
            samples_per_second=self.speed_tracker.get_samples_per_second(),
            grad_norm=grad_norm,
        )

        # Store history
        self._loss_history.append(loss)
        self._steps.append(step)
        self._lr_history.append(learning_rate)
        self._metrics_history.append(metrics)

        if validation_loss is not None:
            self._val_loss_history.append(validation_loss)
            self._val_steps.append(step)

        # Log to W&B
        if self._wandb_initialized:
            log_dict = metrics.to_dict()
            if extra_metrics:
                log_dict.update(extra_metrics)
            wandb.log(log_dict, step=step)

        # Periodically save loss curve plot
        if step > 0 and step % self.plot_interval == 0:
            self.plot_loss_curves(save=True)

        return metrics

    def log_sample_images(
        self,
        step: int,
        images: List[Union["Image.Image", np.ndarray, torch.Tensor]],
        condition_maps: Optional[List[Union["Image.Image", np.ndarray]]] = None,
        prompts: Optional[List[str]] = None,
        prefix: str = "sample",
    ) -> Path:
        """
        Log generated sample images during training.

        Saves a grid of generated images locally and logs to W&B.
        Call this at regular intervals to monitor generation quality.

        Args:
            step: Current training step
            images: List of generated images (PIL, numpy, or tensor)
            condition_maps: Optional list of condition maps used for generation
            prompts: Optional list of text prompts used
            prefix: Filename prefix for saved images

        Returns:
            Path to the saved image grid
        """
        if not PIL_AVAILABLE:
            logger.warning("Pillow not available, skipping sample image logging")
            return self.samples_dir

        # Convert images to PIL
        pil_images = [self._to_pil(img) for img in images]

        # Create image grid
        grid = self._create_image_grid(pil_images, condition_maps)

        # Save locally
        save_path = self.samples_dir / f"{prefix}_step_{step:07d}.png"
        grid.save(save_path)
        logger.info(f"Sample images saved: {save_path}")

        # Log to W&B
        if self._wandb_initialized:
            wandb_images = []
            for i, img in enumerate(pil_images):
                caption = prompts[i] if prompts and i < len(prompts) else f"Sample {i}"
                wandb_images.append(wandb.Image(img, caption=caption))

            wandb.log({"samples/generated": wandb_images}, step=step)

            # Log condition maps if provided
            if condition_maps:
                cond_pil = [self._to_pil(c) for c in condition_maps]
                wandb_conds = [
                    wandb.Image(c, caption=f"Condition {i}")
                    for i, c in enumerate(cond_pil)
                ]
                wandb.log({"samples/conditions": wandb_conds}, step=step)

        return save_path

    def plot_loss_curves(
        self,
        save: bool = True,
        show: bool = False,
        smoothing_weight: float = 0.9,
    ) -> Optional[Path]:
        """
        Plot training and validation loss curves with smoothing.

        Creates a publication-quality plot with proper axis labels, legends,
        and optional exponential moving average smoothing.

        Args:
            save: Whether to save the plot to disk
            show: Whether to display the plot (for notebooks)
            smoothing_weight: EMA smoothing factor (0=no smoothing, 1=max smoothing)

        Returns:
            Path to saved plot file, or None if not saved
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available, skipping loss curve plot")
            return None

        if len(self._loss_history) < 2:
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Training Metrics - {self.experiment_name}",
            fontsize=14,
            fontweight="bold",
        )

        # --- Plot 1: Training Loss ---
        ax = axes[0, 0]
        ax.plot(
            self._steps,
            self._loss_history,
            alpha=0.3,
            color="blue",
            linewidth=0.5,
            label="Raw loss",
        )

        # Smoothed loss
        smoothed = self._exponential_moving_average(
            self._loss_history, smoothing_weight
        )
        ax.plot(
            self._steps,
            smoothed,
            color="blue",
            linewidth=2,
            label=f"Smoothed (α={smoothing_weight})",
        )

        # Validation loss
        if self._val_loss_history:
            ax.plot(
                self._val_steps,
                self._val_loss_history,
                color="red",
                linewidth=2,
                marker="o",
                markersize=3,
                label="Validation loss",
            )

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # --- Plot 2: Learning Rate ---
        ax = axes[0, 1]
        ax.plot(self._steps, self._lr_history, color="green", linewidth=1.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

        # --- Plot 3: GPU Memory ---
        ax = axes[1, 0]
        if self.gpu_tracker._memory_history:
            mem_steps = self._steps[-len(self.gpu_tracker._memory_history):]
            allocated = [
                s["allocated_mb"] for s in self.gpu_tracker._memory_history
            ]
            reserved = [
                s["reserved_mb"] for s in self.gpu_tracker._memory_history
            ]

            # Align lengths
            min_len = min(len(mem_steps), len(allocated))
            ax.plot(
                mem_steps[:min_len],
                allocated[:min_len],
                color="purple",
                linewidth=1.5,
                label="Allocated",
            )
            ax.plot(
                mem_steps[:min_len],
                reserved[:min_len],
                color="orange",
                linewidth=1.5,
                alpha=0.7,
                label="Reserved",
            )
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Memory (MB)")
            ax.set_title("GPU Memory Usage")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5, 0.5, "No GPU data available",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_title("GPU Memory Usage")

        # --- Plot 4: Training Speed ---
        ax = axes[1, 1]
        if len(self._metrics_history) > 1:
            speed_steps = [m.step for m in self._metrics_history]
            sps = [m.steps_per_second for m in self._metrics_history]
            samples_ps = [m.samples_per_second for m in self._metrics_history]

            ax.plot(
                speed_steps,
                sps,
                color="teal",
                linewidth=1.5,
                label="Steps/sec",
            )
            ax2 = ax.twinx()
            ax2.plot(
                speed_steps,
                samples_ps,
                color="coral",
                linewidth=1.5,
                label="Samples/sec",
            )
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Steps/sec", color="teal")
            ax2.set_ylabel("Samples/sec", color="coral")
            ax.set_title("Training Speed")

            # Combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(
                0.5, 0.5, "Insufficient data",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_title("Training Speed")

        plt.tight_layout()

        save_path = None
        if save:
            save_path = self.plots_dir / "training_metrics.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.debug(f"Loss curve plot saved: {save_path}")

            # Also log to W&B
            if self._wandb_initialized:
                wandb.log({"plots/training_metrics": wandb.Image(str(save_path))})

        if show:
            plt.show()
        else:
            plt.close(fig)

        return save_path

    def plot_gpu_memory(self, save: bool = True) -> Optional[Path]:
        """
        Plot detailed GPU memory usage over time.

        Args:
            save: Whether to save the plot

        Returns:
            Path to saved plot, or None
        """
        if not MATPLOTLIB_AVAILABLE:
            return None

        if not self.gpu_tracker._memory_history:
            logger.warning("No GPU memory data to plot")
            return None

        fig, ax = plt.subplots(figsize=(10, 5))

        history = self.gpu_tracker._memory_history
        steps = list(range(len(history)))
        allocated = [s["allocated_mb"] for s in history]
        reserved = [s["reserved_mb"] for s in history]
        total = history[0]["total_mb"] if history else 0

        ax.fill_between(steps, 0, allocated, alpha=0.4, color="blue", label="Allocated")
        ax.fill_between(
            steps, allocated, reserved, alpha=0.3, color="orange", label="Reserved (cache)"
        )
        if total > 0:
            ax.axhline(
                y=total, color="red", linestyle="--", linewidth=1.5, label=f"Total ({total:.0f} MB)"
            )

        ax.set_xlabel("Snapshot Index")
        ax.set_ylabel("Memory (MB)")
        ax.set_title(f"GPU Memory Usage - {self.experiment_name}")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = None
        if save:
            save_path = self.plots_dir / "gpu_memory.png"
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

            if self._wandb_initialized:
                wandb.log({"plots/gpu_memory": wandb.Image(str(save_path))})

        plt.close(fig)
        return save_path

    def log_gpu_metrics(self, step: int) -> Dict[str, float]:
        """
        Log current GPU memory usage and training speed metrics.

        Args:
            step: Current training step

        Returns:
            Dictionary of GPU and speed metrics
        """
        gpu_snapshot = self.gpu_tracker.snapshot()
        peak_memory = self.gpu_tracker.get_peak_memory_mb()

        metrics = {
            "memory/gpu_allocated_mb": gpu_snapshot["allocated_mb"],
            "memory/gpu_reserved_mb": gpu_snapshot["reserved_mb"],
            "memory/gpu_free_mb": gpu_snapshot["free_mb"],
            "memory/gpu_utilization_percent": gpu_snapshot["utilization_percent"],
            "memory/gpu_peak_mb": peak_memory,
            "performance/steps_per_second": self.speed_tracker.get_steps_per_second(),
            "performance/samples_per_second": self.speed_tracker.get_samples_per_second(),
            "performance/total_elapsed_seconds": self.speed_tracker.get_total_elapsed(),
        }

        if self._wandb_initialized:
            wandb.log(metrics, step=step)

        return metrics

    def log_epoch_summary(
        self,
        epoch: int,
        avg_train_loss: float,
        avg_val_loss: Optional[float] = None,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Log end-of-epoch summary metrics.

        Args:
            epoch: Epoch number
            avg_train_loss: Average training loss for the epoch
            avg_val_loss: Average validation loss for the epoch
            extra_metrics: Additional epoch-level metrics
        """
        summary = {
            "epoch/train_loss": avg_train_loss,
            "epoch/number": epoch,
        }
        if avg_val_loss is not None:
            summary["epoch/val_loss"] = avg_val_loss
        if extra_metrics:
            summary.update(extra_metrics)

        if self._wandb_initialized:
            wandb.log(summary)

        logger.info(
            f"Epoch {epoch} summary: train_loss={avg_train_loss:.6f}"
            + (f", val_loss={avg_val_loss:.6f}" if avg_val_loss is not None else "")
        )

    def should_log_samples(self, step: int) -> bool:
        """
        Check if sample images should be logged at this step.

        Args:
            step: Current training step

        Returns:
            True if samples should be logged
        """
        return step > 0 and step % self.sample_log_interval == 0

    def finish(self) -> None:
        """
        Finalize visualization and close W&B run.

        Call this at the end of training to save final plots and
        properly close the W&B run.
        """
        # Save final plots
        self.plot_loss_curves(save=True)
        self.plot_gpu_memory(save=True)

        # Save metrics history
        self._save_metrics_history()

        # Close W&B
        if self._wandb_initialized:
            wandb.finish()
            logger.info("W&B run finished")

        logger.info(f"Training visualization finalized. Outputs saved to: {self.output_dir}")

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------

    def _save_metrics_history(self) -> Path:
        """Save full metrics history to a JSON file."""
        import json

        history_path = self.output_dir / "metrics_history.json"
        history_data = [m.to_dict() for m in self._metrics_history]

        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=2)

        logger.info(f"Metrics history saved: {history_path}")
        return history_path

    @staticmethod
    def _exponential_moving_average(
        values: List[float], weight: float = 0.9
    ) -> List[float]:
        """
        Compute exponential moving average for smoothing.

        Args:
            values: Raw values to smooth
            weight: Smoothing weight (higher = smoother)

        Returns:
            Smoothed values
        """
        smoothed = []
        last = values[0]
        for v in values:
            smoothed_val = last * weight + (1 - weight) * v
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    def _to_pil(self, image: Union["Image.Image", np.ndarray, torch.Tensor]) -> "Image.Image":
        """
        Convert various image formats to PIL Image.

        Args:
            image: Input image (PIL, numpy array, or torch tensor)

        Returns:
            PIL Image
        """
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, torch.Tensor):
            # Handle batched tensors
            if image.dim() == 4:
                image = image[0]
            # Convert CHW to HWC
            if image.dim() == 3 and image.shape[0] in (1, 3):
                image = image.permute(1, 2, 0)
            # Normalize to 0-255
            image = image.detach().cpu().numpy()
            if image.max() <= 1.0:
                image = (image * 255).clip(0, 255)
            image = image.astype(np.uint8)

        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                return Image.fromarray(image, mode="L")
            if image.shape[-1] == 1:
                return Image.fromarray(image.squeeze(-1), mode="L")
            if image.max() <= 1.0:
                image = (image * 255).clip(0, 255).astype(np.uint8)
            return Image.fromarray(image.astype(np.uint8))

        raise ValueError(f"Unsupported image type: {type(image)}")

    def _create_image_grid(
        self,
        images: List["Image.Image"],
        condition_maps: Optional[List[Union["Image.Image", np.ndarray]]] = None,
        max_cols: int = 4,
        padding: int = 4,
    ) -> "Image.Image":
        """
        Create a grid of images, optionally with condition maps above.

        Args:
            images: List of generated images
            condition_maps: Optional condition maps to show above generated images
            max_cols: Maximum columns in the grid
            padding: Padding between images in pixels

        Returns:
            Combined grid as a PIL Image
        """
        n = len(images)
        cols = min(n, max_cols)
        rows = (n + cols - 1) // cols

        # Determine cell size from first image
        cell_w, cell_h = images[0].size

        # If condition maps provided, double the rows
        total_rows = rows * 2 if condition_maps else rows

        grid_w = cols * cell_w + (cols - 1) * padding
        grid_h = total_rows * cell_h + (total_rows - 1) * padding

        grid = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))

        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols

            # Resize if needed
            if img.size != (cell_w, cell_h):
                img = img.resize((cell_w, cell_h), Image.LANCZOS)

            # If condition maps, place them in the top rows
            if condition_maps and idx < len(condition_maps):
                cond = self._to_pil(condition_maps[idx])
                if cond.mode != "RGB":
                    cond = cond.convert("RGB")
                if cond.size != (cell_w, cell_h):
                    cond = cond.resize((cell_w, cell_h), Image.LANCZOS)

                cond_x = col * (cell_w + padding)
                cond_y = row * 2 * (cell_h + padding)
                grid.paste(cond, (cond_x, cond_y))

                img_x = col * (cell_w + padding)
                img_y = row * 2 * (cell_h + padding) + cell_h + padding
            else:
                img_x = col * (cell_w + padding)
                img_y = row * (cell_h + padding)

            if img.mode != "RGB":
                img = img.convert("RGB")
            grid.paste(img, (img_x, img_y))

        return grid


def create_visualizer(
    output_dir: Union[str, Path],
    experiment_name: str = "controlnet_training",
    use_wandb: bool = True,
    wandb_project: str = "controlnet-training",
    wandb_config: Optional[Dict[str, Any]] = None,
    sample_log_interval: int = 500,
    plot_interval: int = 100,
) -> TrainingVisualizer:
    """
    Factory function to create a TrainingVisualizer with sensible defaults.

    Args:
        output_dir: Directory for saving outputs
        experiment_name: Name of the experiment
        use_wandb: Whether to enable W&B logging
        wandb_project: W&B project name
        wandb_config: Configuration to log to W&B
        sample_log_interval: Steps between sample logs
        plot_interval: Steps between plot updates

    Returns:
        Configured TrainingVisualizer instance
    """
    return TrainingVisualizer(
        output_dir=output_dir,
        experiment_name=experiment_name,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_config=wandb_config,
        sample_log_interval=sample_log_interval,
        plot_interval=plot_interval,
    )


if __name__ == "__main__":
    """Demonstration of the training visualization module."""
    import tempfile

    print("Testing Training Visualization Module")
    print("=" * 50)

    # Create a temporary output directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Initialize visualizer without W&B for testing
        visualizer = TrainingVisualizer(
            output_dir=tmpdir,
            experiment_name="test_run",
            use_wandb=False,
            sample_log_interval=10,
            plot_interval=20,
        )

        print("\n1. Simulating training steps...")
        # Simulate training loop
        for step in range(100):
            # Simulate decreasing loss
            loss = 1.0 / (1 + step * 0.05) + np.random.normal(0, 0.02)
            lr = 1e-5 * (1 - step / 100)

            val_loss = None
            if step % 25 == 0 and step > 0:
                val_loss = 1.0 / (1 + step * 0.04) + np.random.normal(0, 0.03)

            metrics = visualizer.log_step(
                step=step,
                loss=loss,
                learning_rate=lr,
                batch_size=1,
                epoch=step // 25,
                validation_loss=val_loss,
            )

            # Log sample images at intervals
            if visualizer.should_log_samples(step):
                # Create dummy sample images
                dummy_images = [
                    np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
                    for _ in range(4)
                ]
                dummy_conditions = [
                    np.random.randint(0, 255, (256, 256), dtype=np.uint8)
                    for _ in range(4)
                ]
                visualizer.log_sample_images(
                    step=step,
                    images=dummy_images,
                    condition_maps=dummy_conditions,
                    prompts=["test prompt"] * 4,
                )

        print(f"   Logged {len(visualizer._metrics_history)} steps")

        # Generate final plots
        print("\n2. Generating loss curve plot...")
        plot_path = visualizer.plot_loss_curves(save=True)
        if plot_path:
            print(f"   Plot saved: {plot_path}")

        print("\n3. Generating GPU memory plot...")
        gpu_plot_path = visualizer.plot_gpu_memory(save=True)
        if gpu_plot_path:
            print(f"   GPU plot saved: {gpu_plot_path}")

        # Log GPU metrics
        print("\n4. GPU metrics snapshot:")
        gpu_metrics = visualizer.log_gpu_metrics(step=99)
        for key, value in gpu_metrics.items():
            print(f"   {key}: {value:.2f}")

        # Speed metrics
        print("\n5. Speed metrics:")
        print(f"   Steps/sec: {visualizer.speed_tracker.get_steps_per_second():.2f}")
        print(f"   Samples/sec: {visualizer.speed_tracker.get_samples_per_second():.2f}")
        print(f"   Total elapsed: {visualizer.speed_tracker.get_total_elapsed():.2f}s")

        # Finalize
        visualizer.finish()

        print("\n" + "=" * 50)
        print("✅ Training visualization module test completed!")
        print("\nKey features implemented:")
        print("✓ Loss curve plotting with smoothing and proper labels (Req 12.1)")
        print("✓ Sample generation logging at intervals (Req 12.2)")
        print("✓ Weights & Biases integration for experiment tracking (Req 12.3)")
        print("✓ GPU memory usage and training speed metrics (Req 12.4)")
        print("✓ Local file saving and W&B cloud logging")
        print("✓ Comprehensive type hints and docstrings")
        print(f"\n📋 Task 7.4 Implementation Complete!")
        print(f"Requirements satisfied: 12.1, 12.2, 12.3, 12.4")
