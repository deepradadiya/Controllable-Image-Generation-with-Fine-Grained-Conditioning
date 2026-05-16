"""
Performance Monitoring and Diagnostics

This module provides comprehensive performance monitoring for the ControlNet
training pipeline, including GPU memory tracking, training speed metrics,
system health monitoring, and aggregated performance reports with optimization
recommendations.

Requirements satisfied: 12.4, 12.5
"""

import time
import logging
import json
import shutil
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque
from pathlib import Path
from enum import Enum

import psutil
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class GPUMemorySnapshot:
    """Snapshot of GPU memory state at a point in time."""

    timestamp: float
    allocated_mb: float
    reserved_mb: float
    free_mb: float
    total_mb: float
    peak_mb: float
    utilization_percent: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class SpeedMetrics:
    """Training speed metrics at a point in time."""

    timestamp: float
    steps_per_second: float
    samples_per_second: float
    data_loading_time_ms: float
    forward_pass_time_ms: float
    backward_pass_time_ms: float
    optimizer_step_time_ms: float
    eta_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class SystemHealthSnapshot:
    """Snapshot of system health metrics."""

    timestamp: float
    cpu_percent: float
    cpu_count: int
    ram_used_mb: float
    ram_total_mb: float
    ram_percent: float
    disk_used_gb: float
    disk_total_gb: float
    disk_percent: float
    gpu_temperature_celsius: Optional[float]
    health_score: float
    status: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)



@dataclass
class PerformanceReport:
    """
    Aggregated performance report combining all monitoring metrics.

    Provides a unified view of GPU memory, training speed, and system health
    along with optimization recommendations. Can be serialized to JSON for
    logging and analysis.
    """

    timestamp: float = field(default_factory=time.time)
    gpu_memory: Optional[GPUMemorySnapshot] = None
    speed_metrics: Optional[SpeedMetrics] = None
    system_health: Optional[SystemHealthSnapshot] = None
    recommendations: List[str] = field(default_factory=list)
    bottleneck: Optional[str] = None
    overall_status: str = HealthStatus.HEALTHY.value

    def to_dict(self) -> Dict[str, Any]:
        """Convert the full report to a dictionary."""
        return {
            "timestamp": self.timestamp,
            "gpu_memory": self.gpu_memory.to_dict() if self.gpu_memory else None,
            "speed_metrics": self.speed_metrics.to_dict() if self.speed_metrics else None,
            "system_health": self.system_health.to_dict() if self.system_health else None,
            "recommendations": self.recommendations,
            "bottleneck": self.bottleneck,
            "overall_status": self.overall_status,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the report to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: Path) -> None:
        """Save the report to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            f.write(self.to_json())
        logger.debug(f"Performance report saved to {filepath}")


class GPUMemoryTracker:
    """
    Track GPU memory usage over time with optimization suggestions.

    Monitors current, peak, and available GPU memory. Maintains a history
    of memory usage for trend analysis and leak detection. Provides
    actionable optimization suggestions when memory usage is high.
    """

    def __init__(
        self,
        high_memory_threshold: float = 0.85,
        critical_memory_threshold: float = 0.95,
        history_size: int = 500,
        leak_detection_window: int = 50,
    ):
        """
        Initialize GPU memory tracker.

        Args:
            high_memory_threshold: Fraction of total memory that triggers warnings (0-1)
            critical_memory_threshold: Fraction of total memory that triggers critical alerts (0-1)
            history_size: Maximum number of snapshots to retain
            leak_detection_window: Number of recent snapshots for leak detection
        """
        self.high_memory_threshold = high_memory_threshold
        self.critical_memory_threshold = critical_memory_threshold
        self.leak_detection_window = leak_detection_window
        self._history: deque = deque(maxlen=history_size)
        self._gpu_available = TORCH_AVAILABLE and torch.cuda.is_available()

    @property
    def gpu_available(self) -> bool:
        """Whether a CUDA GPU is available."""
        return self._gpu_available

    def snapshot(self) -> GPUMemorySnapshot:
        """
        Take a snapshot of current GPU memory state.

        Returns:
            GPUMemorySnapshot with current memory metrics
        """
        if not self._gpu_available:
            snap = GPUMemorySnapshot(
                timestamp=time.time(),
                allocated_mb=0.0,
                reserved_mb=0.0,
                free_mb=0.0,
                total_mb=0.0,
                peak_mb=0.0,
                utilization_percent=0.0,
            )
            self._history.append(snap)
            return snap

        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        free = total - reserved
        peak = torch.cuda.max_memory_allocated() / (1024**2)
        utilization = (reserved / total * 100) if total > 0 else 0.0

        snap = GPUMemorySnapshot(
            timestamp=time.time(),
            allocated_mb=allocated,
            reserved_mb=reserved,
            free_mb=free,
            total_mb=total,
            peak_mb=peak,
            utilization_percent=utilization,
        )
        self._history.append(snap)
        return snap

    def get_current_usage(self) -> Dict[str, float]:
        """
        Get current GPU memory usage summary.

        Returns:
            Dictionary with current, peak, available, and utilization metrics
        """
        snap = self.snapshot()
        return {
            "current_allocated_mb": snap.allocated_mb,
            "peak_allocated_mb": snap.peak_mb,
            "available_mb": snap.free_mb,
            "total_mb": snap.total_mb,
            "utilization_percent": snap.utilization_percent,
        }

    def get_memory_history(self) -> List[Dict[str, float]]:
        """
        Get the full memory usage history.

        Returns:
            List of memory snapshots as dictionaries
        """
        return [snap.to_dict() for snap in self._history]

    def detect_memory_leak(self) -> Dict[str, Any]:
        """
        Detect potential memory leaks by analyzing monotonically increasing usage.

        Checks if GPU memory allocation is consistently increasing over the
        recent window, which indicates a potential memory leak.

        Returns:
            Dictionary with leak detection results including:
            - leak_detected: bool
            - trend_mb_per_snapshot: float (rate of increase)
            - monotonic_increase_count: int
            - analysis: str (human-readable description)
        """
        if len(self._history) < self.leak_detection_window:
            return {
                "leak_detected": False,
                "trend_mb_per_snapshot": 0.0,
                "monotonic_increase_count": 0,
                "analysis": "Insufficient data for leak detection",
            }

        recent = list(self._history)[-self.leak_detection_window:]
        allocated_values = [s.allocated_mb for s in recent]

        # Check for monotonic increase
        monotonic_count = 0
        for i in range(1, len(allocated_values)):
            if allocated_values[i] > allocated_values[i - 1]:
                monotonic_count += 1

        # Calculate linear trend
        x = np.arange(len(allocated_values))
        slope = np.polyfit(x, allocated_values, 1)[0]

        # Leak is detected if memory is mostly increasing and trend is positive
        monotonic_ratio = monotonic_count / (len(allocated_values) - 1)
        leak_detected = monotonic_ratio > 0.8 and slope > 1.0  # >1MB per snapshot

        analysis = "No memory leak detected"
        if leak_detected:
            analysis = (
                f"Potential memory leak: memory increasing at {slope:.2f} MB/snapshot, "
                f"{monotonic_ratio * 100:.0f}% of snapshots show increase"
            )
            logger.warning(analysis)

        return {
            "leak_detected": leak_detected,
            "trend_mb_per_snapshot": slope,
            "monotonic_increase_count": monotonic_count,
            "monotonic_ratio": monotonic_ratio,
            "analysis": analysis,
        }

    def get_optimization_suggestions(self) -> List[str]:
        """
        Generate optimization suggestions based on current memory state.

        Analyzes current memory usage and provides actionable recommendations
        to reduce memory consumption or prevent OOM errors.

        Returns:
            List of optimization suggestion strings
        """
        if not self._gpu_available:
            return ["No GPU available - consider using a GPU-enabled environment"]

        snap = self.snapshot()
        suggestions = []

        utilization_fraction = snap.utilization_percent / 100.0

        if utilization_fraction >= self.critical_memory_threshold:
            suggestions.extend([
                "⚠️ CRITICAL: GPU memory usage is extremely high ({:.1f}%)".format(
                    snap.utilization_percent
                ),
                "• Reduce batch size immediately to prevent OOM",
                "• Enable gradient checkpointing: model.enable_gradient_checkpointing()",
                "• Switch to mixed precision (FP16) training",
                "• Clear unused tensors and call torch.cuda.empty_cache()",
            ])
        elif utilization_fraction >= self.high_memory_threshold:
            suggestions.extend([
                "⚡ WARNING: GPU memory usage is high ({:.1f}%)".format(
                    snap.utilization_percent
                ),
                "• Consider reducing batch size",
                "• Enable gradient checkpointing if not already active",
                "• Use gradient accumulation with smaller micro-batches",
                "• Ensure unused tensors are deleted promptly",
            ])
        else:
            suggestions.append(
                "✅ GPU memory usage is within normal range ({:.1f}%)".format(
                    snap.utilization_percent
                )
            )

        # Check for leak
        leak_info = self.detect_memory_leak()
        if leak_info["leak_detected"]:
            suggestions.extend([
                "🔍 Memory leak detected:",
                "• Check for tensors accumulating in lists or dictionaries",
                "• Ensure .detach() is called on tensors stored for logging",
                "• Verify optimizer.zero_grad() is called each step",
                "• Use torch.no_grad() for inference/evaluation code",
            ])

        # Peak memory analysis
        if snap.peak_mb > snap.total_mb * 0.9:
            suggestions.append(
                "• Peak memory ({:.0f} MB) is near total ({:.0f} MB) - "
                "consider model parallelism or offloading".format(
                    snap.peak_mb, snap.total_mb
                )
            )

        return suggestions



class TrainingSpeedMonitor:
    """
    Monitor training speed and identify performance bottlenecks.

    Tracks steps/second, samples/second, and timing breakdowns for
    data loading, forward pass, backward pass, and optimizer steps.
    Provides ETA estimation and throughput trend analysis.
    """

    def __init__(self, window_size: int = 100, total_steps: Optional[int] = None):
        """
        Initialize training speed monitor.

        Args:
            window_size: Number of recent steps for rolling average calculations
            total_steps: Total planned training steps (for ETA estimation)
        """
        self.window_size = window_size
        self.total_steps = total_steps
        self._step_timestamps: deque = deque(maxlen=window_size)
        self._batch_sizes: deque = deque(maxlen=window_size)
        self._data_load_times: deque = deque(maxlen=window_size)
        self._forward_times: deque = deque(maxlen=window_size)
        self._backward_times: deque = deque(maxlen=window_size)
        self._optimizer_times: deque = deque(maxlen=window_size)
        self._throughput_history: List[float] = []
        self._start_time: Optional[float] = None
        self._current_step: int = 0

    def record_step(
        self,
        batch_size: int = 1,
        data_load_time: float = 0.0,
        forward_time: float = 0.0,
        backward_time: float = 0.0,
        optimizer_time: float = 0.0,
    ) -> None:
        """
        Record timing information for a single training step.

        Args:
            batch_size: Number of samples in this step
            data_load_time: Time spent loading data (seconds)
            forward_time: Time spent on forward pass (seconds)
            backward_time: Time spent on backward pass (seconds)
            optimizer_time: Time spent on optimizer step (seconds)
        """
        now = time.time()
        if self._start_time is None:
            self._start_time = now

        self._step_timestamps.append(now)
        self._batch_sizes.append(batch_size)
        self._data_load_times.append(data_load_time)
        self._forward_times.append(forward_time)
        self._backward_times.append(backward_time)
        self._optimizer_times.append(optimizer_time)
        self._current_step += 1

        # Record throughput for trend analysis
        sps = self.get_steps_per_second()
        if sps > 0:
            self._throughput_history.append(sps)

    def get_steps_per_second(self) -> float:
        """
        Get smoothed steps per second over the rolling window.

        Returns:
            Steps per second (0.0 if insufficient data)
        """
        if len(self._step_timestamps) < 2:
            return 0.0

        elapsed = self._step_timestamps[-1] - self._step_timestamps[0]
        if elapsed <= 0:
            return 0.0

        return (len(self._step_timestamps) - 1) / elapsed

    def get_samples_per_second(self) -> float:
        """
        Get smoothed samples per second over the rolling window.

        Returns:
            Samples per second (0.0 if insufficient data)
        """
        if len(self._step_timestamps) < 2:
            return 0.0

        elapsed = self._step_timestamps[-1] - self._step_timestamps[0]
        if elapsed <= 0:
            return 0.0

        total_samples = sum(self._batch_sizes) - self._batch_sizes[0]
        return total_samples / elapsed

    def get_eta(self) -> float:
        """
        Estimate time remaining for training completion.

        Returns:
            Estimated seconds remaining (inf if cannot estimate)
        """
        if self.total_steps is None:
            return float("inf")

        sps = self.get_steps_per_second()
        if sps <= 0:
            return float("inf")

        remaining_steps = self.total_steps - self._current_step
        if remaining_steps <= 0:
            return 0.0

        return remaining_steps / sps

    def get_eta_formatted(self) -> str:
        """
        Get ETA as a human-readable string.

        Returns:
            Formatted ETA string (e.g., "2h 15m 30s")
        """
        eta = self.get_eta()
        if eta == float("inf"):
            return "unknown"

        hours = int(eta // 3600)
        minutes = int((eta % 3600) // 60)
        seconds = int(eta % 60)

        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{seconds}s")

        return " ".join(parts)

    def identify_bottleneck(self) -> Dict[str, Any]:
        """
        Identify the primary training bottleneck.

        Analyzes timing breakdowns to determine whether data loading,
        forward pass, backward pass, or optimizer step is the bottleneck.

        Returns:
            Dictionary with bottleneck identification and timing breakdown
        """
        if len(self._data_load_times) < 5:
            return {
                "bottleneck": "unknown",
                "analysis": "Insufficient data for bottleneck analysis",
                "breakdown": {},
            }

        avg_data = np.mean(list(self._data_load_times)) * 1000  # ms
        avg_forward = np.mean(list(self._forward_times)) * 1000
        avg_backward = np.mean(list(self._backward_times)) * 1000
        avg_optimizer = np.mean(list(self._optimizer_times)) * 1000

        total_time = avg_data + avg_forward + avg_backward + avg_optimizer

        breakdown = {
            "data_loading_ms": avg_data,
            "forward_pass_ms": avg_forward,
            "backward_pass_ms": avg_backward,
            "optimizer_step_ms": avg_optimizer,
            "total_step_ms": total_time,
        }

        if total_time <= 0:
            return {
                "bottleneck": "unknown",
                "analysis": "No timing data recorded",
                "breakdown": breakdown,
            }

        # Calculate percentages
        breakdown["data_loading_percent"] = (avg_data / total_time) * 100
        breakdown["forward_pass_percent"] = (avg_forward / total_time) * 100
        breakdown["backward_pass_percent"] = (avg_backward / total_time) * 100
        breakdown["optimizer_step_percent"] = (avg_optimizer / total_time) * 100

        # Identify bottleneck
        components = {
            "data_loading": avg_data,
            "forward_pass": avg_forward,
            "backward_pass": avg_backward,
            "optimizer_step": avg_optimizer,
        }
        bottleneck = max(components, key=components.get)

        # Generate analysis
        suggestions = []
        if bottleneck == "data_loading":
            suggestions = [
                "Data loading is the bottleneck ({:.1f}% of step time)".format(
                    breakdown["data_loading_percent"]
                ),
                "• Increase num_workers in DataLoader",
                "• Enable pin_memory=True for faster GPU transfer",
                "• Use prefetch_factor > 2 for more aggressive prefetching",
                "• Consider caching preprocessed data to disk",
            ]
        elif bottleneck == "forward_pass":
            suggestions = [
                "Forward pass is the bottleneck ({:.1f}% of step time)".format(
                    breakdown["forward_pass_percent"]
                ),
                "• Enable mixed precision (FP16) to speed up computation",
                "• Use torch.compile() for graph optimization (PyTorch 2.0+)",
                "• Consider using xformers for memory-efficient attention",
            ]
        elif bottleneck == "backward_pass":
            suggestions = [
                "Backward pass is the bottleneck ({:.1f}% of step time)".format(
                    breakdown["backward_pass_percent"]
                ),
                "• Enable mixed precision for faster gradient computation",
                "• Gradient checkpointing trades speed for memory - disable if memory allows",
                "• Ensure no unnecessary computation graphs are retained",
            ]
        elif bottleneck == "optimizer_step":
            suggestions = [
                "Optimizer step is the bottleneck ({:.1f}% of step time)".format(
                    breakdown["optimizer_step_percent"]
                ),
                "• Consider using fused optimizer implementations",
                "• Use 8-bit optimizers (bitsandbytes) for faster updates",
                "• Reduce the number of trainable parameters",
            ]

        return {
            "bottleneck": bottleneck,
            "analysis": suggestions[0] if suggestions else "No bottleneck identified",
            "suggestions": suggestions,
            "breakdown": breakdown,
        }

    def get_throughput_trend(self) -> Dict[str, Any]:
        """
        Analyze throughput trend over training.

        Returns:
            Dictionary with trend analysis including direction and magnitude
        """
        if len(self._throughput_history) < 10:
            return {
                "trend": "insufficient_data",
                "direction": "unknown",
                "change_percent": 0.0,
            }

        recent = self._throughput_history[-50:]
        early = recent[: len(recent) // 2]
        late = recent[len(recent) // 2:]

        early_avg = np.mean(early)
        late_avg = np.mean(late)

        if early_avg > 0:
            change_percent = ((late_avg - early_avg) / early_avg) * 100
        else:
            change_percent = 0.0

        if change_percent > 5:
            direction = "improving"
        elif change_percent < -5:
            direction = "degrading"
        else:
            direction = "stable"

        return {
            "trend": "analyzed",
            "direction": direction,
            "change_percent": change_percent,
            "current_throughput": late_avg,
            "initial_throughput": early_avg,
        }

    def get_speed_metrics(self) -> SpeedMetrics:
        """
        Get current speed metrics as a SpeedMetrics dataclass.

        Returns:
            SpeedMetrics with all current timing information
        """
        avg_data = np.mean(list(self._data_load_times)) * 1000 if self._data_load_times else 0.0
        avg_forward = np.mean(list(self._forward_times)) * 1000 if self._forward_times else 0.0
        avg_backward = np.mean(list(self._backward_times)) * 1000 if self._backward_times else 0.0
        avg_optimizer = np.mean(list(self._optimizer_times)) * 1000 if self._optimizer_times else 0.0

        return SpeedMetrics(
            timestamp=time.time(),
            steps_per_second=self.get_steps_per_second(),
            samples_per_second=self.get_samples_per_second(),
            data_loading_time_ms=avg_data,
            forward_pass_time_ms=avg_forward,
            backward_pass_time_ms=avg_backward,
            optimizer_step_time_ms=avg_optimizer,
            eta_seconds=self.get_eta(),
        )



class SystemHealthMonitor:
    """
    Monitor system health including CPU, RAM, disk, and GPU temperature.

    Provides an overall health score (0-100) based on resource utilization
    and identifies potential system-level issues that could affect training.
    """

    def __init__(
        self,
        cpu_warning_threshold: float = 90.0,
        ram_warning_threshold: float = 85.0,
        disk_warning_threshold: float = 90.0,
        gpu_temp_warning_celsius: float = 80.0,
    ):
        """
        Initialize system health monitor.

        Args:
            cpu_warning_threshold: CPU usage percent that triggers warning
            ram_warning_threshold: RAM usage percent that triggers warning
            disk_warning_threshold: Disk usage percent that triggers warning
            gpu_temp_warning_celsius: GPU temperature that triggers warning
        """
        self.cpu_warning_threshold = cpu_warning_threshold
        self.ram_warning_threshold = ram_warning_threshold
        self.disk_warning_threshold = disk_warning_threshold
        self.gpu_temp_warning_celsius = gpu_temp_warning_celsius
        self._history: List[SystemHealthSnapshot] = []

    def get_cpu_usage(self) -> Dict[str, Any]:
        """
        Get current CPU utilization.

        Returns:
            Dictionary with CPU usage metrics
        """
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()

        result = {
            "percent": cpu_percent,
            "count": cpu_count,
        }
        if cpu_freq:
            result["frequency_mhz"] = cpu_freq.current

        return result

    def get_ram_usage(self) -> Dict[str, float]:
        """
        Get current system RAM usage.

        Returns:
            Dictionary with RAM usage metrics in MB
        """
        mem = psutil.virtual_memory()
        return {
            "used_mb": mem.used / (1024**2),
            "total_mb": mem.total / (1024**2),
            "available_mb": mem.available / (1024**2),
            "percent": mem.percent,
        }

    def get_disk_usage(self, path: str = "/") -> Dict[str, float]:
        """
        Get disk space usage for the specified path.

        Args:
            path: Filesystem path to check (default: root)

        Returns:
            Dictionary with disk usage metrics in GB
        """
        try:
            usage = shutil.disk_usage(path)
            total_gb = usage.total / (1024**3)
            used_gb = usage.used / (1024**3)
            free_gb = usage.free / (1024**3)
            percent = (usage.used / usage.total) * 100 if usage.total > 0 else 0.0

            return {
                "used_gb": used_gb,
                "total_gb": total_gb,
                "free_gb": free_gb,
                "percent": percent,
            }
        except OSError as e:
            logger.warning(f"Failed to get disk usage for {path}: {e}")
            return {"used_gb": 0.0, "total_gb": 0.0, "free_gb": 0.0, "percent": 0.0}

    def get_gpu_temperature(self) -> Optional[float]:
        """
        Get GPU temperature in Celsius if available.

        Uses nvidia-smi via PyTorch or pynvml if available.

        Returns:
            GPU temperature in Celsius, or None if unavailable
        """
        if not (TORCH_AVAILABLE and torch.cuda.is_available()):
            return None

        try:
            # Try using pynvml through torch
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                temp_str = result.stdout.strip().split("\n")[0]
                return float(temp_str)
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
            pass

        return None

    def compute_health_score(self) -> Tuple[float, HealthStatus]:
        """
        Compute an overall system health score (0-100).

        The score is based on:
        - CPU utilization (25% weight)
        - RAM utilization (25% weight)
        - Disk space (25% weight)
        - GPU temperature (25% weight, if available)

        Returns:
            Tuple of (health_score, HealthStatus)
        """
        scores = []

        # CPU score (lower usage = higher score)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_score = max(0, 100 - cpu_percent)
        scores.append(cpu_score)

        # RAM score
        ram = psutil.virtual_memory()
        ram_score = max(0, 100 - ram.percent)
        scores.append(ram_score)

        # Disk score
        disk = self.get_disk_usage()
        disk_score = max(0, 100 - disk["percent"])
        scores.append(disk_score)

        # GPU temperature score (if available)
        gpu_temp = self.get_gpu_temperature()
        if gpu_temp is not None:
            # Score decreases as temperature approaches warning threshold
            temp_score = max(0, 100 - (gpu_temp / self.gpu_temp_warning_celsius) * 100)
            temp_score = min(100, max(0, 100 - max(0, gpu_temp - 40) * 1.5))
            scores.append(temp_score)

        # Weighted average
        health_score = np.mean(scores)

        # Determine status
        if health_score >= 60:
            status = HealthStatus.HEALTHY
        elif health_score >= 30:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.CRITICAL

        return health_score, status

    def snapshot(self) -> SystemHealthSnapshot:
        """
        Take a complete system health snapshot.

        Returns:
            SystemHealthSnapshot with all current metrics
        """
        cpu = self.get_cpu_usage()
        ram = self.get_ram_usage()
        disk = self.get_disk_usage()
        gpu_temp = self.get_gpu_temperature()
        health_score, status = self.compute_health_score()

        snap = SystemHealthSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu["percent"],
            cpu_count=cpu["count"],
            ram_used_mb=ram["used_mb"],
            ram_total_mb=ram["total_mb"],
            ram_percent=ram["percent"],
            disk_used_gb=disk["used_gb"],
            disk_total_gb=disk["total_gb"],
            disk_percent=disk["percent"],
            gpu_temperature_celsius=gpu_temp,
            health_score=health_score,
            status=status.value,
        )

        self._history.append(snap)
        return snap

    def get_resource_alerts(self) -> List[str]:
        """
        Get alerts for any resources that are in warning or critical state.

        Returns:
            List of alert messages
        """
        alerts = []

        cpu_percent = psutil.cpu_percent(interval=0.1)
        if cpu_percent >= self.cpu_warning_threshold:
            alerts.append(
                f"⚠️ High CPU usage: {cpu_percent:.1f}% "
                f"(threshold: {self.cpu_warning_threshold}%)"
            )

        ram = psutil.virtual_memory()
        if ram.percent >= self.ram_warning_threshold:
            alerts.append(
                f"⚠️ High RAM usage: {ram.percent:.1f}% "
                f"({ram.used / (1024**3):.1f} GB / {ram.total / (1024**3):.1f} GB)"
            )

        disk = self.get_disk_usage()
        if disk["percent"] >= self.disk_warning_threshold:
            alerts.append(
                f"⚠️ Low disk space: {disk['percent']:.1f}% used "
                f"({disk['free_gb']:.1f} GB free)"
            )

        gpu_temp = self.get_gpu_temperature()
        if gpu_temp is not None and gpu_temp >= self.gpu_temp_warning_celsius:
            alerts.append(
                f"🌡️ High GPU temperature: {gpu_temp:.0f}°C "
                f"(threshold: {self.gpu_temp_warning_celsius}°C)"
            )

        return alerts



class PerformanceMonitor:
    """
    Unified performance monitoring system that combines GPU memory tracking,
    training speed monitoring, and system health monitoring.

    Designed for periodic polling at a configurable interval. Generates
    aggregated PerformanceReport instances with optimization recommendations.
    """

    def __init__(
        self,
        poll_interval_seconds: float = 30.0,
        total_training_steps: Optional[int] = None,
        output_dir: Optional[Path] = None,
        high_memory_threshold: float = 0.85,
    ):
        """
        Initialize the unified performance monitor.

        Args:
            poll_interval_seconds: Seconds between automatic polling
            total_training_steps: Total planned training steps for ETA
            output_dir: Directory for saving performance reports
            high_memory_threshold: GPU memory fraction that triggers warnings
        """
        self.poll_interval = poll_interval_seconds
        self.output_dir = Path(output_dir) if output_dir else None
        self._last_poll_time: float = 0.0

        # Sub-monitors
        self.gpu_tracker = GPUMemoryTracker(
            high_memory_threshold=high_memory_threshold
        )
        self.speed_monitor = TrainingSpeedMonitor(
            total_steps=total_training_steps
        )
        self.health_monitor = SystemHealthMonitor()

        # Report history
        self._reports: List[PerformanceReport] = []

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"PerformanceMonitor initialized: poll_interval={poll_interval_seconds}s, "
            f"total_steps={total_training_steps}"
        )

    def should_poll(self) -> bool:
        """
        Check if it's time for the next polling cycle.

        Returns:
            True if poll_interval has elapsed since last poll
        """
        return (time.time() - self._last_poll_time) >= self.poll_interval

    def record_training_step(
        self,
        batch_size: int = 1,
        data_load_time: float = 0.0,
        forward_time: float = 0.0,
        backward_time: float = 0.0,
        optimizer_time: float = 0.0,
    ) -> None:
        """
        Record timing for a training step.

        Args:
            batch_size: Number of samples in this step
            data_load_time: Time spent loading data (seconds)
            forward_time: Time spent on forward pass (seconds)
            backward_time: Time spent on backward pass (seconds)
            optimizer_time: Time spent on optimizer step (seconds)
        """
        self.speed_monitor.record_step(
            batch_size=batch_size,
            data_load_time=data_load_time,
            forward_time=forward_time,
            backward_time=backward_time,
            optimizer_time=optimizer_time,
        )

    def generate_report(self) -> PerformanceReport:
        """
        Generate a comprehensive performance report.

        Collects metrics from all sub-monitors and produces optimization
        recommendations based on current system state.

        Returns:
            PerformanceReport with all metrics and recommendations
        """
        self._last_poll_time = time.time()

        # Collect metrics
        gpu_snapshot = self.gpu_tracker.snapshot()
        speed_metrics = self.speed_monitor.get_speed_metrics()
        health_snapshot = self.health_monitor.snapshot()

        # Generate recommendations
        recommendations = []
        recommendations.extend(self.gpu_tracker.get_optimization_suggestions())

        bottleneck_info = self.speed_monitor.identify_bottleneck()
        if bottleneck_info.get("suggestions"):
            recommendations.extend(bottleneck_info["suggestions"])

        alerts = self.health_monitor.get_resource_alerts()
        recommendations.extend(alerts)

        # Throughput trend
        trend = self.speed_monitor.get_throughput_trend()
        if trend.get("direction") == "degrading":
            recommendations.append(
                "📉 Training throughput is degrading ({:.1f}% decrease) - "
                "check for resource contention".format(abs(trend["change_percent"]))
            )

        # Determine overall status
        if health_snapshot.status == HealthStatus.CRITICAL.value:
            overall_status = HealthStatus.CRITICAL.value
        elif (
            health_snapshot.status == HealthStatus.WARNING.value
            or gpu_snapshot.utilization_percent > 85
        ):
            overall_status = HealthStatus.WARNING.value
        else:
            overall_status = HealthStatus.HEALTHY.value

        report = PerformanceReport(
            timestamp=time.time(),
            gpu_memory=gpu_snapshot,
            speed_metrics=speed_metrics,
            system_health=health_snapshot,
            recommendations=recommendations,
            bottleneck=bottleneck_info.get("bottleneck"),
            overall_status=overall_status,
        )

        self._reports.append(report)

        # Save report if output directory is configured
        if self.output_dir:
            report_path = self.output_dir / "latest_performance_report.json"
            report.save(report_path)

        return report

    def poll(self) -> Optional[PerformanceReport]:
        """
        Poll for metrics if the interval has elapsed.

        Returns:
            PerformanceReport if polling occurred, None otherwise
        """
        if self.should_poll():
            return self.generate_report()
        return None

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a concise summary of current performance state.

        Returns:
            Dictionary with key performance indicators
        """
        gpu_usage = self.gpu_tracker.get_current_usage()
        speed = self.speed_monitor.get_steps_per_second()
        eta = self.speed_monitor.get_eta_formatted()
        health_score, health_status = self.health_monitor.compute_health_score()

        return {
            "gpu_memory_percent": gpu_usage.get("utilization_percent", 0.0),
            "gpu_allocated_mb": gpu_usage.get("current_allocated_mb", 0.0),
            "steps_per_second": speed,
            "samples_per_second": self.speed_monitor.get_samples_per_second(),
            "eta": eta,
            "health_score": health_score,
            "health_status": health_status.value,
            "bottleneck": self.speed_monitor.identify_bottleneck().get("bottleneck", "unknown"),
        }

    def get_report_history(self) -> List[Dict[str, Any]]:
        """
        Get history of all generated reports.

        Returns:
            List of report dictionaries
        """
        return [r.to_dict() for r in self._reports]


def main():
    """Demonstrate performance monitoring capabilities."""
    import sys

    print("=" * 60)
    print("Performance Monitoring and Diagnostics Demo")
    print("=" * 60)

    # Initialize the unified monitor
    monitor = PerformanceMonitor(
        poll_interval_seconds=5.0,
        total_training_steps=1000,
    )

    # --- GPU Memory Tracking ---
    print("\n1. GPU Memory Tracking")
    print("-" * 40)
    gpu_usage = monitor.gpu_tracker.get_current_usage()
    print(f"  Allocated: {gpu_usage['current_allocated_mb']:.1f} MB")
    print(f"  Peak: {gpu_usage['peak_allocated_mb']:.1f} MB")
    print(f"  Available: {gpu_usage['available_mb']:.1f} MB")
    print(f"  Total: {gpu_usage['total_mb']:.1f} MB")
    print(f"  Utilization: {gpu_usage['utilization_percent']:.1f}%")

    suggestions = monitor.gpu_tracker.get_optimization_suggestions()
    print("\n  Optimization suggestions:")
    for s in suggestions:
        print(f"    {s}")

    # --- Training Speed Monitoring ---
    print("\n2. Training Speed Monitoring")
    print("-" * 40)

    # Simulate some training steps
    for i in range(20):
        monitor.record_training_step(
            batch_size=4,
            data_load_time=0.05 + np.random.normal(0, 0.01),
            forward_time=0.15 + np.random.normal(0, 0.02),
            backward_time=0.20 + np.random.normal(0, 0.02),
            optimizer_time=0.02 + np.random.normal(0, 0.005),
        )
        time.sleep(0.01)  # Small delay to simulate real timing

    print(f"  Steps/second: {monitor.speed_monitor.get_steps_per_second():.2f}")
    print(f"  Samples/second: {monitor.speed_monitor.get_samples_per_second():.2f}")
    print(f"  ETA: {monitor.speed_monitor.get_eta_formatted()}")

    bottleneck = monitor.speed_monitor.identify_bottleneck()
    print(f"\n  Bottleneck: {bottleneck['bottleneck']}")
    print(f"  Analysis: {bottleneck['analysis']}")
    if bottleneck.get("breakdown"):
        bd = bottleneck["breakdown"]
        print(f"  Breakdown:")
        print(f"    Data loading: {bd.get('data_loading_ms', 0):.1f} ms ({bd.get('data_loading_percent', 0):.1f}%)")
        print(f"    Forward pass: {bd.get('forward_pass_ms', 0):.1f} ms ({bd.get('forward_pass_percent', 0):.1f}%)")
        print(f"    Backward pass: {bd.get('backward_pass_ms', 0):.1f} ms ({bd.get('backward_pass_percent', 0):.1f}%)")
        print(f"    Optimizer: {bd.get('optimizer_step_ms', 0):.1f} ms ({bd.get('optimizer_step_percent', 0):.1f}%)")

    # --- System Health Monitoring ---
    print("\n3. System Health Monitoring")
    print("-" * 40)
    health = monitor.health_monitor.snapshot()
    print(f"  CPU: {health.cpu_percent:.1f}% ({health.cpu_count} cores)")
    print(f"  RAM: {health.ram_used_mb:.0f} MB / {health.ram_total_mb:.0f} MB ({health.ram_percent:.1f}%)")
    print(f"  Disk: {health.disk_used_gb:.1f} GB / {health.disk_total_gb:.1f} GB ({health.disk_percent:.1f}%)")
    print(f"  GPU Temp: {health.gpu_temperature_celsius or 'N/A'}°C")
    print(f"  Health Score: {health.health_score:.1f}/100")
    print(f"  Status: {health.status}")

    alerts = monitor.health_monitor.get_resource_alerts()
    if alerts:
        print("\n  Alerts:")
        for alert in alerts:
            print(f"    {alert}")
    else:
        print("\n  No resource alerts")

    # --- Generate Full Report ---
    print("\n4. Full Performance Report")
    print("-" * 40)
    report = monitor.generate_report()
    print(f"  Overall Status: {report.overall_status}")
    print(f"  Bottleneck: {report.bottleneck}")
    print(f"  Recommendations ({len(report.recommendations)}):")
    for rec in report.recommendations[:5]:
        print(f"    {rec}")

    # Show JSON serialization
    print("\n5. JSON Serialization (truncated)")
    print("-" * 40)
    json_str = report.to_json()
    # Show first 500 chars
    print(f"  {json_str[:500]}...")

    print("\n" + "=" * 60)
    print("✅ Performance monitoring demo complete!")
    print(f"\nKey features implemented:")
    print("  ✓ GPU memory usage tracking with leak detection")
    print("  ✓ Training speed metrics with bottleneck identification")
    print("  ✓ System health monitoring (CPU, RAM, disk, GPU temp)")
    print("  ✓ Aggregated performance reports with recommendations")
    print("  ✓ JSON serialization for logging")
    print("  ✓ Configurable polling interval")
    print(f"\n📋 Task 12.2 Implementation Complete!")
    print(f"Requirements satisfied: 12.4, 12.5")


if __name__ == "__main__":
    main()
