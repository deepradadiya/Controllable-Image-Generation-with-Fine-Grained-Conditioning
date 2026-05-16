"""
Test suite for performance monitoring and diagnostics.

Tests GPU memory tracking, training speed monitoring, system health monitoring,
and aggregated performance reports.

Requirements tested: 12.4, 12.5
"""

import time
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

from src.utils.performance_monitor import (
    GPUMemoryTracker,
    GPUMemorySnapshot,
    TrainingSpeedMonitor,
    SpeedMetrics,
    SystemHealthMonitor,
    SystemHealthSnapshot,
    PerformanceMonitor,
    PerformanceReport,
    HealthStatus,
)


class TestGPUMemoryTracker:
    """Test GPU memory tracking and optimization suggestions."""

    def test_initialization(self):
        """Test tracker initializes with correct defaults."""
        tracker = GPUMemoryTracker()
        assert tracker.high_memory_threshold == 0.85
        assert tracker.critical_memory_threshold == 0.95
        assert tracker.leak_detection_window == 50

    def test_snapshot_without_gpu(self):
        """Test snapshot returns zeros when no GPU is available."""
        tracker = GPUMemoryTracker()
        tracker._gpu_available = False

        snap = tracker.snapshot()
        assert snap.allocated_mb == 0.0
        assert snap.reserved_mb == 0.0
        assert snap.total_mb == 0.0
        assert snap.utilization_percent == 0.0

    def test_snapshot_stored_in_history(self):
        """Test that snapshots are stored in history."""
        tracker = GPUMemoryTracker()
        tracker._gpu_available = False

        tracker.snapshot()
        tracker.snapshot()
        tracker.snapshot()

        assert len(tracker._history) == 3

    def test_get_current_usage(self):
        """Test get_current_usage returns expected keys."""
        tracker = GPUMemoryTracker()
        tracker._gpu_available = False

        usage = tracker.get_current_usage()
        assert "current_allocated_mb" in usage
        assert "peak_allocated_mb" in usage
        assert "available_mb" in usage
        assert "total_mb" in usage
        assert "utilization_percent" in usage

    def test_get_memory_history(self):
        """Test memory history retrieval."""
        tracker = GPUMemoryTracker()
        tracker._gpu_available = False

        for _ in range(5):
            tracker.snapshot()

        history = tracker.get_memory_history()
        assert len(history) == 5
        assert all(isinstance(h, dict) for h in history)

    def test_detect_memory_leak_insufficient_data(self):
        """Test leak detection with insufficient data."""
        tracker = GPUMemoryTracker(leak_detection_window=50)
        tracker._gpu_available = False

        # Only take a few snapshots
        for _ in range(5):
            tracker.snapshot()

        result = tracker.detect_memory_leak()
        assert result["leak_detected"] is False
        assert "Insufficient data" in result["analysis"]

    def test_detect_memory_leak_no_leak(self):
        """Test leak detection when memory is stable."""
        tracker = GPUMemoryTracker(leak_detection_window=10)

        # Simulate stable memory usage
        for i in range(15):
            snap = GPUMemorySnapshot(
                timestamp=time.time(),
                allocated_mb=1000.0 + np.random.normal(0, 2),  # Small fluctuation
                reserved_mb=1200.0,
                free_mb=800.0,
                total_mb=2000.0,
                peak_mb=1050.0,
                utilization_percent=60.0,
            )
            tracker._history.append(snap)

        result = tracker.detect_memory_leak()
        assert result["leak_detected"] is False

    def test_detect_memory_leak_with_leak(self):
        """Test leak detection when memory is monotonically increasing."""
        tracker = GPUMemoryTracker(leak_detection_window=10)

        # Simulate memory leak (monotonically increasing)
        for i in range(15):
            snap = GPUMemorySnapshot(
                timestamp=time.time(),
                allocated_mb=1000.0 + i * 10.0,  # Increasing by 10MB each step
                reserved_mb=1200.0 + i * 10.0,
                free_mb=800.0 - i * 10.0,
                total_mb=2000.0,
                peak_mb=1000.0 + i * 10.0,
                utilization_percent=60.0 + i * 0.5,
            )
            tracker._history.append(snap)

        result = tracker.detect_memory_leak()
        assert result["leak_detected"] == True
        assert result["trend_mb_per_snapshot"] > 1.0
        assert result["monotonic_ratio"] > 0.8

    def test_optimization_suggestions_no_gpu(self):
        """Test suggestions when no GPU is available."""
        tracker = GPUMemoryTracker()
        tracker._gpu_available = False

        suggestions = tracker.get_optimization_suggestions()
        assert len(suggestions) > 0
        assert any("No GPU" in s for s in suggestions)

    def test_history_size_limit(self):
        """Test that history respects the size limit."""
        tracker = GPUMemoryTracker(history_size=10)
        tracker._gpu_available = False

        for _ in range(20):
            tracker.snapshot()

        assert len(tracker._history) == 10


class TestTrainingSpeedMonitor:
    """Test training speed monitoring and bottleneck identification."""

    def test_initialization(self):
        """Test monitor initializes correctly."""
        monitor = TrainingSpeedMonitor(window_size=50, total_steps=1000)
        assert monitor.window_size == 50
        assert monitor.total_steps == 1000
        assert monitor._current_step == 0

    def test_record_step(self):
        """Test recording a training step."""
        monitor = TrainingSpeedMonitor()
        monitor.record_step(
            batch_size=4,
            data_load_time=0.05,
            forward_time=0.15,
            backward_time=0.20,
            optimizer_time=0.02,
        )
        assert monitor._current_step == 1
        assert len(monitor._step_timestamps) == 1

    def test_steps_per_second(self):
        """Test steps per second calculation."""
        monitor = TrainingSpeedMonitor()

        # Record steps with known timing
        for i in range(10):
            monitor.record_step(batch_size=1)
            time.sleep(0.01)

        sps = monitor.get_steps_per_second()
        # Should be roughly 100 steps/sec (1/0.01), but allow tolerance
        assert sps > 0

    def test_steps_per_second_insufficient_data(self):
        """Test steps/sec returns 0 with insufficient data."""
        monitor = TrainingSpeedMonitor()
        assert monitor.get_steps_per_second() == 0.0

        monitor.record_step()
        assert monitor.get_steps_per_second() == 0.0

    def test_samples_per_second(self):
        """Test samples per second calculation."""
        monitor = TrainingSpeedMonitor()

        for _ in range(10):
            monitor.record_step(batch_size=8)
            time.sleep(0.01)

        sps = monitor.get_samples_per_second()
        assert sps > 0

    def test_eta_with_total_steps(self):
        """Test ETA estimation with known total steps."""
        monitor = TrainingSpeedMonitor(total_steps=100)

        for _ in range(10):
            monitor.record_step(batch_size=1)
            time.sleep(0.01)

        eta = monitor.get_eta()
        assert eta > 0
        assert eta != float("inf")

    def test_eta_without_total_steps(self):
        """Test ETA returns inf when total_steps is not set."""
        monitor = TrainingSpeedMonitor(total_steps=None)
        monitor.record_step()
        assert monitor.get_eta() == float("inf")

    def test_eta_formatted(self):
        """Test formatted ETA string."""
        monitor = TrainingSpeedMonitor(total_steps=100)

        for _ in range(10):
            monitor.record_step()
            time.sleep(0.01)

        eta_str = monitor.get_eta_formatted()
        assert isinstance(eta_str, str)
        assert eta_str != "unknown"

    def test_identify_bottleneck_insufficient_data(self):
        """Test bottleneck identification with insufficient data."""
        monitor = TrainingSpeedMonitor()
        monitor.record_step(data_load_time=0.1, forward_time=0.2)

        result = monitor.identify_bottleneck()
        assert result["bottleneck"] == "unknown"

    def test_identify_bottleneck_data_loading(self):
        """Test bottleneck identification when data loading is slowest."""
        monitor = TrainingSpeedMonitor()

        for _ in range(10):
            monitor.record_step(
                batch_size=1,
                data_load_time=0.5,  # Dominant
                forward_time=0.1,
                backward_time=0.1,
                optimizer_time=0.01,
            )

        result = monitor.identify_bottleneck()
        assert result["bottleneck"] == "data_loading"
        assert "data_loading_ms" in result["breakdown"]

    def test_identify_bottleneck_backward_pass(self):
        """Test bottleneck identification when backward pass is slowest."""
        monitor = TrainingSpeedMonitor()

        for _ in range(10):
            monitor.record_step(
                batch_size=1,
                data_load_time=0.01,
                forward_time=0.05,
                backward_time=0.5,  # Dominant
                optimizer_time=0.01,
            )

        result = monitor.identify_bottleneck()
        assert result["bottleneck"] == "backward_pass"

    def test_throughput_trend_insufficient_data(self):
        """Test throughput trend with insufficient data."""
        monitor = TrainingSpeedMonitor()
        monitor.record_step()

        trend = monitor.get_throughput_trend()
        assert trend["trend"] == "insufficient_data"

    def test_get_speed_metrics(self):
        """Test getting speed metrics as dataclass."""
        monitor = TrainingSpeedMonitor(total_steps=100)

        for _ in range(5):
            monitor.record_step(
                batch_size=4,
                data_load_time=0.05,
                forward_time=0.1,
                backward_time=0.15,
                optimizer_time=0.02,
            )
            time.sleep(0.01)

        metrics = monitor.get_speed_metrics()
        assert isinstance(metrics, SpeedMetrics)
        assert metrics.steps_per_second >= 0
        assert metrics.data_loading_time_ms > 0
        assert metrics.forward_pass_time_ms > 0


class TestSystemHealthMonitor:
    """Test system health monitoring."""

    def test_initialization(self):
        """Test monitor initializes with correct defaults."""
        monitor = SystemHealthMonitor()
        assert monitor.cpu_warning_threshold == 90.0
        assert monitor.ram_warning_threshold == 85.0
        assert monitor.disk_warning_threshold == 90.0
        assert monitor.gpu_temp_warning_celsius == 80.0

    def test_get_cpu_usage(self):
        """Test CPU usage retrieval."""
        monitor = SystemHealthMonitor()
        cpu = monitor.get_cpu_usage()

        assert "percent" in cpu
        assert "count" in cpu
        assert cpu["percent"] >= 0
        assert cpu["count"] > 0

    def test_get_ram_usage(self):
        """Test RAM usage retrieval."""
        monitor = SystemHealthMonitor()
        ram = monitor.get_ram_usage()

        assert "used_mb" in ram
        assert "total_mb" in ram
        assert "available_mb" in ram
        assert "percent" in ram
        assert ram["total_mb"] > 0
        assert ram["used_mb"] > 0
        assert 0 <= ram["percent"] <= 100

    def test_get_disk_usage(self):
        """Test disk usage retrieval."""
        monitor = SystemHealthMonitor()
        disk = monitor.get_disk_usage("/")

        assert "used_gb" in disk
        assert "total_gb" in disk
        assert "free_gb" in disk
        assert "percent" in disk
        assert disk["total_gb"] > 0

    def test_get_disk_usage_invalid_path(self):
        """Test disk usage with invalid path returns zeros."""
        monitor = SystemHealthMonitor()
        disk = monitor.get_disk_usage("/nonexistent/path/that/does/not/exist")

        assert disk["total_gb"] == 0.0

    def test_compute_health_score(self):
        """Test health score computation."""
        monitor = SystemHealthMonitor()
        score, status = monitor.compute_health_score()

        assert 0 <= score <= 100
        assert isinstance(status, HealthStatus)

    def test_snapshot(self):
        """Test taking a system health snapshot."""
        monitor = SystemHealthMonitor()
        snap = monitor.snapshot()

        assert isinstance(snap, SystemHealthSnapshot)
        assert snap.cpu_percent >= 0
        assert snap.cpu_count > 0
        assert snap.ram_total_mb > 0
        assert snap.disk_total_gb > 0
        assert 0 <= snap.health_score <= 100
        assert snap.status in [s.value for s in HealthStatus]

    def test_snapshot_stored_in_history(self):
        """Test snapshots are stored in history."""
        monitor = SystemHealthMonitor()
        monitor.snapshot()
        monitor.snapshot()

        assert len(monitor._history) == 2

    def test_get_resource_alerts_normal(self):
        """Test resource alerts under normal conditions."""
        monitor = SystemHealthMonitor(
            cpu_warning_threshold=99.0,  # Very high threshold
            ram_warning_threshold=99.0,
            disk_warning_threshold=99.0,
        )
        alerts = monitor.get_resource_alerts()
        # With very high thresholds, should have no alerts
        assert isinstance(alerts, list)

    def test_snapshot_to_dict(self):
        """Test snapshot serialization to dict."""
        monitor = SystemHealthMonitor()
        snap = monitor.snapshot()
        d = snap.to_dict()

        assert isinstance(d, dict)
        assert "cpu_percent" in d
        assert "ram_used_mb" in d
        assert "health_score" in d


class TestPerformanceReport:
    """Test performance report aggregation and serialization."""

    def test_empty_report(self):
        """Test creating an empty report."""
        report = PerformanceReport()
        assert report.gpu_memory is None
        assert report.speed_metrics is None
        assert report.system_health is None
        assert report.recommendations == []
        assert report.overall_status == HealthStatus.HEALTHY.value

    def test_report_to_dict(self):
        """Test report serialization to dictionary."""
        report = PerformanceReport(
            recommendations=["Reduce batch size"],
            bottleneck="backward_pass",
            overall_status=HealthStatus.WARNING.value,
        )
        d = report.to_dict()

        assert isinstance(d, dict)
        assert d["bottleneck"] == "backward_pass"
        assert d["overall_status"] == "warning"
        assert "Reduce batch size" in d["recommendations"]

    def test_report_to_json(self):
        """Test report JSON serialization."""
        report = PerformanceReport(
            recommendations=["Enable mixed precision"],
            bottleneck="forward_pass",
        )
        json_str = report.to_json()

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert parsed["bottleneck"] == "forward_pass"
        assert "Enable mixed precision" in parsed["recommendations"]

    def test_report_save(self):
        """Test saving report to file."""
        report = PerformanceReport(
            recommendations=["Test recommendation"],
            overall_status=HealthStatus.HEALTHY.value,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "report.json"
            report.save(filepath)

            assert filepath.exists()
            with open(filepath) as f:
                loaded = json.load(f)
            assert "Test recommendation" in loaded["recommendations"]

    def test_report_with_full_data(self):
        """Test report with all metrics populated."""
        gpu_snap = GPUMemorySnapshot(
            timestamp=time.time(),
            allocated_mb=5000.0,
            reserved_mb=6000.0,
            free_mb=9000.0,
            total_mb=15000.0,
            peak_mb=7000.0,
            utilization_percent=40.0,
        )
        speed = SpeedMetrics(
            timestamp=time.time(),
            steps_per_second=2.5,
            samples_per_second=10.0,
            data_loading_time_ms=50.0,
            forward_pass_time_ms=150.0,
            backward_pass_time_ms=200.0,
            optimizer_step_time_ms=20.0,
            eta_seconds=3600.0,
        )
        health = SystemHealthSnapshot(
            timestamp=time.time(),
            cpu_percent=45.0,
            cpu_count=8,
            ram_used_mb=8000.0,
            ram_total_mb=16000.0,
            ram_percent=50.0,
            disk_used_gb=200.0,
            disk_total_gb=500.0,
            disk_percent=40.0,
            gpu_temperature_celsius=65.0,
            health_score=75.0,
            status=HealthStatus.HEALTHY.value,
        )

        report = PerformanceReport(
            gpu_memory=gpu_snap,
            speed_metrics=speed,
            system_health=health,
            recommendations=["All systems nominal"],
            bottleneck="backward_pass",
            overall_status=HealthStatus.HEALTHY.value,
        )

        d = report.to_dict()
        assert d["gpu_memory"]["allocated_mb"] == 5000.0
        assert d["speed_metrics"]["steps_per_second"] == 2.5
        assert d["system_health"]["cpu_percent"] == 45.0


class TestPerformanceMonitor:
    """Test the unified performance monitor."""

    def test_initialization(self):
        """Test unified monitor initializes correctly."""
        monitor = PerformanceMonitor(
            poll_interval_seconds=10.0,
            total_training_steps=500,
        )
        assert monitor.poll_interval == 10.0
        assert monitor.speed_monitor.total_steps == 500

    def test_should_poll(self):
        """Test polling interval check."""
        monitor = PerformanceMonitor(poll_interval_seconds=60.0)

        # Should poll immediately (last poll time is 0)
        assert monitor.should_poll() is True

        # After generating a report, should not poll immediately
        monitor.generate_report()
        assert monitor.should_poll() is False

    def test_record_training_step(self):
        """Test recording training steps through unified monitor."""
        monitor = PerformanceMonitor()
        monitor.record_training_step(
            batch_size=4,
            data_load_time=0.05,
            forward_time=0.1,
            backward_time=0.15,
            optimizer_time=0.02,
        )
        assert monitor.speed_monitor._current_step == 1

    def test_generate_report(self):
        """Test generating a full performance report."""
        monitor = PerformanceMonitor(total_training_steps=100)

        # Record some steps
        for _ in range(10):
            monitor.record_training_step(
                batch_size=2,
                data_load_time=0.05,
                forward_time=0.1,
                backward_time=0.15,
                optimizer_time=0.02,
            )
            time.sleep(0.01)

        report = monitor.generate_report()

        assert isinstance(report, PerformanceReport)
        assert report.gpu_memory is not None
        assert report.speed_metrics is not None
        assert report.system_health is not None
        assert report.overall_status in [s.value for s in HealthStatus]

    def test_poll_returns_none_before_interval(self):
        """Test poll returns None when interval hasn't elapsed."""
        monitor = PerformanceMonitor(poll_interval_seconds=100.0)
        monitor.generate_report()  # Reset the timer

        result = monitor.poll()
        assert result is None

    def test_poll_returns_report_after_interval(self):
        """Test poll returns report when interval has elapsed."""
        monitor = PerformanceMonitor(poll_interval_seconds=0.05)

        time.sleep(0.1)
        result = monitor.poll()
        assert isinstance(result, PerformanceReport)

    def test_get_summary(self):
        """Test getting a concise performance summary."""
        monitor = PerformanceMonitor(total_training_steps=100)

        for _ in range(5):
            monitor.record_training_step(batch_size=4)
            time.sleep(0.01)

        summary = monitor.get_summary()

        assert "gpu_memory_percent" in summary
        assert "steps_per_second" in summary
        assert "samples_per_second" in summary
        assert "eta" in summary
        assert "health_score" in summary
        assert "health_status" in summary
        assert "bottleneck" in summary

    def test_report_history(self):
        """Test report history accumulation."""
        monitor = PerformanceMonitor(poll_interval_seconds=0.01)

        monitor.generate_report()
        monitor.generate_report()
        monitor.generate_report()

        history = monitor.get_report_history()
        assert len(history) == 3
        assert all(isinstance(r, dict) for r in history)

    def test_output_dir_report_saving(self):
        """Test that reports are saved when output_dir is configured."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = PerformanceMonitor(
                output_dir=Path(tmpdir),
                poll_interval_seconds=0.01,
            )

            monitor.generate_report()

            report_path = Path(tmpdir) / "latest_performance_report.json"
            assert report_path.exists()

            with open(report_path) as f:
                data = json.load(f)
            assert "overall_status" in data
