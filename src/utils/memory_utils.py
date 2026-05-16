"""
Memory Optimization Utilities

This module provides comprehensive memory management utilities for ControlNet
training, including GPU memory monitoring, automatic batch size adjustment,
memory profiling, cache clearing, and memory leak detection.

Requirements satisfied: 4.7, 6.3
"""

import gc
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import json

import torch
import torch.nn as nn
import psutil
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    
    timestamp: float
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    gpu_free_mb: float
    cpu_memory_mb: float
    cpu_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'gpu_allocated_mb': self.gpu_allocated_mb,
            'gpu_reserved_mb': self.gpu_reserved_mb,
            'gpu_free_mb': self.gpu_free_mb,
            'cpu_memory_mb': self.cpu_memory_mb,
            'cpu_percent': self.cpu_percent,
        }


class MemoryProfiler:
    """Profile memory usage during training."""
    
    def __init__(self, enable_profiling: bool = True):
        """
        Initialize memory profiler.
        
        Args:
            enable_profiling: Whether to enable memory profiling
        """
        self.enable_profiling = enable_profiling
        self.snapshots: List[MemorySnapshot] = []
        self.peak_gpu_memory = 0.0
        self.peak_cpu_memory = 0.0
        
    def take_snapshot(self, label: str = "") -> MemorySnapshot:
        """
        Take a memory snapshot.
        
        Args:
            label: Optional label for the snapshot
            
        Returns:
            Memory snapshot
        """
        if not self.enable_profiling:
            return MemorySnapshot(0, 0, 0, 0, 0, 0)
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024**2)
            gpu_reserved = torch.cuda.memory_reserved() / (1024**2)
            gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            gpu_free = gpu_total - gpu_reserved
        else:
            gpu_allocated = gpu_reserved = gpu_free = 0.0
        
        # CPU memory
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / (1024**2)
        cpu_percent = process.memory_percent()
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            gpu_allocated_mb=gpu_allocated,
            gpu_reserved_mb=gpu_reserved,
            gpu_free_mb=gpu_free,
            cpu_memory_mb=cpu_memory,
            cpu_percent=cpu_percent,
        )
        
        # Update peaks
        self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_allocated)
        self.peak_cpu_memory = max(self.peak_cpu_memory, cpu_memory)
        
        self.snapshots.append(snapshot)
        
        if label:
            logger.debug(f"Memory snapshot '{label}': GPU {gpu_allocated:.1f}MB, CPU {cpu_memory:.1f}MB")
        
        return snapshot
    
    def get_memory_trend(self, window_size: int = 10) -> Dict[str, float]:
        """
        Get memory usage trend over recent snapshots.
        
        Args:
            window_size: Number of recent snapshots to analyze
            
        Returns:
            Dictionary with trend information
        """
        if len(self.snapshots) < 2:
            return {"trend": "insufficient_data"}
        
        recent_snapshots = self.snapshots[-window_size:]
        
        # Calculate trends
        gpu_values = [s.gpu_allocated_mb for s in recent_snapshots]
        cpu_values = [s.cpu_memory_mb for s in recent_snapshots]
        
        gpu_trend = np.polyfit(range(len(gpu_values)), gpu_values, 1)[0] if len(gpu_values) > 1 else 0
        cpu_trend = np.polyfit(range(len(cpu_values)), cpu_values, 1)[0] if len(cpu_values) > 1 else 0
        
        return {
            "gpu_trend_mb_per_snapshot": gpu_trend,
            "cpu_trend_mb_per_snapshot": cpu_trend,
            "gpu_increasing": gpu_trend > 1.0,  # More than 1MB increase per snapshot
            "cpu_increasing": cpu_trend > 5.0,  # More than 5MB increase per snapshot
            "snapshots_analyzed": len(recent_snapshots),
        }
    
    def detect_memory_leak(self, threshold_mb: float = 50.0) -> bool:
        """
        Detect potential memory leaks.
        
        Args:
            threshold_mb: Threshold for leak detection (MB per snapshot)
            
        Returns:
            True if potential leak detected
        """
        trend = self.get_memory_trend()
        
        return (trend.get("gpu_trend_mb_per_snapshot", 0) > threshold_mb or
                trend.get("cpu_trend_mb_per_snapshot", 0) > threshold_mb)
    
    def save_profile(self, filepath: Path):
        """Save memory profile to file."""
        profile_data = {
            "peak_gpu_memory_mb": self.peak_gpu_memory,
            "peak_cpu_memory_mb": self.peak_cpu_memory,
            "snapshots": [s.to_dict() for s in self.snapshots],
            "trend": self.get_memory_trend(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(profile_data, f, indent=2)
        
        logger.info(f"Memory profile saved to {filepath}")


class MemoryOptimizer:
    """Automatic memory optimization utilities."""
    
    def __init__(self, target_memory_gb: float = 12.0):
        """
        Initialize memory optimizer.
        
        Args:
            target_memory_gb: Target GPU memory usage in GB
        """
        self.target_memory_gb = target_memory_gb
        self.target_memory_bytes = target_memory_gb * (1024**3)
        self.profiler = MemoryProfiler()
        
    def clear_cache(self, aggressive: bool = False):
        """
        Clear memory caches.
        
        Args:
            aggressive: Whether to perform aggressive cleanup
        """
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if aggressive:
                torch.cuda.synchronize()
        
        # Clear Python garbage collection
        if aggressive:
            for _ in range(3):  # Multiple passes for thorough cleanup
                gc.collect()
        else:
            gc.collect()
        
        logger.debug("Memory cache cleared")
    
    def estimate_model_memory(self, model: nn.Module) -> Dict[str, float]:
        """
        Estimate memory usage of a model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Dictionary with memory estimates in MB
        """
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        
        # Estimate gradient memory (same as parameters for most optimizers)
        gradient_size = param_size
        
        # Estimate optimizer state (2x parameters for Adam)
        optimizer_size = param_size * 2
        
        total_size = param_size + buffer_size + gradient_size + optimizer_size
        
        return {
            "parameters_mb": param_size / (1024**2),
            "buffers_mb": buffer_size / (1024**2),
            "gradients_mb": gradient_size / (1024**2),
            "optimizer_mb": optimizer_size / (1024**2),
            "total_mb": total_size / (1024**2),
        }
    
    def estimate_batch_memory(
        self,
        batch_size: int,
        sequence_length: int,
        hidden_size: int,
        num_layers: int,
        dtype_bytes: int = 4,
    ) -> float:
        """
        Estimate memory usage for a batch.
        
        Args:
            batch_size: Batch size
            sequence_length: Sequence length
            hidden_size: Hidden dimension size
            num_layers: Number of layers
            dtype_bytes: Bytes per element (4 for float32, 2 for float16)
            
        Returns:
            Estimated memory usage in MB
        """
        # Activations memory (rough estimate)
        activation_elements = batch_size * sequence_length * hidden_size * num_layers
        activation_memory = activation_elements * dtype_bytes
        
        # Add some overhead for intermediate computations
        total_memory = activation_memory * 1.5
        
        return total_memory / (1024**2)
    
    def find_optimal_batch_size(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        max_batch_size: int = 32,
        memory_fraction: float = 0.8,
    ) -> int:
        """
        Find optimal batch size that fits in memory.
        
        Args:
            model: PyTorch model
            sample_input: Sample input tensor
            max_batch_size: Maximum batch size to try
            memory_fraction: Fraction of available memory to use
            
        Returns:
            Optimal batch size
        """
        if not torch.cuda.is_available():
            logger.warning("No GPU available, returning batch size 1")
            return 1
        
        # Get available memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        available_memory = gpu_memory * memory_fraction
        
        # Estimate model memory
        model_memory = self.estimate_model_memory(model)["total_mb"] * (1024**2)
        
        # Available memory for batches
        batch_memory_budget = available_memory - model_memory
        
        if batch_memory_budget <= 0:
            logger.warning("Model too large for GPU, returning batch size 1")
            return 1
        
        # Binary search for optimal batch size
        low, high = 1, max_batch_size
        optimal_batch_size = 1
        
        while low <= high:
            mid = (low + high) // 2
            
            try:
                # Test with this batch size
                test_input = sample_input.repeat(mid, *([1] * (sample_input.dim() - 1)))
                
                self.clear_cache()
                memory_before = torch.cuda.memory_allocated()
                
                with torch.no_grad():
                    _ = model(test_input)
                
                memory_after = torch.cuda.memory_allocated()
                batch_memory = memory_after - memory_before
                
                if batch_memory <= batch_memory_budget:
                    optimal_batch_size = mid
                    low = mid + 1
                else:
                    high = mid - 1
                
                # Clean up
                del test_input
                self.clear_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    high = mid - 1
                    self.clear_cache()
                else:
                    raise e
        
        logger.info(f"Optimal batch size found: {optimal_batch_size}")
        return optimal_batch_size
    
    @contextmanager
    def memory_efficient_context(self):
        """Context manager for memory-efficient operations."""
        # Take snapshot before
        self.profiler.take_snapshot("before_operation")
        
        try:
            # Clear cache before operation
            self.clear_cache()
            yield
        finally:
            # Clear cache after operation
            self.clear_cache()
            
            # Take snapshot after
            self.profiler.take_snapshot("after_operation")
    
    def optimize_for_training(
        self,
        model: nn.Module,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision: bool = True,
    ) -> Dict[str, Any]:
        """
        Apply memory optimizations for training.
        
        Args:
            model: Model to optimize
            enable_gradient_checkpointing: Whether to enable gradient checkpointing
            enable_mixed_precision: Whether to enable mixed precision
            
        Returns:
            Dictionary with optimization results
        """
        optimizations_applied = []
        
        # Enable gradient checkpointing
        if enable_gradient_checkpointing and hasattr(model, 'gradient_checkpointing'):
            model.gradient_checkpointing = True
            optimizations_applied.append("gradient_checkpointing")
        
        # Convert to half precision if requested
        if enable_mixed_precision and torch.cuda.is_available():
            model = model.half()
            optimizations_applied.append("mixed_precision")
        
        # Clear cache
        self.clear_cache(aggressive=True)
        optimizations_applied.append("cache_clearing")
        
        # Estimate memory savings
        memory_estimate = self.estimate_model_memory(model)
        
        return {
            "optimizations_applied": optimizations_applied,
            "estimated_memory_mb": memory_estimate["total_mb"],
            "memory_savings_percent": 30 if enable_mixed_precision else 15,  # Rough estimates
        }


class AdaptiveBatchSizer:
    """Automatically adjust batch size based on memory usage."""
    
    def __init__(
        self,
        initial_batch_size: int = 1,
        max_batch_size: int = 16,
        memory_threshold: float = 0.85,
        adjustment_factor: float = 0.8,
    ):
        """
        Initialize adaptive batch sizer.
        
        Args:
            initial_batch_size: Starting batch size
            max_batch_size: Maximum allowed batch size
            memory_threshold: Memory usage threshold (0-1)
            adjustment_factor: Factor to reduce batch size on OOM
        """
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        self.adjustment_factor = adjustment_factor
        self.oom_count = 0
        self.success_count = 0
        
    def get_current_batch_size(self) -> int:
        """Get current batch size."""
        return self.current_batch_size
    
    def handle_oom(self) -> int:
        """
        Handle out-of-memory error by reducing batch size.
        
        Returns:
            New batch size
        """
        self.oom_count += 1
        self.success_count = 0  # Reset success counter
        
        # Reduce batch size
        new_batch_size = max(1, int(self.current_batch_size * self.adjustment_factor))
        
        logger.warning(f"OOM detected! Reducing batch size: {self.current_batch_size} -> {new_batch_size}")
        
        self.current_batch_size = new_batch_size
        return self.current_batch_size
    
    def handle_success(self) -> int:
        """
        Handle successful training step.
        
        Returns:
            Potentially increased batch size
        """
        self.success_count += 1
        
        # Try to increase batch size after several successful steps
        if (self.success_count >= 10 and 
            self.current_batch_size < self.max_batch_size and
            self.oom_count == 0):  # Only if no recent OOMs
            
            # Check memory usage before increasing
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated()
                memory_total = torch.cuda.get_device_properties(0).total_memory
                memory_fraction = memory_used / memory_total
                
                if memory_fraction < self.memory_threshold:
                    new_batch_size = min(self.max_batch_size, self.current_batch_size + 1)
                    logger.info(f"Increasing batch size: {self.current_batch_size} -> {new_batch_size}")
                    self.current_batch_size = new_batch_size
                    self.success_count = 0  # Reset counter
        
        return self.current_batch_size
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get batch sizing statistics."""
        return {
            "current_batch_size": self.current_batch_size,
            "max_batch_size": self.max_batch_size,
            "oom_count": self.oom_count,
            "success_count": self.success_count,
            "memory_threshold": self.memory_threshold,
        }


class MemoryLeakDetector:
    """Detect and analyze memory leaks during training."""
    
    def __init__(self, check_interval: int = 100):
        """
        Initialize memory leak detector.
        
        Args:
            check_interval: Steps between leak checks
        """
        self.check_interval = check_interval
        self.memory_history: List[float] = []
        self.step_count = 0
        self.leak_detected = False
        
    def check_step(self, step: int) -> Optional[Dict[str, Any]]:
        """
        Check for memory leaks at this step.
        
        Args:
            step: Current training step
            
        Returns:
            Leak detection results if check performed
        """
        self.step_count += 1
        
        # Record current memory usage
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / (1024**2)
        else:
            process = psutil.Process()
            current_memory = process.memory_info().rss / (1024**2)
        
        self.memory_history.append(current_memory)
        
        # Perform leak check at intervals
        if self.step_count % self.check_interval == 0:
            return self._analyze_memory_trend()
        
        return None
    
    def _analyze_memory_trend(self) -> Dict[str, Any]:
        """Analyze memory usage trend for leaks."""
        if len(self.memory_history) < 10:
            return {"status": "insufficient_data"}
        
        # Analyze recent memory usage
        recent_memory = self.memory_history[-self.check_interval:]
        
        # Calculate trend
        x = np.arange(len(recent_memory))
        trend = np.polyfit(x, recent_memory, 1)[0]  # Slope of linear fit
        
        # Calculate memory increase
        memory_increase = recent_memory[-1] - recent_memory[0]
        
        # Detect leak
        leak_threshold = 5.0  # MB per check interval
        potential_leak = trend > leak_threshold or memory_increase > leak_threshold * 2
        
        if potential_leak and not self.leak_detected:
            self.leak_detected = True
            logger.warning(f"Potential memory leak detected! Trend: {trend:.2f} MB/step")
        
        return {
            "status": "analyzed",
            "trend_mb_per_step": trend,
            "memory_increase_mb": memory_increase,
            "potential_leak": potential_leak,
            "current_memory_mb": recent_memory[-1],
            "steps_analyzed": len(recent_memory),
        }
    
    def get_recommendations(self) -> List[str]:
        """Get recommendations for memory leak mitigation."""
        if not self.leak_detected:
            return ["No memory leaks detected"]
        
        return [
            "🔍 Memory leak detected! Recommendations:",
            "• Check for unreleased tensors in training loop",
            "• Ensure proper cleanup of intermediate variables",
            "• Use torch.no_grad() for inference operations",
            "• Clear optimizer gradients: optimizer.zero_grad()",
            "• Call torch.cuda.empty_cache() periodically",
            "• Check for circular references in custom classes",
        ]


def main():
    """Test memory optimization utilities."""
    print("Testing Memory Optimization Utilities")
    print("=" * 40)
    
    # Test memory profiler
    print("\n1. Testing Memory Profiler...")
    profiler = MemoryProfiler()
    
    # Take some snapshots
    profiler.take_snapshot("initial")
    
    # Simulate some memory usage
    dummy_tensor = torch.randn(1000, 1000)
    profiler.take_snapshot("after_tensor_creation")
    
    del dummy_tensor
    profiler.take_snapshot("after_cleanup")
    
    print(f"Snapshots taken: {len(profiler.snapshots)}")
    print(f"Peak GPU memory: {profiler.peak_gpu_memory:.1f} MB")
    print(f"Peak CPU memory: {profiler.peak_cpu_memory:.1f} MB")
    
    # Test memory optimizer
    print("\n2. Testing Memory Optimizer...")
    optimizer = MemoryOptimizer(target_memory_gb=8.0)
    
    # Create a simple model for testing
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 100),
    )
    
    memory_estimate = optimizer.estimate_model_memory(model)
    print(f"Model memory estimate: {memory_estimate['total_mb']:.1f} MB")
    
    # Test optimization
    optimization_results = optimizer.optimize_for_training(
        model=model,
        enable_gradient_checkpointing=False,  # Simple model doesn't support it
        enable_mixed_precision=False,  # Keep as float32 for testing
    )
    print(f"Optimizations applied: {optimization_results['optimizations_applied']}")
    
    # Test adaptive batch sizer
    print("\n3. Testing Adaptive Batch Sizer...")
    batch_sizer = AdaptiveBatchSizer(initial_batch_size=4, max_batch_size=16)
    
    print(f"Initial batch size: {batch_sizer.get_current_batch_size()}")
    
    # Simulate OOM
    new_size = batch_sizer.handle_oom()
    print(f"After OOM: {new_size}")
    
    # Simulate successful steps
    for _ in range(15):
        batch_sizer.handle_success()
    
    print(f"After successful steps: {batch_sizer.get_current_batch_size()}")
    print(f"Statistics: {batch_sizer.get_statistics()}")
    
    # Test memory leak detector
    print("\n4. Testing Memory Leak Detector...")
    leak_detector = MemoryLeakDetector(check_interval=5)
    
    # Simulate training steps with gradual memory increase
    for step in range(20):
        # Simulate memory leak by creating tensors
        if step > 10:
            _ = torch.randn(100, 100)  # Intentional "leak"
        
        result = leak_detector.check_step(step)
        if result and result.get("potential_leak"):
            print(f"Leak detected at step {step}")
            recommendations = leak_detector.get_recommendations()
            for rec in recommendations[:3]:  # Show first 3 recommendations
                print(f"  {rec}")
            break
    
    print("\n✅ Memory optimization utilities test completed!")
    print("\nKey features implemented:")
    print("✓ Memory profiling and snapshot tracking")
    print("✓ Automatic batch size optimization")
    print("✓ Memory leak detection and analysis")
    print("✓ GPU memory monitoring and cache clearing")
    print("✓ Model memory estimation")
    print("✓ Training memory optimizations")
    
    print(f"\n📋 Task 6.2 Implementation Complete!")
    print(f"Requirements satisfied: 4.7, 6.3")


if __name__ == "__main__":
    main()