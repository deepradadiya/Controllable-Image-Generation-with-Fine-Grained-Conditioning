"""
Unit tests for deployment optimization module.

Tests cover:
- ModelOptimizer: quantization, torch.compile, ONNX export, attention optimization
- InferenceCache: LRU caching, TTL expiration, eviction, stats tracking
- ConcurrencyManager: request queuing, priority scheduling, degradation, cleanup
"""

import sys
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn
import numpy as np
import pytest

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.deployment.optimization import (
    ModelOptimizer,
    InferenceCache,
    ConcurrencyManager,
    OptimizationConfig,
    CacheConfig,
    ConcurrencyConfig,
    QuantizationType,
    RequestPriority,
    RequestStatus,
    InferenceRequest,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class SimpleModel(nn.Module):
    """Simple model for testing optimization."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 64)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


@pytest.fixture
def simple_model():
    return SimpleModel()


@pytest.fixture
def optimizer():
    return ModelOptimizer(OptimizationConfig())


@pytest.fixture
def cache():
    return InferenceCache(
        CacheConfig(max_text_embeddings=4, max_condition_maps=4, max_latent_cache=4)
    )


@pytest.fixture
def manager():
    return ConcurrencyManager(
        ConcurrencyConfig(max_concurrent_requests=2, max_queue_size=5)
    )


# =============================================================================
# ModelOptimizer Tests
# =============================================================================


class TestModelOptimizer:
    """Tests for ModelOptimizer class."""

    def test_initialization_default_config(self):
        """ModelOptimizer initializes with default config."""
        opt = ModelOptimizer()
        assert opt.config.quantization == QuantizationType.FP16
        assert opt.config.enable_torch_compile is True

    def test_initialization_custom_config(self):
        """ModelOptimizer accepts custom configuration."""
        config = OptimizationConfig(
            quantization=QuantizationType.INT8,
            enable_torch_compile=False,
        )
        opt = ModelOptimizer(config)
        assert opt.config.quantization == QuantizationType.INT8
        assert opt.config.enable_torch_compile is False

    def test_fp32_no_quantization(self, optimizer, simple_model):
        """FP32 quantization returns model unchanged."""
        result = optimizer.optimize_model(simple_model, QuantizationType.FP32)
        assert next(result.parameters()).dtype == torch.float32

    def test_int8_quantization_graceful_fallback(self, simple_model):
        """INT8 quantization handles unsupported platforms gracefully."""
        opt = ModelOptimizer(OptimizationConfig(quantization=QuantizationType.INT8))
        # Should not raise, even on platforms without quantization support
        result = opt.optimize_model(simple_model, QuantizationType.INT8)
        assert result is not None

    def test_torch_compile_available_check(self, optimizer):
        """torch.compile availability is correctly detected."""
        major = int(torch.__version__.split(".")[0])
        assert optimizer._torch_compile_available == (major >= 2)

    def test_torch_compile_disabled(self, simple_model):
        """torch.compile returns original model when disabled."""
        opt = ModelOptimizer(OptimizationConfig(enable_torch_compile=False))
        result = opt.apply_torch_compile(simple_model)
        # When disabled, should return the same model object
        assert result is simple_model

    def test_torch_compile_enabled(self, simple_model):
        """torch.compile wraps model when enabled and available."""
        opt = ModelOptimizer(OptimizationConfig(enable_torch_compile=True))
        if opt._torch_compile_available:
            result = opt.apply_torch_compile(simple_model)
            # Should return a compiled wrapper
            assert result is not None

    def test_onnx_export_disabled(self, optimizer, simple_model):
        """ONNX export returns None when disabled and no path given."""
        config = OptimizationConfig(enable_onnx_export=False)
        opt = ModelOptimizer(config)
        result = opt.export_onnx(simple_model, {"input": torch.randn(1, 64)})
        assert result is None

    def test_optimization_report(self, optimizer, simple_model):
        """Optimization report contains expected keys."""
        report = optimizer.get_optimization_report(simple_model)
        assert "quantization" in report
        assert "model_size_mb" in report
        assert "parameter_count" in report
        assert "device" in report
        assert "dtype" in report
        assert report["parameter_count"] == sum(
            p.numel() for p in simple_model.parameters()
        )

    def test_model_size_calculation(self, simple_model):
        """Model size calculation returns reasonable value."""
        size = ModelOptimizer._get_model_size_mb(simple_model)
        assert size > 0
        # SimpleModel has ~17K params * 4 bytes = ~68KB
        assert size < 1.0  # Should be well under 1MB

    def test_unsupported_quantization_raises(self, optimizer, simple_model):
        """Invalid quantization type raises ValueError."""
        with pytest.raises(ValueError):
            optimizer.optimize_model(simple_model, "invalid_type")


# =============================================================================
# InferenceCache Tests
# =============================================================================


class TestInferenceCache:
    """Tests for InferenceCache class."""

    def test_initialization(self, cache):
        """Cache initializes with correct configuration."""
        assert len(cache._text_embedding_cache) == 0
        assert len(cache._condition_map_cache) == 0
        assert len(cache._latent_cache) == 0

    def test_text_embedding_put_and_get(self, cache):
        """Text embeddings can be stored and retrieved."""
        embedding = torch.randn(1, 77, 768)
        cache.put_text_embedding("test prompt", embedding)
        retrieved = cache.get_text_embedding("test prompt")
        assert retrieved is not None
        assert torch.allclose(retrieved, embedding)

    def test_text_embedding_cache_miss(self, cache):
        """Cache miss returns None."""
        result = cache.get_text_embedding("nonexistent")
        assert result is None

    def test_condition_map_put_and_get(self, cache):
        """Condition maps can be stored and retrieved."""
        condition = torch.randn(1, 3, 256, 256)
        cache.put_condition_map("hash123", "depth", (256, 256), condition)
        retrieved = cache.get_condition_map("hash123", "depth", (256, 256))
        assert retrieved is not None
        assert torch.allclose(retrieved, condition)

    def test_condition_map_different_types(self, cache):
        """Different condition types are cached separately."""
        depth = torch.randn(1, 3, 256, 256)
        edge = torch.randn(1, 3, 256, 256)
        cache.put_condition_map("hash123", "depth", (256, 256), depth)
        cache.put_condition_map("hash123", "edge", (256, 256), edge)

        retrieved_depth = cache.get_condition_map("hash123", "depth", (256, 256))
        retrieved_edge = cache.get_condition_map("hash123", "edge", (256, 256))
        assert torch.allclose(retrieved_depth, depth)
        assert torch.allclose(retrieved_edge, edge)

    def test_latent_cache_put_and_get(self, cache):
        """Latents can be stored and retrieved with full parameter matching."""
        latent = torch.randn(1, 4, 64, 64)
        cache.put_latent("prompt", "cond_hash", 42, 20, 7.5, 1.0, latent)
        retrieved = cache.get_latent("prompt", "cond_hash", 42, 20, 7.5, 1.0)
        assert retrieved is not None
        assert torch.allclose(retrieved, latent)

    def test_latent_cache_different_seeds(self, cache):
        """Different seeds produce different cache keys."""
        latent1 = torch.randn(1, 4, 64, 64)
        latent2 = torch.randn(1, 4, 64, 64)
        cache.put_latent("prompt", "cond", 42, 20, 7.5, 1.0, latent1)
        cache.put_latent("prompt", "cond", 99, 20, 7.5, 1.0, latent2)

        r1 = cache.get_latent("prompt", "cond", 42, 20, 7.5, 1.0)
        r2 = cache.get_latent("prompt", "cond", 99, 20, 7.5, 1.0)
        assert torch.allclose(r1, latent1)
        assert torch.allclose(r2, latent2)

    def test_lru_eviction(self, cache):
        """Oldest entries are evicted when cache is full."""
        # Cache has max_text_embeddings=4
        for i in range(5):
            cache.put_text_embedding(f"prompt_{i}", torch.randn(1, 77, 768))

        # First entry should be evicted
        assert cache.get_text_embedding("prompt_0") is None
        # Last entries should still be present
        assert cache.get_text_embedding("prompt_4") is not None
        assert cache.get_text_embedding("prompt_3") is not None

    def test_ttl_expiration(self):
        """Entries expire after TTL."""
        config = CacheConfig(
            max_text_embeddings=10,
            text_embedding_ttl_seconds=0.01,  # 10ms TTL
        )
        cache = InferenceCache(config)
        cache.put_text_embedding("test", torch.randn(1, 77, 768))

        # Should be available immediately
        assert cache.get_text_embedding("test") is not None

        # Wait for TTL to expire
        time.sleep(0.02)
        assert cache.get_text_embedding("test") is None

    def test_cache_stats(self, cache):
        """Stats correctly track hits and misses."""
        cache.put_text_embedding("exists", torch.randn(1, 77, 768))
        cache.get_text_embedding("exists")  # hit
        cache.get_text_embedding("missing")  # miss

        stats = cache.get_stats()
        assert stats["text_hits"] == 1
        assert stats["text_misses"] == 1
        assert stats["text_hit_rate"] == 0.5

    def test_cache_clear(self, cache):
        """Clear removes all entries and resets stats."""
        cache.put_text_embedding("test", torch.randn(1, 77, 768))
        cache.get_text_embedding("test")
        cache.clear()

        assert cache.get_text_embedding("test") is None
        stats = cache.get_stats()
        # After clear, stats are reset. The get above adds 1 miss.
        assert stats["text_hits"] == 0
        assert stats["text_misses"] == 1  # The get after clear counts as a miss

    def test_memory_usage(self, cache):
        """Memory usage is tracked correctly."""
        # Empty cache
        assert cache.get_memory_usage_mb() == 0.0

        # Add a tensor
        embedding = torch.randn(1, 77, 768)  # ~236KB
        cache.put_text_embedding("test", embedding)
        mem = cache.get_memory_usage_mb()
        assert mem > 0

    def test_image_hash_deterministic(self):
        """Image hash is deterministic for same content."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        hash1 = InferenceCache.compute_image_hash(img)
        hash2 = InferenceCache.compute_image_hash(img)
        assert hash1 == hash2

    def test_image_hash_different_for_different_images(self):
        """Different images produce different hashes."""
        img1 = np.zeros((64, 64, 3), dtype=np.uint8)
        img2 = np.ones((64, 64, 3), dtype=np.uint8)
        hash1 = InferenceCache.compute_image_hash(img1)
        hash2 = InferenceCache.compute_image_hash(img2)
        assert hash1 != hash2

    def test_thread_safety(self, cache):
        """Cache handles concurrent access safely."""
        errors = []

        def writer():
            try:
                for i in range(50):
                    cache.put_text_embedding(
                        f"thread_prompt_{i}", torch.randn(1, 77, 768)
                    )
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(50):
                    cache.get_text_embedding(f"thread_prompt_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"


# =============================================================================
# ConcurrencyManager Tests
# =============================================================================


class TestConcurrencyManager:
    """Tests for ConcurrencyManager class."""

    def test_initialization(self, manager):
        """Manager initializes with correct configuration."""
        stats = manager.get_stats()
        assert stats["queue_size"] == 0
        assert stats["active_requests"] == 0
        assert stats["total_processed"] == 0

    def test_submit_request(self, manager):
        """Requests can be submitted to the queue."""
        req = manager.submit_request({"prompt": "test"})
        assert req is not None
        assert req.status == RequestStatus.QUEUED
        assert req.params == {"prompt": "test"}

    def test_submit_request_queue_full(self, manager):
        """Requests are rejected when queue is full."""
        # Fill the queue (max_queue_size=5)
        for i in range(5):
            manager.submit_request({"prompt": f"test_{i}"})

        # Next request should be rejected
        rejected = manager.submit_request({"prompt": "overflow"})
        assert rejected is None

    def test_priority_ordering(self, manager):
        """High priority requests are processed first."""
        manager.submit_request({"prompt": "low"}, RequestPriority.LOW)
        manager.submit_request({"prompt": "high"}, RequestPriority.HIGH)
        manager.submit_request({"prompt": "normal"}, RequestPriority.NORMAL)

        params = manager.acquire_slot()
        assert params["prompt"] == "high"

    def test_can_process(self, manager):
        """can_process returns True when slots and requests are available."""
        assert not manager.can_process()  # No requests in queue

        manager.submit_request({"prompt": "test"})
        assert manager.can_process()

    def test_acquire_and_release_slot(self, manager):
        """Slots can be acquired and released."""
        manager.submit_request({"prompt": "test", "num_inference_steps": 20})
        params = manager.acquire_slot()
        assert params is not None
        assert params["prompt"] == "test"

        stats = manager.get_stats()
        assert stats["active_requests"] == 1

    def test_release_slot_completed(self, manager):
        """Released slots mark requests as completed."""
        req = manager.submit_request({"prompt": "test"})
        manager.acquire_slot(req)
        manager.release_slot(req.request_id, result="image_data")

        status = manager.get_request_status(req.request_id)
        assert status == RequestStatus.COMPLETED

    def test_release_slot_failed(self, manager):
        """Failed requests are marked correctly."""
        req = manager.submit_request({"prompt": "test"})
        manager.acquire_slot(req)
        manager.release_slot(req.request_id, error="OOM error")

        status = manager.get_request_status(req.request_id)
        assert status == RequestStatus.FAILED

    def test_queue_position(self, manager):
        """Queue position is correctly reported."""
        req1 = manager.submit_request({"prompt": "first"})
        req2 = manager.submit_request({"prompt": "second"})

        assert manager.get_queue_position(req1.request_id) == 0
        assert manager.get_queue_position(req2.request_id) == 1

    def test_estimated_wait_time(self, manager):
        """Wait time estimation is reasonable."""
        req = manager.submit_request({"prompt": "test"})
        wait = manager.get_estimated_wait_time(req.request_id)
        assert wait is not None
        assert wait > 0

    def test_cleanup_expired(self):
        """Expired requests are cleaned up."""
        config = ConcurrencyConfig(
            max_concurrent_requests=2,
            max_queue_size=5,
            request_timeout_seconds=0.01,
        )
        manager = ConcurrencyManager(config)
        manager.submit_request({"prompt": "will_expire"})

        time.sleep(0.02)
        removed = manager.cleanup_expired()
        assert removed == 1
        assert manager.get_stats()["queue_size"] == 0

    def test_concurrent_request_limit(self, manager):
        """Cannot exceed max concurrent requests."""
        # Submit and acquire max concurrent (2)
        req1 = manager.submit_request({"prompt": "r1"})
        req2 = manager.submit_request({"prompt": "r2"})
        req3 = manager.submit_request({"prompt": "r3"})

        manager.acquire_slot(req1)
        manager.acquire_slot(req2)

        # Third acquire should fail (at max concurrent)
        params = manager.acquire_slot(req3)
        assert params is None

    def test_stats_tracking(self, manager):
        """Statistics are correctly tracked."""
        req = manager.submit_request({"prompt": "test"})
        manager.acquire_slot(req)
        manager.release_slot(req.request_id, result="done")

        stats = manager.get_stats()
        assert stats["total_processed"] == 1
        assert stats["active_requests"] == 0

    def test_unknown_request_status(self, manager):
        """Unknown request ID returns None."""
        assert manager.get_request_status("nonexistent") is None

    def test_unknown_request_queue_position(self, manager):
        """Unknown request ID returns None for queue position."""
        assert manager.get_queue_position("nonexistent") is None
