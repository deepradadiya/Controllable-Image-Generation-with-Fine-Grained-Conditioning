"""
Unit tests for the ModelManager class.

Tests cover:
- ModelManager initialization and configuration
- Lazy loading behavior
- Model caching (avoids reloading on each request)
- Model switching between condition types
- Memory management (unloading unused models)
- Error handling with user-friendly messages
- Status reporting
"""

import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
import unittest

import torch

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.app.model_manager import (
    ModelManager,
    ManagerConfig,
    ModelLoadError,
    ModelStatus,
    ModelInfo,
)
from src.inference.model_loader import ModelLoadResult, CompatibilityReport


class TestManagerConfig(unittest.TestCase):
    """Test ManagerConfig defaults and customization."""

    def test_default_config(self):
        """Default config should have sensible values for HF Spaces."""
        config = ManagerConfig()
        self.assertEqual(config.max_loaded_models, 1)
        self.assertTrue(config.lazy_load)
        self.assertEqual(config.torch_dtype, torch.float16)
        self.assertIn("depth", config.controlnet_model_ids)
        self.assertIn("pose", config.controlnet_model_ids)
        self.assertIn("edge", config.controlnet_model_ids)

    def test_custom_config(self):
        """Custom config values should be respected."""
        config = ManagerConfig(
            max_loaded_models=2,
            lazy_load=False,
            gpu_memory_limit_gb=8.0,
            cache_dir="/tmp/custom_cache",
        )
        self.assertEqual(config.max_loaded_models, 2)
        self.assertFalse(config.lazy_load)
        self.assertEqual(config.gpu_memory_limit_gb, 8.0)
        self.assertEqual(config.cache_dir, "/tmp/custom_cache")


class TestModelManagerInit(unittest.TestCase):
    """Test ModelManager initialization."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = ManagerConfig(
            cache_dir=f"{self.test_dir}/cache",
            lazy_load=True,
            device="cpu",
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_initialization_creates_model_info(self):
        """ModelManager should create ModelInfo for all supported condition types."""
        manager = ModelManager(self.config)
        self.assertIn("depth", manager._models)
        self.assertIn("pose", manager._models)
        self.assertIn("edge", manager._models)

    def test_initialization_all_models_not_loaded(self):
        """With lazy_load=True, no models should be loaded at init."""
        manager = ModelManager(self.config)
        for info in manager._models.values():
            self.assertEqual(info.status, ModelStatus.NOT_LOADED)

    def test_supported_condition_types(self):
        """get_supported_condition_types should return all configured types."""
        manager = ModelManager(self.config)
        types = manager.get_supported_condition_types()
        self.assertIn("depth", types)
        self.assertIn("pose", types)
        self.assertIn("edge", types)

    def test_is_ready_false_initially(self):
        """No model should be ready initially with lazy loading."""
        manager = ModelManager(self.config)
        self.assertFalse(manager.is_ready())
        self.assertFalse(manager.is_ready("depth"))

    def test_active_condition_type_none_initially(self):
        """Active condition type should be None before any loading."""
        manager = ModelManager(self.config)
        self.assertIsNone(manager.get_active_condition_type())


class TestModelManagerLazyLoading(unittest.TestCase):
    """Test lazy loading behavior."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = ManagerConfig(
            cache_dir=f"{self.test_dir}/cache",
            lazy_load=True,
            device="cpu",
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch.object(ModelManager, "_load_pipeline")
    @patch.object(ModelManager, "_load_controlnet")
    def test_get_pipeline_triggers_loading(self, mock_load_cn, mock_load_pipe):
        """get_pipeline should trigger model loading on first call."""
        mock_controlnet = MagicMock()
        mock_pipeline = MagicMock()
        mock_load_cn.return_value = mock_controlnet
        mock_load_pipe.return_value = mock_pipeline

        manager = ModelManager(self.config)
        pipeline = manager.get_pipeline("depth")

        mock_load_cn.assert_called_once_with("depth")
        mock_load_pipe.assert_called_once_with("depth", mock_controlnet)
        self.assertEqual(pipeline, mock_pipeline)

    @patch.object(ModelManager, "_load_pipeline")
    @patch.object(ModelManager, "_load_controlnet")
    def test_second_call_uses_cache(self, mock_load_cn, mock_load_pipe):
        """Second get_pipeline call should use cached pipeline."""
        mock_controlnet = MagicMock()
        mock_pipeline = MagicMock()
        mock_load_cn.return_value = mock_controlnet
        mock_load_pipe.return_value = mock_pipeline

        manager = ModelManager(self.config)

        # First call loads
        pipeline1 = manager.get_pipeline("depth")
        # Second call should use cache
        pipeline2 = manager.get_pipeline("depth")

        # Should only load once
        self.assertEqual(mock_load_cn.call_count, 1)
        self.assertEqual(mock_load_pipe.call_count, 1)
        self.assertEqual(pipeline1, pipeline2)


class TestModelManagerCaching(unittest.TestCase):
    """Test model caching behavior."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = ManagerConfig(
            cache_dir=f"{self.test_dir}/cache",
            lazy_load=True,
            device="cpu",
            max_loaded_models=2,
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch.object(ModelManager, "_load_pipeline")
    @patch.object(ModelManager, "_load_controlnet")
    def test_cached_pipeline_returned(self, mock_load_cn, mock_load_pipe):
        """A cached pipeline should be returned without reloading."""
        mock_controlnet = MagicMock()
        mock_pipeline = MagicMock()
        mock_load_cn.return_value = mock_controlnet
        mock_load_pipe.return_value = mock_pipeline

        manager = ModelManager(self.config)
        manager.get_pipeline("depth")

        # Verify it's cached
        self.assertTrue(manager.is_ready("depth"))

        # Get again - should not reload
        manager.get_pipeline("depth")
        self.assertEqual(mock_load_cn.call_count, 1)

    @patch.object(ModelManager, "_load_pipeline")
    @patch.object(ModelManager, "_load_controlnet")
    def test_status_updates_on_load(self, mock_load_cn, mock_load_pipe):
        """Model status should update to READY after successful load."""
        mock_load_cn.return_value = MagicMock()
        mock_load_pipe.return_value = MagicMock()

        manager = ModelManager(self.config)
        manager.get_pipeline("depth")

        self.assertEqual(manager._models["depth"].status, ModelStatus.READY)
        self.assertGreater(manager._models["depth"].last_used, 0)


class TestModelManagerSwitching(unittest.TestCase):
    """Test model switching between condition types."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = ManagerConfig(
            cache_dir=f"{self.test_dir}/cache",
            lazy_load=True,
            device="cpu",
            max_loaded_models=2,
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch.object(ModelManager, "_load_pipeline")
    @patch.object(ModelManager, "_load_controlnet")
    def test_switch_condition_type(self, mock_load_cn, mock_load_pipe):
        """switch_condition_type should load the new model."""
        mock_load_cn.return_value = MagicMock()
        mock_load_pipe.return_value = MagicMock()

        manager = ModelManager(self.config)
        manager.get_pipeline("depth")
        self.assertEqual(manager.get_active_condition_type(), "depth")

        manager.switch_condition_type("pose")
        self.assertEqual(manager.get_active_condition_type(), "pose")

    @patch.object(ModelManager, "_load_pipeline")
    @patch.object(ModelManager, "_load_controlnet")
    def test_switch_back_uses_cache(self, mock_load_cn, mock_load_pipe):
        """Switching back to a cached model should not reload."""
        mock_load_cn.return_value = MagicMock()
        mock_load_pipe.return_value = MagicMock()

        manager = ModelManager(self.config)
        manager.get_pipeline("depth")
        manager.get_pipeline("pose")

        # Switch back to depth - should use cache
        call_count_before = mock_load_cn.call_count
        manager.get_pipeline("depth")
        # Should not have loaded again since max_loaded_models=2
        self.assertEqual(mock_load_cn.call_count, call_count_before)


class TestModelManagerMemoryManagement(unittest.TestCase):
    """Test memory management and model unloading."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = ManagerConfig(
            cache_dir=f"{self.test_dir}/cache",
            lazy_load=True,
            device="cpu",
            max_loaded_models=1,  # Only allow 1 model at a time
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch.object(ModelManager, "_load_pipeline")
    @patch.object(ModelManager, "_load_controlnet")
    def test_lru_unloading(self, mock_load_cn, mock_load_pipe):
        """Loading a new model should unload the LRU model when at limit."""
        mock_load_cn.return_value = MagicMock()
        mock_load_pipe.return_value = MagicMock()

        manager = ModelManager(self.config)

        # Load depth
        manager.get_pipeline("depth")
        self.assertTrue(manager.is_ready("depth"))

        # Load pose - should unload depth (max_loaded_models=1)
        manager.get_pipeline("pose")
        self.assertTrue(manager.is_ready("pose"))
        self.assertFalse(manager.is_ready("depth"))

    def test_unload_specific_model(self):
        """unload should free a specific model."""
        manager = ModelManager(self.config)

        # Manually set up a loaded state
        manager._pipelines["depth"] = MagicMock()
        manager._controlnets["depth"] = MagicMock()
        manager._models["depth"].status = ModelStatus.READY

        manager.unload("depth")

        self.assertNotIn("depth", manager._pipelines)
        self.assertNotIn("depth", manager._controlnets)
        self.assertEqual(manager._models["depth"].status, ModelStatus.UNLOADED)

    def test_unload_all(self):
        """unload_all should free all models."""
        manager = ModelManager(self.config)

        # Manually set up loaded states
        for ctype in ["depth", "pose"]:
            manager._pipelines[ctype] = MagicMock()
            manager._controlnets[ctype] = MagicMock()
            manager._models[ctype].status = ModelStatus.READY

        manager.unload_all()

        self.assertEqual(len(manager._pipelines), 0)
        self.assertEqual(len(manager._controlnets), 0)
        self.assertIsNone(manager.get_active_condition_type())

    def test_get_memory_usage(self):
        """get_memory_usage should return memory information."""
        manager = ModelManager(self.config)
        usage = manager.get_memory_usage()

        self.assertIn("total_model_memory_mb", usage)
        self.assertIn("loaded_models", usage)
        self.assertIn("gpu_allocated_mb", usage)
        self.assertEqual(usage["loaded_models"], 0)


class TestModelManagerErrorHandling(unittest.TestCase):
    """Test error handling with user-friendly messages."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = ManagerConfig(
            cache_dir=f"{self.test_dir}/cache",
            lazy_load=True,
            device="cpu",
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_unsupported_condition_type(self):
        """Requesting an unsupported condition type should raise ModelLoadError."""
        manager = ModelManager(self.config)

        with self.assertRaises(ModelLoadError) as ctx:
            manager.get_pipeline("unknown_type")

        self.assertIn("Unsupported condition type", ctx.exception.user_message)
        self.assertIn("unknown_type", ctx.exception.user_message)

    def test_format_user_error_not_found(self):
        """404 errors should produce a user-friendly 'not found' message."""
        manager = ModelManager(self.config)
        msg = manager._format_user_error("depth", "404 Client Error: Not Found")
        self.assertIn("could not be found", msg)
        self.assertIn("internet connection", msg)

    def test_format_user_error_oom(self):
        """OOM errors should suggest memory solutions."""
        manager = ModelManager(self.config)
        msg = manager._format_user_error("depth", "CUDA out of memory")
        self.assertIn("GPU memory", msg)

    def test_format_user_error_network(self):
        """Network errors should suggest checking connection."""
        manager = ModelManager(self.config)
        msg = manager._format_user_error("depth", "Connection timeout")
        self.assertIn("Network error", msg)
        self.assertIn("internet connection", msg)

    def test_format_user_error_permission(self):
        """Permission errors should mention authentication."""
        manager = ModelManager(self.config)
        msg = manager._format_user_error("depth", "Permission denied: invalid token")
        self.assertIn("Permission denied", msg)
        self.assertIn("token", msg)

    def test_format_user_error_incompatible(self):
        """Incompatibility errors should explain the issue."""
        manager = ModelManager(self.config)
        msg = manager._format_user_error("depth", "Model architecture mismatch")
        self.assertIn("not compatible", msg)

    def test_format_user_error_generic(self):
        """Unknown errors should still produce a helpful message."""
        manager = ModelManager(self.config)
        msg = manager._format_user_error("depth", "Something unexpected happened")
        self.assertIn("Failed to load", msg)
        self.assertIn("depth", msg)

    @patch("src.inference.model_loader.ModelLoader.load_and_verify_controlnet")
    def test_load_failure_sets_error_status(self, mock_load):
        """A failed load should set the model status to ERROR."""
        mock_load.return_value = ModelLoadResult(
            success=False,
            error_message="Model not found on Hub",
        )

        manager = ModelManager(self.config)

        with self.assertRaises(ModelLoadError):
            manager.get_pipeline("depth")

        self.assertEqual(manager._models["depth"].status, ModelStatus.ERROR)


class TestModelManagerStatus(unittest.TestCase):
    """Test status reporting methods."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = ManagerConfig(
            cache_dir=f"{self.test_dir}/cache",
            lazy_load=True,
            device="cpu",
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_get_status_all_models(self):
        """get_status should return info for all condition types."""
        manager = ModelManager(self.config)
        status = manager.get_status()

        self.assertIn("depth", status)
        self.assertIn("pose", status)
        self.assertIn("edge", status)

        for ctype, info in status.items():
            self.assertIn("status", info)
            self.assertIn("model_id", info)
            self.assertEqual(info["status"], "not_loaded")

    @patch.object(ModelManager, "_load_pipeline")
    @patch.object(ModelManager, "_load_controlnet")
    def test_status_after_loading(self, mock_load_cn, mock_load_pipe):
        """Status should reflect loaded state after get_pipeline."""
        mock_load_cn.return_value = MagicMock()
        mock_load_pipe.return_value = MagicMock()

        manager = ModelManager(self.config)
        manager.get_pipeline("depth")

        status = manager.get_status()
        self.assertEqual(status["depth"]["status"], "ready")
        self.assertEqual(status["pose"]["status"], "not_loaded")

    def test_is_ready_specific_type(self):
        """is_ready should check specific condition type."""
        manager = ModelManager(self.config)
        manager._models["depth"].status = ModelStatus.READY

        self.assertTrue(manager.is_ready("depth"))
        self.assertFalse(manager.is_ready("pose"))

    def test_is_ready_any(self):
        """is_ready without args should check if any model is ready."""
        manager = ModelManager(self.config)
        self.assertFalse(manager.is_ready())

        manager._models["depth"].status = ModelStatus.READY
        self.assertTrue(manager.is_ready())


class TestModelLoadError(unittest.TestCase):
    """Test ModelLoadError exception class."""

    def test_error_attributes(self):
        """ModelLoadError should carry condition_type and technical_detail."""
        error = ModelLoadError(
            message="User-friendly message",
            condition_type="depth",
            technical_detail="OSError: 404",
        )
        self.assertEqual(error.user_message, "User-friendly message")
        self.assertEqual(error.condition_type, "depth")
        self.assertEqual(error.technical_detail, "OSError: 404")
        self.assertEqual(str(error), "User-friendly message")

    def test_error_is_exception(self):
        """ModelLoadError should be catchable as Exception."""
        with self.assertRaises(Exception):
            raise ModelLoadError(message="test error")


class TestModelManagerConditionTypeValidation(unittest.TestCase):
    """Test condition type validation and normalization."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = ManagerConfig(
            cache_dir=f"{self.test_dir}/cache",
            lazy_load=True,
            device="cpu",
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch.object(ModelManager, "_load_pipeline")
    @patch.object(ModelManager, "_load_controlnet")
    def test_case_insensitive_condition_type(self, mock_load_cn, mock_load_pipe):
        """Condition type should be case-insensitive."""
        mock_load_cn.return_value = MagicMock()
        mock_load_pipe.return_value = MagicMock()

        manager = ModelManager(self.config)
        manager.get_pipeline("DEPTH")

        self.assertTrue(manager.is_ready("depth"))

    @patch.object(ModelManager, "_load_pipeline")
    @patch.object(ModelManager, "_load_controlnet")
    def test_whitespace_trimmed(self, mock_load_cn, mock_load_pipe):
        """Condition type should have whitespace trimmed."""
        mock_load_cn.return_value = MagicMock()
        mock_load_pipe.return_value = MagicMock()

        manager = ModelManager(self.config)
        manager.get_pipeline("  depth  ")

        self.assertTrue(manager.is_ready("depth"))


if __name__ == "__main__":
    unittest.main()
