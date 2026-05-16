"""
Unit tests for model loading and compatibility verification module.

Tests cover:
- ModelLoader initialization and configuration
- Local model loading with various directory structures
- HuggingFace Hub loading (mocked)
- Architecture compatibility verification
- Graceful fallback for missing/incompatible models
- Model caching behavior
- Convenience function load_models_for_inference
"""

import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
import unittest

import torch
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

import importlib.util

spec = importlib.util.spec_from_file_location(
    "model_loader",
    Path(__file__).parent.parent.parent / "src" / "inference" / "model_loader.py",
)
model_loader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_loader_module)

ModelLoader = model_loader_module.ModelLoader
ModelLoadResult = model_loader_module.ModelLoadResult
CompatibilityReport = model_loader_module.CompatibilityReport
load_models_for_inference = model_loader_module.load_models_for_inference
DEFAULT_SD15_MODEL_ID = model_loader_module.DEFAULT_SD15_MODEL_ID
DEFAULT_CONTROLNET_MODELS = model_loader_module.DEFAULT_CONTROLNET_MODELS
EXPECTED_SD15_CONFIG = model_loader_module.EXPECTED_SD15_CONFIG


class TestModelLoadResult(unittest.TestCase):
    """Test ModelLoadResult dataclass."""

    def test_default_result_is_not_loaded(self):
        """A default ModelLoadResult should indicate failure."""
        result = ModelLoadResult()
        self.assertFalse(result.is_loaded)
        self.assertFalse(result.success)
        self.assertIsNone(result.model)

    def test_successful_result(self):
        """A successful result should report is_loaded=True."""
        result = ModelLoadResult(model="fake_model", success=True, source="hub")
        self.assertTrue(result.is_loaded)
        self.assertEqual(result.source, "hub")

    def test_failed_result_with_error(self):
        """A failed result should carry the error message."""
        result = ModelLoadResult(success=False, error_message="Not found")
        self.assertFalse(result.is_loaded)
        self.assertEqual(result.error_message, "Not found")


class TestCompatibilityReport(unittest.TestCase):
    """Test CompatibilityReport dataclass."""

    def test_default_report_is_compatible(self):
        """A fresh report should be compatible by default."""
        report = CompatibilityReport()
        self.assertTrue(report.compatible)
        self.assertEqual(len(report.issues), 0)

    def test_add_issue_makes_incompatible(self):
        """Adding an issue should mark the report as incompatible."""
        report = CompatibilityReport()
        report.add_issue("Cross attention dim mismatch")
        self.assertFalse(report.compatible)
        self.assertEqual(len(report.issues), 1)

    def test_add_warning_stays_compatible(self):
        """Adding a warning should not change compatibility status."""
        report = CompatibilityReport()
        report.add_warning("Layers per block differs")
        self.assertTrue(report.compatible)
        self.assertEqual(len(report.warnings), 1)

    def test_summary_format(self):
        """Summary should include status and issues/warnings."""
        report = CompatibilityReport()
        report.add_issue("Bad dimension")
        report.add_warning("Minor difference")
        summary = report.summary()
        self.assertIn("INCOMPATIBLE", summary)
        self.assertIn("Bad dimension", summary)
        self.assertIn("Minor difference", summary)


class TestModelLoaderInit(unittest.TestCase):
    """Test ModelLoader initialization."""

    def test_default_initialization(self):
        """ModelLoader should initialize with sensible defaults."""
        loader = ModelLoader(cache_dir="/tmp/test_cache_ml")
        self.assertEqual(loader.torch_dtype, torch.float16)
        self.assertTrue(loader.cache_dir.exists())
        # Cleanup
        shutil.rmtree("/tmp/test_cache_ml", ignore_errors=True)

    def test_custom_device(self):
        """ModelLoader should accept a custom device."""
        loader = ModelLoader(cache_dir="/tmp/test_cache_ml2", device="cpu")
        self.assertEqual(loader.device, "cpu")
        shutil.rmtree("/tmp/test_cache_ml2", ignore_errors=True)

    def test_custom_dtype(self):
        """ModelLoader should accept a custom dtype."""
        loader = ModelLoader(
            cache_dir="/tmp/test_cache_ml3", torch_dtype=torch.float32
        )
        self.assertEqual(loader.torch_dtype, torch.float32)
        shutil.rmtree("/tmp/test_cache_ml3", ignore_errors=True)


class TestVerifyCompatibility(unittest.TestCase):
    """Test architecture compatibility verification."""

    def setUp(self):
        self.loader = ModelLoader(cache_dir="/tmp/test_compat_cache", device="cpu")

    def tearDown(self):
        shutil.rmtree("/tmp/test_compat_cache", ignore_errors=True)

    def _make_mock_model(self, config_dict):
        """Create a mock model with a given config dict."""
        model = MagicMock()
        model.config = config_dict
        return model

    def test_compatible_model(self):
        """A model with matching SD1.5 config should pass verification."""
        model = self._make_mock_model({
            "cross_attention_dim": 768,
            "block_out_channels": [320, 640, 1280, 1280],
            "in_channels": 4,
            "layers_per_block": 2,
        })
        report = self.loader.verify_compatibility(model)
        self.assertTrue(report.compatible)
        self.assertEqual(len(report.issues), 0)

    def test_wrong_cross_attention_dim(self):
        """A model with wrong cross_attention_dim should be incompatible."""
        model = self._make_mock_model({
            "cross_attention_dim": 1024,  # SDXL dimension, not SD1.5
            "block_out_channels": [320, 640, 1280, 1280],
            "in_channels": 4,
            "layers_per_block": 2,
        })
        report = self.loader.verify_compatibility(model)
        self.assertFalse(report.compatible)
        self.assertTrue(any("cross_attention_dim" in i.lower() or "cross attention" in i.lower() for i in report.issues))

    def test_wrong_block_channels(self):
        """A model with wrong block_out_channels should be incompatible."""
        model = self._make_mock_model({
            "cross_attention_dim": 768,
            "block_out_channels": [320, 640, 1280],  # Missing last block
            "in_channels": 4,
            "layers_per_block": 2,
        })
        report = self.loader.verify_compatibility(model)
        self.assertFalse(report.compatible)

    def test_wrong_in_channels(self):
        """A model with wrong in_channels should be incompatible."""
        model = self._make_mock_model({
            "cross_attention_dim": 768,
            "block_out_channels": [320, 640, 1280, 1280],
            "in_channels": 3,  # Wrong, should be 4
            "layers_per_block": 2,
        })
        report = self.loader.verify_compatibility(model)
        self.assertFalse(report.compatible)

    def test_different_layers_per_block_is_warning(self):
        """Different layers_per_block should produce a warning, not an issue."""
        model = self._make_mock_model({
            "cross_attention_dim": 768,
            "block_out_channels": [320, 640, 1280, 1280],
            "in_channels": 4,
            "layers_per_block": 3,  # Different but not critical
        })
        report = self.loader.verify_compatibility(model)
        # Should still be compatible (warning only)
        self.assertTrue(report.compatible)
        self.assertTrue(len(report.warnings) > 0)

    def test_no_config_model(self):
        """A model without extractable config should report an issue."""
        model = MagicMock(spec=[])  # No config attribute
        del model.config
        report = self.loader.verify_compatibility(model)
        self.assertFalse(report.compatible)


class TestLoadControlnetLocal(unittest.TestCase):
    """Test local ControlNet loading."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.loader = ModelLoader(cache_dir=f"{self.test_dir}/cache", device="cpu")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch.object(ModelLoader, "_load_controlnet_hub")
    def test_nonexistent_path_falls_through(self, mock_hub):
        """A nonexistent local path should skip local and try hub/fallback."""
        # Make hub loading fail so we can verify the full fallback chain
        mock_hub.return_value = ModelLoadResult(
            success=False, error_message="Hub unavailable"
        )
        result = self.loader.load_controlnet(
            model_id_or_path="/nonexistent/path/model",
            condition_type="depth",
        )
        # Should fail since both hub attempts are mocked to fail
        self.assertFalse(result.success)

    def test_empty_directory_fails(self):
        """A directory without model files should fail gracefully."""
        empty_dir = Path(self.test_dir) / "empty_model"
        empty_dir.mkdir()

        result = self.loader._load_controlnet_local(empty_dir)
        self.assertFalse(result.success)
        self.assertIn("No model weights found", result.error_message)

    @patch("diffusers.ControlNetModel.from_pretrained")
    def test_valid_local_directory_with_config(self, mock_from_pretrained):
        """A valid local directory with config.json should load successfully."""
        model_dir = Path(self.test_dir) / "valid_model"
        model_dir.mkdir()

        # Create a config.json
        config = {
            "cross_attention_dim": 768,
            "block_out_channels": [320, 640, 1280, 1280],
            "in_channels": 4,
        }
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)

        # Mock the model
        mock_model = MagicMock()
        mock_model.config = config
        mock_model.to.return_value = mock_model
        mock_from_pretrained.return_value = mock_model

        result = self.loader._load_controlnet_local(model_dir)
        self.assertTrue(result.success)
        self.assertEqual(result.source, "local")


class TestLoadControlnetHub(unittest.TestCase):
    """Test HuggingFace Hub ControlNet loading."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.loader = ModelLoader(cache_dir=f"{self.test_dir}/cache", device="cpu")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("diffusers.ControlNetModel.from_pretrained")
    def test_successful_hub_load(self, mock_from_pretrained):
        """Successful Hub load should return a valid result."""
        mock_model = MagicMock()
        mock_model.config = {"cross_attention_dim": 768}
        mock_model.to.return_value = mock_model
        mock_from_pretrained.return_value = mock_model

        result = self.loader._load_controlnet_hub("lllyasviel/control_v11f1p_sd15_depth")
        self.assertTrue(result.success)
        self.assertEqual(result.source, "hub")

    @patch("diffusers.ControlNetModel.from_pretrained")
    def test_hub_model_not_found(self, mock_from_pretrained):
        """A 404 error from Hub should produce a clear error message."""
        mock_from_pretrained.side_effect = OSError("404 Client Error: Not Found")

        result = self.loader._load_controlnet_hub("nonexistent/model")
        self.assertFalse(result.success)
        self.assertIn("not found", result.error_message.lower())

    @patch("diffusers.ControlNetModel.from_pretrained")
    def test_hub_network_error(self, mock_from_pretrained):
        """A network error should be reported gracefully."""
        mock_from_pretrained.side_effect = OSError("Connection timeout")

        result = self.loader._load_controlnet_hub("some/model")
        self.assertFalse(result.success)
        self.assertIn("Network error", result.error_message)


class TestModelCaching(unittest.TestCase):
    """Test model caching behavior."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.loader = ModelLoader(cache_dir=f"{self.test_dir}/cache", device="cpu")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("diffusers.ControlNetModel.from_pretrained")
    def test_cached_model_reused(self, mock_from_pretrained):
        """A previously loaded model should be returned from cache."""
        mock_model = MagicMock()
        mock_model.config = {"cross_attention_dim": 768}
        mock_model.to.return_value = mock_model
        mock_from_pretrained.return_value = mock_model

        # First load
        result1 = self.loader._load_controlnet_hub("test/model")
        self.assertTrue(result1.success)

        # Store in cache manually (simulating what load_controlnet does)
        self.loader._loaded_models["controlnet_test/model"] = mock_model

        # Second load should use cache
        result2 = self.loader.load_controlnet("test/model")
        self.assertTrue(result2.success)
        self.assertEqual(result2.source, "cache")

        # from_pretrained should only have been called once
        self.assertEqual(mock_from_pretrained.call_count, 1)

    def test_clear_cache_removes_models(self):
        """clear_cache should remove all cached models."""
        self.loader._loaded_models["key1"] = "model1"
        self.loader._loaded_models["key2"] = "model2"

        self.loader.clear_cache()
        self.assertEqual(len(self.loader._loaded_models), 0)

    def test_clear_cache_specific_key(self):
        """clear_cache with a key should only remove that model."""
        self.loader._loaded_models["key1"] = "model1"
        self.loader._loaded_models["key2"] = "model2"

        self.loader.clear_cache("key1")
        self.assertNotIn("key1", self.loader._loaded_models)
        self.assertIn("key2", self.loader._loaded_models)


class TestLoadAndVerifyControlnet(unittest.TestCase):
    """Test combined load + verify workflow."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.loader = ModelLoader(cache_dir=f"{self.test_dir}/cache", device="cpu")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch.object(ModelLoader, "load_controlnet")
    def test_load_failure_propagates(self, mock_load):
        """If loading fails, the error should propagate."""
        mock_load.return_value = ModelLoadResult(
            success=False, error_message="Not found"
        )

        result = self.loader.load_and_verify_controlnet(
            model_id_or_path="bad/model", condition_type="depth"
        )
        self.assertFalse(result.success)

    @patch.object(ModelLoader, "load_controlnet")
    @patch.object(ModelLoader, "verify_compatibility")
    def test_incompatible_strict_fails(self, mock_verify, mock_load):
        """In strict mode, incompatible model should fail."""
        mock_model = MagicMock()
        mock_load.return_value = ModelLoadResult(
            model=mock_model, success=True, source="hub"
        )

        bad_report = CompatibilityReport()
        bad_report.add_issue("Wrong dimensions")
        mock_verify.return_value = bad_report

        result = self.loader.load_and_verify_controlnet(
            model_id_or_path="some/model",
            condition_type="depth",
            strict=True,
        )
        self.assertFalse(result.success)
        self.assertIn("incompatible", result.error_message.lower())

    @patch.object(ModelLoader, "load_controlnet")
    @patch.object(ModelLoader, "verify_compatibility")
    def test_incompatible_nonstrict_warns(self, mock_verify, mock_load):
        """In non-strict mode, incompatible model should load with warnings."""
        mock_model = MagicMock()
        mock_load.return_value = ModelLoadResult(
            model=mock_model, success=True, source="hub"
        )

        bad_report = CompatibilityReport()
        bad_report.add_issue("Wrong dimensions")
        mock_verify.return_value = bad_report

        result = self.loader.load_and_verify_controlnet(
            model_id_or_path="some/model",
            condition_type="depth",
            strict=False,
        )
        # Should still succeed in non-strict mode
        self.assertTrue(result.success)
        self.assertTrue(len(result.warnings) > 0)


class TestGetModelInfo(unittest.TestCase):
    """Test get_model_info method."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.loader = ModelLoader(cache_dir=f"{self.test_dir}/cache", device="cpu")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_local_model_with_config(self):
        """get_model_info should read config from local model directory."""
        model_dir = Path(self.test_dir) / "my_model"
        model_dir.mkdir()

        config = {"cross_attention_dim": 768, "condition_type": "depth"}
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)

        info = self.loader.get_model_info(str(model_dir))
        self.assertTrue(info["available"])
        self.assertEqual(info["source"], "local")
        self.assertEqual(info["config"]["cross_attention_dim"], 768)

    def test_local_model_with_metadata(self):
        """get_model_info should read metadata from local model directory."""
        model_dir = Path(self.test_dir) / "my_model2"
        model_dir.mkdir()

        metadata = {"model_name": "controlnet_depth", "model_version": "1.0.0"}
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        info = self.loader.get_model_info(str(model_dir))
        self.assertTrue(info["available"])
        self.assertEqual(info["metadata"]["model_name"], "controlnet_depth")

    def test_nonexistent_path_not_available(self):
        """get_model_info for nonexistent path should report not available."""
        info = self.loader.get_model_info("/nonexistent/path")
        self.assertFalse(info["available"])


class TestDefaultControlnetModels(unittest.TestCase):
    """Test default model ID mappings."""

    def test_all_condition_types_have_defaults(self):
        """All supported condition types should have default model IDs."""
        for ctype in ["depth", "pose", "edge"]:
            self.assertIn(ctype, DEFAULT_CONTROLNET_MODELS)
            self.assertTrue(len(DEFAULT_CONTROLNET_MODELS[ctype]) > 0)

    def test_expected_sd15_config_values(self):
        """Expected SD1.5 config should have correct values."""
        self.assertEqual(EXPECTED_SD15_CONFIG["cross_attention_dim"], 768)
        self.assertEqual(EXPECTED_SD15_CONFIG["in_channels"], 4)
        self.assertEqual(EXPECTED_SD15_CONFIG["block_out_channels"], [320, 640, 1280, 1280])


class TestFallbackBehavior(unittest.TestCase):
    """Test graceful fallback when models are missing or incompatible."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.loader = ModelLoader(cache_dir=f"{self.test_dir}/cache", device="cpu")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_none_model_id_uses_default(self):
        """Passing None as model_id should use the default for condition_type."""
        with patch.object(self.loader, "_load_controlnet_hub") as mock_hub:
            mock_model = MagicMock()
            mock_model.config = {}
            mock_model.to.return_value = mock_model
            mock_hub.return_value = ModelLoadResult(
                model=mock_model, success=True, source="hub"
            )

            result = self.loader.load_controlnet(
                model_id_or_path=None, condition_type="depth"
            )
            # Should have tried the default depth model
            mock_hub.assert_called()
            call_args = mock_hub.call_args[0][0]
            self.assertEqual(call_args, DEFAULT_CONTROLNET_MODELS["depth"])

    def test_unknown_condition_type_no_default(self):
        """An unknown condition type with no default should fail gracefully."""
        result = self.loader.load_controlnet(
            model_id_or_path=None, condition_type="unknown_type"
        )
        self.assertFalse(result.success)
        self.assertIn("No default ControlNet model", result.error_message)


if __name__ == "__main__":
    unittest.main()
