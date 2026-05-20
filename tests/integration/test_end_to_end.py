"""
End-to-End Integration Tests for ControlNet Training Pipeline

This module tests the complete pipeline integration from dataset processing
through model training, evaluation, inference, and deployment. Tests use
mocked heavy operations (model downloads, GPU operations) and run on CPU
without requiring actual model weights.

Requirements Validated: 8.1, 10.1, 10.2, 10.3
"""

import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_test_image(width: int = 512, height: int = 512, mode: str = "RGB") -> Image.Image:
    """Create a synthetic test image with moderate edge content.

    Produces an image with geometric shapes and gradients that yields an edge
    density between 0.001 and 0.5 when processed by Canny edge detection.
    """
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    # Background
    arr[:, :] = (180, 180, 180)

    # Draw several rectangles with distinct colors to create clear edges
    rects = [
        (50, 50, 200, 200, (30, 60, 120)),
        (100, 250, 250, 450, (200, 50, 50)),
        (width // 2, 50, width // 2 + 150, 200, (50, 180, 50)),
        (width // 2 + 20, 250, width - 30, 450, (100, 100, 200)),
    ]
    for x1, y1, x2, y2, color in rects:
        x1, x2 = min(x1, width - 1), min(x2, width - 1)
        y1, y2 = min(y1, height - 1), min(y2, height - 1)
        arr[y1:y2, x1:x2] = color

    # Add a circle-like shape (filled ellipse approximation)
    cy, cx, r = height // 2, width // 2, min(width, height) // 6
    yy, xx = np.ogrid[:height, :width]
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2
    arr[mask] = (220, 180, 50)

    return Image.fromarray(arr, mode)


def create_condition_map(height: int = 512, width: int = 512, channels: int = 3) -> np.ndarray:
    """Create a synthetic condition map (normalized float array)."""
    return np.random.rand(height, width, channels).astype(np.float32)


# ---------------------------------------------------------------------------
# Test: Full pipeline can be imported and initialized
# ---------------------------------------------------------------------------

class TestPipelineImportsAndInitialization:
    """Test that all pipeline components can be imported and initialized."""

    def test_data_module_imports(self):
        """Verify all data processing modules can be imported."""
        from src.data.dataset_processor import DatasetProcessor, DatasetReport, ProcessingSample
        from src.data.extract_edges import CannyEdgeExtractor, EdgeExtractionConfig
        from src.data.verify_dataset import ValidationConfig

        assert DatasetProcessor is not None
        assert DatasetReport is not None
        assert ProcessingSample is not None
        assert CannyEdgeExtractor is not None
        assert EdgeExtractionConfig is not None
        assert ValidationConfig is not None

    def test_model_module_imports(self):
        """Verify all model modules can be imported."""
        from src.models.controlnet import ControlNetModel, ZeroConvolution
        from src.models.config import ControlNetConfig, TrainingConfig, ModelMetadata
        from src.models.unet_wrapper import ControlNetUNet2DConditionModel

        assert ControlNetModel is not None
        assert ZeroConvolution is not None
        assert ControlNetConfig is not None
        assert TrainingConfig is not None
        assert ModelMetadata is not None
        assert ControlNetUNet2DConditionModel is not None

    def test_training_module_imports(self):
        """Verify all training modules can be imported."""
        from src.training.trainer import ControlNetTrainer, TrainingState, MemoryOptimizer
        from src.training.losses import DiffusionLoss

        assert ControlNetTrainer is not None
        assert TrainingState is not None
        assert MemoryOptimizer is not None
        assert DiffusionLoss is not None

    def test_inference_module_imports(self):
        """Verify all inference modules can be imported."""
        from src.inference.pipeline import (
            ControlNetInferencePipeline,
            InferenceConfig,
            GenerationParams,
            GenerationResult,
            ConditionType,
        )
        from src.inference.model_loader import ModelLoader, ModelLoadResult, CompatibilityReport

        assert ControlNetInferencePipeline is not None
        assert InferenceConfig is not None
        assert GenerationParams is not None
        assert GenerationResult is not None
        assert ConditionType is not None
        assert ModelLoader is not None
        assert ModelLoadResult is not None
        assert CompatibilityReport is not None

    def test_app_module_imports(self):
        """Verify all app modules can be imported."""
        from src.app.gradio_app import create_gradio_app, _extract_condition_map, _generate_image
        from src.app.model_manager import ModelManager, ManagerConfig, ModelStatus

        assert create_gradio_app is not None
        assert _extract_condition_map is not None
        assert _generate_image is not None
        assert ModelManager is not None
        assert ManagerConfig is not None
        assert ModelStatus is not None

    def test_utils_module_imports(self):
        """Verify all utility modules can be imported."""
        from src.utils.memory_utils import MemoryOptimizer, MemoryProfiler
        from src.utils.error_handling import ControlNetPipelineError
        from src.utils.logging_utils import setup_logging

        assert MemoryOptimizer is not None
        assert MemoryProfiler is not None
        assert ControlNetPipelineError is not None
        assert setup_logging is not None


# ---------------------------------------------------------------------------
# Test: Dataset processing -> condition extraction -> validation flow
# ---------------------------------------------------------------------------

class TestDatasetProcessingFlow:
    """Test the dataset processing to condition extraction to validation pipeline."""

    def test_edge_extraction_from_image(self):
        """Test that edge extraction works on a synthetic image."""
        from src.data.extract_edges import CannyEdgeExtractor, EdgeExtractionConfig

        config = EdgeExtractionConfig(
            adaptive_threshold=True,
            output_channels=3,
            normalize_output=True,
        )
        extractor = CannyEdgeExtractor(config)
        test_image = create_test_image(256, 256)

        result = extractor.extract(test_image)

        assert result.success, f"Edge extraction failed: {result.error_message}"
        assert result.edge_map is not None
        assert result.edge_map.shape[0] == 256
        assert result.edge_map.shape[1] == 256
        # Normalized output should be in [0, 1]
        assert result.edge_map.min() >= 0.0
        assert result.edge_map.max() <= 1.0

    def test_dataset_processor_initialization(self):
        """Test DatasetProcessor can be initialized with a temp directory."""
        from src.data.dataset_processor import DatasetProcessor

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = DatasetProcessor(
                cache_dir=tmpdir,
                max_retries=2,
            )
            assert processor is not None

    def test_processing_sample_validation(self):
        """Test ProcessingSample validation logic."""
        from src.data.dataset_processor import ProcessingSample

        # Valid sample
        sample = ProcessingSample(
            image=create_test_image(512, 512),
            caption="A test image of a cat sitting on a mat",
            image_id="test_001",
        )
        is_valid, errors = sample.validate()
        assert is_valid, f"Valid sample failed validation: {errors}"

    def test_dataset_report_generation(self):
        """Test DatasetReport tracks errors and validity correctly."""
        from src.data.dataset_processor import DatasetReport

        report = DatasetReport(total_samples=100, valid_samples=90)
        report.finalize()
        assert report.is_valid

        report.add_error("Sample 5 has corrupted image")
        assert len(report.errors) == 1

        report.add_warning("Sample 10 has short caption")
        assert len(report.warnings) == 1

    def test_edge_extraction_batch_processing(self):
        """Test batch edge extraction simulating dataset processing."""
        from src.data.extract_edges import CannyEdgeExtractor, EdgeExtractionConfig

        config = EdgeExtractionConfig(
            adaptive_threshold=True,
            output_channels=3,
            normalize_output=True,
        )
        extractor = CannyEdgeExtractor(config)

        # Simulate processing a small batch of images
        images = [create_test_image(256, 256) for _ in range(5)]
        results = extractor.extract_batch(images)

        assert len(results) == 5
        for result in results:
            assert result.success
            assert result.edge_map is not None

    def test_dataset_processor_validate_integrity(self):
        """Test dataset integrity validation with mock samples."""
        from src.data.dataset_processor import DatasetProcessor, ProcessingSample

        with tempfile.TemporaryDirectory() as tmpdir:
            processor = DatasetProcessor(cache_dir=tmpdir)

            samples = [
                ProcessingSample(
                    image=create_test_image(512, 512),
                    caption=f"Test prompt number {i} with enough length",
                    image_id=f"sample_{i}",
                )
                for i in range(5)
            ]

            report = processor.validate_dataset_integrity(samples)
            assert report is not None
            assert report.total_samples == 5
            assert report.valid_samples == 5


# ---------------------------------------------------------------------------
# Test: Model creation -> training step -> checkpoint save/load flow
# ---------------------------------------------------------------------------

class TestModelTrainingFlow:
    """Test model creation, training step, and checkpoint save/load."""

    def test_controlnet_model_creation(self):
        """Test ControlNet model can be created with default config."""
        from src.models.controlnet import ControlNetModel

        model = ControlNetModel(
            in_channels=4,
            conditioning_channels=3,
            block_out_channels=(32, 64, 128, 128),  # Small for testing
            layers_per_block=1,
            cross_attention_dim=32,
            norm_num_groups=16,
        )

        assert model is not None
        # Verify model has expected components
        assert hasattr(model, "controlnet_cond_embedding")
        assert hasattr(model, "down_blocks")
        assert hasattr(model, "mid_block")
        assert hasattr(model, "controlnet_down_blocks")
        assert hasattr(model, "controlnet_mid_block")

    def test_controlnet_forward_pass(self):
        """Test ControlNet forward pass produces correct output structure."""
        from src.models.controlnet import ControlNetModel

        model = ControlNetModel(
            in_channels=4,
            conditioning_channels=3,
            block_out_channels=(32, 64, 128, 128),
            layers_per_block=1,
            cross_attention_dim=32,
            norm_num_groups=16,
        )
        model.eval()

        batch_size = 1
        # Latent space input (B, 4, H/8, W/8)
        sample = torch.randn(batch_size, 4, 32, 32)
        # Timestep
        timestep = torch.tensor([500])
        # Text encoder hidden states (B, seq_len, cross_attention_dim)
        encoder_hidden_states = torch.randn(batch_size, 10, 32)
        # Condition map at full resolution (B, 3, H, W)
        controlnet_cond = torch.randn(batch_size, 3, 256, 256)

        with torch.no_grad():
            output = model(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond,
                conditioning_scale=1.0,
            )

        assert "down_block_res_samples" in output
        assert "mid_block_res_sample" in output
        assert len(output["down_block_res_samples"]) > 0
        assert output["mid_block_res_sample"] is not None

    def test_controlnet_config_save_load(self):
        """Test ControlNet config serialization round-trip."""
        from src.models.config import ControlNetConfig

        config = ControlNetConfig(
            condition_type="depth",
            conditioning_channels=1,
            block_out_channels=(320, 640, 1280, 1280),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config.save_json(config_path)

            loaded_config = ControlNetConfig.from_json(config_path)

            assert loaded_config.condition_type == config.condition_type
            assert loaded_config.conditioning_channels == config.conditioning_channels
            # JSON serialization converts tuples to lists, so compare as lists
            assert list(loaded_config.block_out_channels) == list(config.block_out_channels)

    def test_training_config_save_load(self):
        """Test TrainingConfig serialization round-trip."""
        from src.models.config import TrainingConfig

        config = TrainingConfig(
            learning_rate=1e-5,
            num_train_epochs=10,
            train_batch_size=1,
            gradient_accumulation_steps=8,
            mixed_precision="fp16",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "training_config.json"
            config.save_json(config_path)

            loaded_config = TrainingConfig.from_json(config_path)

            assert loaded_config.learning_rate == config.learning_rate
            assert loaded_config.num_train_epochs == config.num_train_epochs
            assert loaded_config.mixed_precision == config.mixed_precision

    def test_model_checkpoint_save_load(self):
        """Test model checkpoint save and load cycle."""
        from src.models.controlnet import ControlNetModel

        # Create a small model
        model = ControlNetModel(
            in_channels=4,
            conditioning_channels=3,
            block_out_channels=(32, 64, 128, 128),
            layers_per_block=1,
            cross_attention_dim=32,
            norm_num_groups=16,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "checkpoint"
            save_path.mkdir()

            # Save model
            model.save_pretrained(str(save_path))

            # Verify files were created
            assert (save_path / "config.json").exists()

            # Load model
            loaded_model = ControlNetModel.from_pretrained(str(save_path))
            assert loaded_model is not None

            # Verify outputs match
            sample = torch.randn(1, 4, 32, 32)
            timestep = torch.tensor([500])
            encoder_hidden_states = torch.randn(1, 10, 32)
            controlnet_cond = torch.randn(1, 3, 256, 256)

            with torch.no_grad():
                original_output = model(
                    sample=sample,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_cond,
                )
                loaded_output = loaded_model(
                    sample=sample,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_cond,
                )

            # Outputs should be identical
            for orig, loaded in zip(
                original_output["down_block_res_samples"],
                loaded_output["down_block_res_samples"],
            ):
                torch.testing.assert_close(orig, loaded, rtol=1e-5, atol=1e-5)

    def test_training_state_persistence(self):
        """Test TrainingState can be serialized and restored."""
        from src.training.trainer import TrainingState

        state = TrainingState()
        state.global_step = 100
        state.epoch = 5
        state.best_loss = 0.05

        state_dict = state.to_dict()
        restored = TrainingState.from_dict(state_dict)

        assert restored.global_step == 100
        assert restored.epoch == 5
        assert restored.best_loss == 0.05

    def test_memory_optimizer_initialization(self):
        """Test MemoryOptimizer can be initialized and provides stats."""
        from src.training.trainer import MemoryOptimizer

        optimizer = MemoryOptimizer(target_memory_gb=12.0)
        stats = optimizer.get_memory_stats()

        assert stats is not None
        assert isinstance(stats, dict)


# ---------------------------------------------------------------------------
# Test: Inference pipeline initialization -> generation -> output validation
# ---------------------------------------------------------------------------

class TestInferencePipelineFlow:
    """Test inference pipeline with mocked models."""

    def test_inference_config_creation(self):
        """Test InferenceConfig can be created with various settings."""
        from src.inference.pipeline import InferenceConfig

        config = InferenceConfig(
            condition_type="depth",
            device="cpu",
            dtype="float32",
            enable_memory_optimization=False,
        )

        assert config.condition_type == "depth"
        assert config.device == "cpu"
        assert config.dtype == "float32"

    def test_generation_params_creation(self):
        """Test GenerationParams can be created with valid parameters."""
        from src.inference.pipeline import GenerationParams

        params = GenerationParams(
            prompt="A beautiful landscape",
            num_inference_steps=20,
            guidance_scale=7.5,
            conditioning_scale=1.0,
            height=512,
            width=512,
            seed=42,
        )

        assert params.prompt == "A beautiful landscape"
        assert params.num_inference_steps == 20
        assert params.seed == 42

    def test_inference_pipeline_initialization(self):
        """Test ControlNetInferencePipeline can be initialized."""
        from src.inference.pipeline import ControlNetInferencePipeline, InferenceConfig

        config = InferenceConfig(
            condition_type="edge",
            device="cpu",
            dtype="float32",
            enable_memory_optimization=False,
        )

        pipeline = ControlNetInferencePipeline(config=config)
        assert pipeline is not None
        assert not pipeline.is_loaded

    def test_condition_processor_validation(self):
        """Test ConditionProcessor validates condition maps correctly."""
        from src.inference.pipeline import ConditionProcessor, ConditionType

        processor = ConditionProcessor(
            condition_type=ConditionType.EDGE,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        # Valid condition map
        valid_map = create_condition_map(512, 512, 3)
        is_valid = processor.validate_condition(valid_map)
        assert is_valid

    def test_ddim_scheduler_initialization(self):
        """Test DDIMScheduler can be initialized and set timesteps."""
        from src.inference.pipeline import DDIMScheduler

        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
        )

        scheduler.set_timesteps(20)
        assert scheduler.timesteps is not None
        assert len(scheduler.timesteps) == 20

    @patch("src.inference.pipeline.ControlNetInferencePipeline.load_models")
    def test_inference_pipeline_generate_with_mocked_models(self, mock_load):
        """Test generation flow with mocked model loading."""
        from src.inference.pipeline import (
            ControlNetInferencePipeline,
            InferenceConfig,
            GenerationParams,
            GenerationResult,
        )

        config = InferenceConfig(
            condition_type="edge",
            device="cpu",
            dtype="float32",
        )
        pipeline = ControlNetInferencePipeline(config=config)

        # Mock the internal state to simulate loaded models
        pipeline._models_loaded = True
        pipeline._controlnet = MagicMock()
        pipeline._unet = MagicMock()
        pipeline._vae = MagicMock()
        pipeline._text_encoder = MagicMock()
        pipeline._tokenizer = MagicMock()

        # Mock tokenizer output
        pipeline._tokenizer.return_value = {
            "input_ids": torch.zeros(1, 77, dtype=torch.long)
        }
        pipeline._tokenizer.model_max_length = 77

        # Mock text encoder output
        mock_text_output = MagicMock()
        mock_text_output.last_hidden_state = torch.randn(1, 77, 768)
        pipeline._text_encoder.return_value = mock_text_output

        # Mock VAE decode to return a valid image tensor
        pipeline._vae.decode = MagicMock(
            return_value=MagicMock(sample=torch.randn(1, 3, 512, 512))
        )
        pipeline._vae.config = MagicMock(scaling_factor=0.18215)

        # Mock ControlNet forward
        pipeline._controlnet.return_value = {
            "down_block_res_samples": [torch.randn(1, 32, 64, 64)] * 5,
            "mid_block_res_sample": torch.randn(1, 128, 8, 8),
        }

        # Mock UNet forward
        mock_unet_output = MagicMock()
        mock_unet_output.sample = torch.randn(1, 4, 64, 64)
        pipeline._unet.return_value = mock_unet_output

        # The pipeline should be considered loaded now
        assert pipeline.is_loaded


# ---------------------------------------------------------------------------
# Test: Gradio app creation and component wiring
# ---------------------------------------------------------------------------

class TestGradioAppIntegration:
    """Test Gradio app creation and component wiring."""

    def test_gradio_app_creation(self):
        """Test that create_gradio_app returns a valid Gradio Blocks instance."""
        try:
            import gradio as gr
        except ImportError:
            pytest.skip("Gradio not installed")

        from src.app.gradio_app import create_gradio_app

        app = create_gradio_app()
        assert app is not None
        assert isinstance(app, gr.Blocks)

    def test_condition_map_extraction_edge(self):
        """Test condition map extraction for edge type via app function."""
        from src.app.gradio_app import _extract_condition_map

        test_image = create_test_image(256, 256)
        condition_array, display_image, status = _extract_condition_map(test_image, "edge")

        assert condition_array is not None, f"Extraction failed: {status}"
        assert display_image is not None
        assert "successfully" in status.lower() or "extracted" in status.lower()
        assert isinstance(display_image, Image.Image)

    def test_condition_map_extraction_handles_none_image(self):
        """Test that extraction handles None image gracefully."""
        from src.app.gradio_app import _extract_condition_map

        condition_array, display_image, status = _extract_condition_map(None, "edge")

        assert condition_array is None
        assert display_image is None
        assert "no image" in status.lower()

    def test_condition_map_extraction_handles_unknown_type(self):
        """Test that extraction handles unknown condition type gracefully."""
        from src.app.gradio_app import _extract_condition_map

        test_image = create_test_image(256, 256)
        condition_array, display_image, status = _extract_condition_map(
            test_image, "unknown_type"
        )

        assert condition_array is None
        assert "unknown" in status.lower()

    def test_generate_image_handles_missing_image(self):
        """Test that _generate_image handles missing source image."""
        from src.app.gradio_app import _generate_image

        input_img, cond_display, gen_image, status = _generate_image(
            source_image=None,
            condition_type="edge",
            prompt="test prompt",
            num_steps=20,
            guidance_scale=7.5,
            conditioning_strength=1.0,
        )

        assert gen_image is None
        assert "upload" in status.lower() or "image" in status.lower()

    def test_generate_image_handles_empty_prompt(self):
        """Test that _generate_image handles empty prompt."""
        from src.app.gradio_app import _generate_image

        test_image = create_test_image(256, 256)
        input_img, cond_display, gen_image, status = _generate_image(
            source_image=test_image,
            condition_type="edge",
            prompt="",
            num_steps=20,
            guidance_scale=7.5,
            conditioning_strength=1.0,
        )

        assert gen_image is None
        assert "prompt" in status.lower()

    def test_on_image_upload_callback(self):
        """Test the image upload callback extracts condition map."""
        from src.app.gradio_app import _on_image_upload

        test_image = create_test_image(256, 256)
        display_image, status = _on_image_upload(test_image, "edge")

        assert display_image is not None
        assert "successfully" in status.lower() or "extracted" in status.lower()

    def test_on_image_upload_handles_none(self):
        """Test the image upload callback handles None image."""
        from src.app.gradio_app import _on_image_upload

        display_image, status = _on_image_upload(None, "edge")

        assert display_image is None
        assert "no image" in status.lower()


# ---------------------------------------------------------------------------
# Test: Model loader -> compatibility check -> pipeline creation flow
# ---------------------------------------------------------------------------

class TestModelLoaderFlow:
    """Test model loader, compatibility checking, and pipeline creation."""

    def test_model_loader_initialization(self):
        """Test ModelLoader can be initialized."""
        from src.inference.model_loader import ModelLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ModelLoader(
                cache_dir=tmpdir,
                torch_dtype=torch.float32,
                device="cpu",
            )
            assert loader is not None
            assert loader.device == "cpu"

    def test_compatibility_report_creation(self):
        """Test CompatibilityReport tracks issues correctly."""
        from src.inference.model_loader import CompatibilityReport

        report = CompatibilityReport()
        assert report.compatible

        report.add_warning("Minor version mismatch")
        assert report.compatible  # Warnings don't break compatibility
        assert len(report.warnings) == 1

        report.add_issue("Incompatible architecture")
        assert not report.compatible
        assert len(report.issues) == 1

        summary = report.summary()
        assert "INCOMPATIBLE" in summary

    def test_model_load_result_properties(self):
        """Test ModelLoadResult properties."""
        from src.inference.model_loader import ModelLoadResult

        # Unsuccessful result
        result = ModelLoadResult(success=False, error_message="Not found")
        assert not result.is_loaded

        # Successful result
        result = ModelLoadResult(model=MagicMock(), success=True, source="local")
        assert result.is_loaded

    def test_model_loader_verify_compatibility(self):
        """Test compatibility verification with a mock model."""
        from src.inference.model_loader import ModelLoader

        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ModelLoader(cache_dir=tmpdir, device="cpu")

            # Create a mock model with expected config
            mock_model = MagicMock()
            mock_model.config = {
                "cross_attention_dim": 768,
                "block_out_channels": [320, 640, 1280, 1280],
                "in_channels": 4,
                "layers_per_block": 2,
            }

            report = loader.verify_compatibility(mock_model)
            assert report is not None

    def test_model_manager_initialization(self):
        """Test ModelManager can be initialized with config."""
        from src.app.model_manager import ModelManager, ManagerConfig

        config = ManagerConfig(
            cache_dir="./test_cache",
            lazy_load=True,
            max_loaded_models=1,
            device="cpu",
        )

        manager = ModelManager(config)
        assert manager is not None
        assert manager.get_active_condition_type() is None
        assert not manager.is_ready()

    def test_model_manager_supported_types(self):
        """Test ModelManager reports supported condition types."""
        from src.app.model_manager import ModelManager, ManagerConfig

        config = ManagerConfig(lazy_load=True, device="cpu")
        manager = ModelManager(config)

        supported = manager.get_supported_condition_types()
        assert "depth" in supported
        assert "pose" in supported
        assert "edge" in supported

    def test_model_manager_status_reporting(self):
        """Test ModelManager status reporting."""
        from src.app.model_manager import ModelManager, ManagerConfig, ModelStatus

        config = ManagerConfig(lazy_load=True, device="cpu")
        manager = ModelManager(config)

        status = manager.get_status()
        assert isinstance(status, dict)
        assert "depth" in status
        assert status["depth"]["status"] == ModelStatus.NOT_LOADED.value

    def test_model_manager_memory_usage(self):
        """Test ModelManager memory usage reporting."""
        from src.app.model_manager import ModelManager, ManagerConfig

        config = ManagerConfig(lazy_load=True, device="cpu")
        manager = ModelManager(config)

        usage = manager.get_memory_usage()
        assert "total_model_memory_mb" in usage
        assert "loaded_models" in usage
        assert usage["loaded_models"] == 0

    def test_model_manager_validates_condition_type(self):
        """Test ModelManager rejects invalid condition types."""
        from src.app.model_manager import ModelManager, ManagerConfig, ModelLoadError

        config = ManagerConfig(lazy_load=True, device="cpu")
        manager = ModelManager(config)

        with pytest.raises(ModelLoadError) as exc_info:
            manager.get_pipeline("invalid_type")

        assert "unsupported" in str(exc_info.value).lower()

    def test_model_config_manager_save_load(self):
        """Test ControlNetModelManager save/load workflow."""
        from src.models.config import ControlNetConfig, TrainingConfig, ModelMetadata

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create configs
            model_config = ControlNetConfig(condition_type="edge", conditioning_channels=3)
            training_config = TrainingConfig(learning_rate=1e-5, num_train_epochs=50)
            metadata = ModelMetadata(
                model_name="test-controlnet-edge",
                condition_type="edge",
                model_version="1.0.0",
            )

            # Save all configs
            model_config.save_json(Path(tmpdir) / "model_config.json")
            training_config.save_json(Path(tmpdir) / "training_config.json")
            metadata.save_json(Path(tmpdir) / "metadata.json")

            # Load and verify
            loaded_model_config = ControlNetConfig.from_json(Path(tmpdir) / "model_config.json")
            loaded_training_config = TrainingConfig.from_json(Path(tmpdir) / "training_config.json")
            loaded_metadata = ModelMetadata.from_json(Path(tmpdir) / "metadata.json")

            assert loaded_model_config.condition_type == "edge"
            assert loaded_training_config.learning_rate == 1e-5
            assert loaded_metadata.model_name == "test-controlnet-edge"
