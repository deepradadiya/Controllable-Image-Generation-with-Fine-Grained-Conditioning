"""
Unit tests for evaluation/pipeline_loader.py

Tests the checkpoint loading utilities including:
- load_controlnet_pipeline: loads pipeline with trained adapter
- load_baseline_pipeline: loads vanilla SD1.5 without ControlNet
- validate_checkpoints: checks which condition types have checkpoints
"""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from evaluation.pipeline_loader import (
    SD15_MODEL_ID,
    load_baseline_pipeline,
    load_controlnet_pipeline,
    validate_checkpoints,
)


class TestValidateCheckpoints:
    """Tests for validate_checkpoints function."""

    def test_returns_empty_list_when_no_checkpoints_exist(self, tmp_path):
        """Should return empty list when checkpoint directory has no valid checkpoints."""
        result = validate_checkpoints(
            checkpoint_dir=str(tmp_path),
            condition_types=["depth", "pose", "edge"],
        )
        assert result == []

    def test_returns_empty_list_when_dir_does_not_exist(self, tmp_path):
        """Should return empty list when checkpoint directory doesn't exist."""
        nonexistent = str(tmp_path / "nonexistent")
        result = validate_checkpoints(
            checkpoint_dir=nonexistent,
            condition_types=["depth", "pose", "edge"],
        )
        assert result == []

    def test_finds_checkpoint_with_state_dict(self, tmp_path):
        """Should find checkpoint when controlnet_state_dict.pt exists."""
        checkpoint_path = tmp_path / "controlnet-sd15-depth"
        checkpoint_path.mkdir()
        (checkpoint_path / "controlnet_state_dict.pt").touch()

        result = validate_checkpoints(
            checkpoint_dir=str(tmp_path),
            condition_types=["depth"],
        )
        assert result == ["depth"]

    def test_finds_checkpoint_with_pytorch_model(self, tmp_path):
        """Should find checkpoint when pytorch_model.bin exists."""
        checkpoint_path = tmp_path / "controlnet-sd15-pose"
        checkpoint_path.mkdir()
        (checkpoint_path / "pytorch_model.bin").touch()

        result = validate_checkpoints(
            checkpoint_dir=str(tmp_path),
            condition_types=["pose"],
        )
        assert result == ["pose"]

    def test_finds_checkpoint_with_safetensors(self, tmp_path):
        """Should find checkpoint when model.safetensors exists."""
        checkpoint_path = tmp_path / "controlnet-sd15-edge"
        checkpoint_path.mkdir()
        (checkpoint_path / "model.safetensors").touch()

        result = validate_checkpoints(
            checkpoint_dir=str(tmp_path),
            condition_types=["edge"],
        )
        assert result == ["edge"]

    def test_returns_only_valid_subset(self, tmp_path):
        """Should return only condition types that have valid checkpoints."""
        # Create valid checkpoint for depth only
        depth_path = tmp_path / "controlnet-sd15-depth"
        depth_path.mkdir()
        (depth_path / "controlnet_state_dict.pt").touch()

        # Create directory for pose but no model files
        pose_path = tmp_path / "controlnet-sd15-pose"
        pose_path.mkdir()

        result = validate_checkpoints(
            checkpoint_dir=str(tmp_path),
            condition_types=["depth", "pose", "edge"],
        )
        assert result == ["depth"]

    def test_finds_multiple_valid_checkpoints(self, tmp_path):
        """Should find all condition types with valid checkpoints."""
        for ctype in ["depth", "pose", "edge"]:
            checkpoint_path = tmp_path / f"controlnet-sd15-{ctype}"
            checkpoint_path.mkdir()
            (checkpoint_path / "controlnet_state_dict.pt").touch()

        result = validate_checkpoints(
            checkpoint_dir=str(tmp_path),
            condition_types=["depth", "pose", "edge"],
        )
        assert result == ["depth", "pose", "edge"]

    def test_uses_default_condition_types(self, tmp_path):
        """Should default to all three condition types when not specified."""
        depth_path = tmp_path / "controlnet-sd15-depth"
        depth_path.mkdir()
        (depth_path / "controlnet_state_dict.pt").touch()

        result = validate_checkpoints(checkpoint_dir=str(tmp_path))
        assert result == ["depth"]

    def test_logs_warning_for_missing_checkpoints(self, tmp_path, caplog):
        """Should log warnings for condition types without checkpoints."""
        with caplog.at_level(logging.WARNING):
            validate_checkpoints(
                checkpoint_dir=str(tmp_path),
                condition_types=["depth"],
            )

        assert "No checkpoint found for 'depth'" in caplog.text

    def test_logs_warning_for_empty_checkpoint_dir(self, tmp_path, caplog):
        """Should log warning when directory exists but has no model weights."""
        checkpoint_path = tmp_path / "controlnet-sd15-depth"
        checkpoint_path.mkdir()
        # Directory exists but no model files

        with caplog.at_level(logging.WARNING):
            validate_checkpoints(
                checkpoint_dir=str(tmp_path),
                condition_types=["depth"],
            )

        assert "no valid model weights" in caplog.text


class TestLoadControlnetPipeline:
    """Tests for load_controlnet_pipeline function."""

    def test_returns_none_when_checkpoint_not_found(self, tmp_path):
        """Should return None when checkpoint directory doesn't exist."""
        result = load_controlnet_pipeline(
            condition_type="depth",
            checkpoint_dir=str(tmp_path),
            device=torch.device("cpu"),
        )
        assert result is None

    def test_logs_warning_when_checkpoint_not_found(self, tmp_path, caplog):
        """Should log a warning when checkpoint is not found."""
        with caplog.at_level(logging.WARNING):
            load_controlnet_pipeline(
                condition_type="depth",
                checkpoint_dir=str(tmp_path),
                device=torch.device("cpu"),
            )

        assert "Checkpoint not found for condition type 'depth'" in caplog.text

    def test_returns_none_for_invalid_condition_type(self, tmp_path):
        """Should return None for a condition type with no checkpoint."""
        result = load_controlnet_pipeline(
            condition_type="invalid_type",
            checkpoint_dir=str(tmp_path),
            device=torch.device("cpu"),
        )
        assert result is None

    def test_returns_none_when_no_model_weights_in_dir(self, tmp_path):
        """Should return None when checkpoint dir exists but has no weights."""
        checkpoint_path = tmp_path / "controlnet-sd15-depth"
        checkpoint_path.mkdir()
        # No model files inside

        result = load_controlnet_pipeline(
            condition_type="depth",
            checkpoint_dir=str(tmp_path),
            device=torch.device("cpu"),
        )
        assert result is None


class TestLoadBaselinePipeline:
    """Tests for load_baseline_pipeline function."""

    @patch("evaluation.pipeline_loader.CLIPTokenizer.from_pretrained")
    @patch("evaluation.pipeline_loader.CLIPTextModel.from_pretrained")
    @patch("evaluation.pipeline_loader.AutoencoderKL.from_pretrained")
    @patch("evaluation.pipeline_loader.UNet2DConditionModel.from_pretrained")
    @patch("evaluation.pipeline_loader.DDIMScheduler.from_pretrained")
    @patch("evaluation.pipeline_loader.ControlNet")
    @patch("evaluation.pipeline_loader.ControlNetPipeline")
    def test_loads_baseline_pipeline_successfully(
        self,
        mock_pipeline_cls,
        mock_controlnet_cls,
        mock_scheduler,
        mock_unet,
        mock_vae,
        mock_text_encoder,
        mock_tokenizer,
    ):
        """Should successfully create a baseline pipeline with mocked components."""
        # Setup mocks
        mock_unet_instance = MagicMock()
        mock_unet_instance.to.return_value = mock_unet_instance
        mock_unet.return_value = mock_unet_instance

        mock_vae_instance = MagicMock()
        mock_vae_instance.to.return_value = mock_vae_instance
        mock_vae.return_value = mock_vae_instance

        mock_text_encoder_instance = MagicMock()
        mock_text_encoder_instance.to.return_value = mock_text_encoder_instance
        mock_text_encoder.return_value = mock_text_encoder_instance

        mock_controlnet_instance = MagicMock()
        mock_controlnet_instance.to.return_value = mock_controlnet_instance
        mock_controlnet_cls.return_value = mock_controlnet_instance

        mock_pipeline_instance = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline_instance

        result = load_baseline_pipeline(device=torch.device("cpu"))

        assert result is not None
        mock_pipeline_cls.assert_called_once()
        mock_controlnet_instance.eval.assert_called_once()

    def test_returns_none_on_failure(self):
        """Should return None and log warning if loading fails."""
        with patch(
            "evaluation.pipeline_loader.CLIPTokenizer.from_pretrained",
            side_effect=Exception("Network error"),
        ):
            result = load_baseline_pipeline(device=torch.device("cpu"))
            assert result is None

    def test_logs_warning_on_failure(self, caplog):
        """Should log a warning when loading fails."""
        with patch(
            "evaluation.pipeline_loader.CLIPTokenizer.from_pretrained",
            side_effect=Exception("Network error"),
        ):
            with caplog.at_level(logging.WARNING):
                load_baseline_pipeline(device=torch.device("cpu"))

        assert "Failed to load baseline SD1.5 pipeline" in caplog.text


class TestModuleConstants:
    """Tests for module-level constants and configuration."""

    def test_sd15_model_id_is_correct(self):
        """Should use the standard SD1.5 model identifier."""
        assert SD15_MODEL_ID == "runwayml/stable-diffusion-v1-5"
