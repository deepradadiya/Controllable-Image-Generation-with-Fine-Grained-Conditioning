"""
Unit tests for ControlNet Inference Pipeline (model/pipeline.py).

Tests pipeline initialization, condition_type validation, output format,
and overlay composite saving using mock models (no SD1.5 weights required).

Validates: Requirements 3.1, 3.2, 3.4, 3.5
"""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from model.pipeline import ControlNetPipeline


def _make_mock_controlnet():
    """Create a mock ControlNet that returns zero feature tensors."""
    mock = MagicMock()
    mock.return_value = {
        "down_block_res_samples": [torch.zeros(1, 320, 64, 64)] * 12,
        "mid_block_res_sample": torch.zeros(1, 1280, 8, 8),
    }
    return mock


def _make_mock_unet():
    """Create a mock UNet that returns a random noise prediction tensor."""
    mock = MagicMock()
    # UNet returns an object with .sample attribute
    output = MagicMock()
    output.sample = torch.randn(1, 4, 64, 64)
    mock.return_value = output
    # UNet needs parameters() for device/dtype detection — called multiple times
    param = torch.nn.Parameter(torch.zeros(1))
    mock.parameters = MagicMock(side_effect=lambda: iter([param]))
    return mock


def _make_mock_vae():
    """Create a mock VAE whose decode returns a (1, 3, 512, 512) tensor."""
    mock = MagicMock()
    decode_output = MagicMock()
    decode_output.sample = torch.randn(1, 3, 512, 512)
    mock.decode = MagicMock(return_value=decode_output)
    return mock


def _make_mock_text_encoder():
    """Create a mock CLIP text encoder returning (1, 77, 768) embeddings."""
    mock = MagicMock()
    # text_encoder(input_ids)[0] should return a (1, 77, 768) tensor
    mock.return_value = (torch.randn(1, 77, 768),)
    return mock


def _make_mock_tokenizer():
    """Create a mock CLIP tokenizer returning input_ids."""
    mock = MagicMock()
    token_output = MagicMock()
    token_output.input_ids = torch.randint(0, 49408, (1, 77))
    mock.return_value = token_output
    return mock


def _make_mock_scheduler():
    """Create a mock DDIMScheduler with required methods."""
    mock = MagicMock()
    mock.set_timesteps = MagicMock()
    mock.timesteps = torch.tensor([999, 750, 500, 250, 0])
    mock.init_noise_sigma = 1.0
    mock.scale_model_input = MagicMock(side_effect=lambda latents, t: latents)
    # scheduler.step returns an object with .prev_sample
    step_output = MagicMock()
    step_output.prev_sample = torch.randn(1, 4, 64, 64)
    mock.step = MagicMock(return_value=step_output)
    return mock


def _create_pipeline():
    """Create a ControlNetPipeline with all mock components."""
    controlnet = _make_mock_controlnet()
    unet = _make_mock_unet()
    vae = _make_mock_vae()
    text_encoder = _make_mock_text_encoder()
    tokenizer = _make_mock_tokenizer()
    scheduler = _make_mock_scheduler()

    pipeline = ControlNetPipeline(
        controlnet=controlnet,
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
    )
    return pipeline


class TestPipelineInitialization(unittest.TestCase):
    """Test pipeline initialization with mock models."""

    def test_pipeline_initialization(self):
        """Verify all components are stored correctly on the pipeline instance."""
        controlnet = _make_mock_controlnet()
        unet = _make_mock_unet()
        vae = _make_mock_vae()
        text_encoder = _make_mock_text_encoder()
        tokenizer = _make_mock_tokenizer()
        scheduler = _make_mock_scheduler()

        pipeline = ControlNetPipeline(
            controlnet=controlnet,
            unet=unet,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
        )

        self.assertIs(pipeline.controlnet, controlnet)
        self.assertIs(pipeline.unet, unet)
        self.assertIs(pipeline.vae, vae)
        self.assertIs(pipeline.text_encoder, text_encoder)
        self.assertIs(pipeline.tokenizer, tokenizer)
        self.assertIs(pipeline.scheduler, scheduler)


class TestConditionTypeValidation(unittest.TestCase):
    """Test condition_type validation against {"depth", "pose", "edge"}."""

    def test_invalid_condition_type_raises_value_error(self):
        """Passing an invalid condition_type should raise ValueError."""
        pipeline = _create_pipeline()
        condition_image = Image.new("RGB", (512, 512), color=(128, 128, 128))

        with self.assertRaises(ValueError) as ctx:
            pipeline(
                text_prompt="a test prompt",
                condition_image=condition_image,
                condition_type="invalid",
            )

        self.assertIn("invalid", str(ctx.exception).lower())

    def test_valid_condition_types_accepted(self):
        """All valid condition types ("depth", "pose", "edge") should not raise."""
        pipeline = _create_pipeline()
        condition_image = Image.new("RGB", (512, 512), color=(128, 128, 128))

        for ctype in ("depth", "pose", "edge"):
            # Should not raise — just verify it completes without error
            try:
                pipeline(
                    text_prompt="a test prompt",
                    condition_image=condition_image,
                    condition_type=ctype,
                    num_inference_steps=1,
                )
            except ValueError as e:
                self.fail(
                    f"Valid condition_type '{ctype}' raised ValueError: {e}"
                )


class TestPipelineOutput(unittest.TestCase):
    """Test that pipeline output is a valid PIL Image of size 512x512."""

    def test_output_is_pil_image_512x512(self):
        """Run pipeline with mocks and verify output is a 512x512 RGB PIL Image."""
        pipeline = _create_pipeline()
        condition_image = Image.new("RGB", (512, 512), color=(100, 150, 200))

        result = pipeline(
            text_prompt="a beautiful landscape",
            condition_image=condition_image,
            condition_type="depth",
            num_inference_steps=1,
            seed=42,
        )

        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (512, 512))
        self.assertEqual(result.mode, "RGB")


class TestSaveWithOverlay(unittest.TestCase):
    """Test overlay composite saving functionality."""

    def test_save_with_overlay_creates_file(self):
        """Test that save_with_overlay creates a PNG file at the given path."""
        pipeline = _create_pipeline()
        generated = Image.new("RGB", (512, 512), color=(255, 0, 0))
        condition = Image.new("RGB", (512, 512), color=(0, 255, 0))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "overlay_output.png")
            pipeline.save_with_overlay(
                generated=generated,
                condition=condition,
                condition_type="depth",
                output_path=output_path,
            )

            self.assertTrue(os.path.exists(output_path))
            # Verify it's a valid image
            saved_image = Image.open(output_path)
            self.assertEqual(saved_image.format, "PNG")

    def test_overlay_dimensions(self):
        """Verify composite width = condition_width + generated_width (side-by-side layout)."""
        pipeline = _create_pipeline()
        generated = Image.new("RGB", (512, 512), color=(255, 0, 0))
        condition = Image.new("RGB", (512, 512), color=(0, 255, 0))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "overlay_dims.png")
            pipeline.save_with_overlay(
                generated=generated,
                condition=condition,
                condition_type="edge",
                output_path=output_path,
            )

            saved_image = Image.open(output_path)
            composite_width, composite_height = saved_image.size

            # Width should be condition_width + generated_width (side-by-side)
            self.assertEqual(composite_width, 2 * 512)
            # Height should be at least the generated image height
            # (may include label area)
            self.assertGreaterEqual(composite_height, 512)

    def test_aspect_ratio_preserved_on_resize(self):
        """Verify condition image is resized preserving aspect ratio when dimensions differ (Req 11.4)."""
        pipeline = _create_pipeline()
        generated = Image.new("RGB", (512, 512), color=(255, 0, 0))
        # Condition image with different dimensions (wider aspect ratio)
        condition = Image.new("RGB", (800, 400), color=(0, 255, 0))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "overlay_aspect.png")
            pipeline.save_with_overlay(
                generated=generated,
                condition=condition,
                condition_type="depth",
                output_path=output_path,
            )

            saved_image = Image.open(output_path)
            composite_width, composite_height = saved_image.size

            # Condition aspect ratio is 800/400 = 2.0
            # Resized to match generated height (512), new width = 512 * 2.0 = 1024
            expected_cond_width = int(512 * (800 / 400))
            expected_composite_width = expected_cond_width + 512
            self.assertEqual(composite_width, expected_composite_width)
            # Height = generated height + label area
            self.assertGreaterEqual(composite_height, 512)

    def test_filename_includes_condition_type_and_timestamp(self):
        """Verify auto-generated filename includes condition_type and timestamp (Req 11.5)."""
        pipeline = _create_pipeline()
        generated = Image.new("RGB", (512, 512), color=(255, 0, 0))
        condition = Image.new("RGB", (512, 512), color=(0, 255, 0))

        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = pipeline.save_with_overlay(
                generated=generated,
                condition=condition,
                condition_type="pose",
                output_dir=tmpdir,
            )

            filename = os.path.basename(result_path)
            # Filename should start with condition_type
            self.assertTrue(filename.startswith("pose_"))
            # Filename should end with .png
            self.assertTrue(filename.endswith(".png"))
            # File should exist
            self.assertTrue(os.path.exists(result_path))

    def test_filename_includes_condition_type_and_index(self):
        """Verify auto-generated filename includes condition_type and index (Req 11.5)."""
        pipeline = _create_pipeline()
        generated = Image.new("RGB", (512, 512), color=(255, 0, 0))
        condition = Image.new("RGB", (512, 512), color=(0, 255, 0))

        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = pipeline.save_with_overlay(
                generated=generated,
                condition=condition,
                condition_type="edge",
                output_dir=tmpdir,
                index=42,
            )

            filename = os.path.basename(result_path)
            # Filename should be "edge_0042.png"
            self.assertEqual(filename, "edge_0042.png")
            self.assertTrue(os.path.exists(result_path))


if __name__ == "__main__":
    unittest.main()
