"""
Unit tests for inference controls module.

Tests conditioning strength, generation parameters, scheduler integration,
and batch inference support.
"""

import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

import numpy as np
import torch

# Import the module under test
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.inference.controls import (
    GenerationParameters,
    ConditioningStrengthSchedule,
    SchedulerManager,
    SchedulerType,
    BatchInferenceManager,
    create_generator,
    prepare_latents,
    apply_conditioning_scale,
    apply_scheduled_conditioning,
)


class TestGenerationParameters(unittest.TestCase):
    """Tests for GenerationParameters dataclass."""

    def test_default_parameters(self):
        """Test default parameter values are valid."""
        params = GenerationParameters()
        self.assertEqual(params.conditioning_scale, 1.0)
        self.assertEqual(params.num_inference_steps, 50)
        self.assertEqual(params.guidance_scale, 7.5)
        self.assertIsNone(params.seed)
        self.assertEqual(params.image_size, (512, 512))
        self.assertEqual(params.eta, 0.0)

    def test_custom_parameters(self):
        """Test creating parameters with custom values."""
        params = GenerationParameters(
            conditioning_scale=0.8,
            num_inference_steps=30,
            guidance_scale=9.0,
            seed=42,
            image_size=(768, 768),
            eta=0.5,
        )
        self.assertEqual(params.conditioning_scale, 0.8)
        self.assertEqual(params.num_inference_steps, 30)
        self.assertEqual(params.guidance_scale, 9.0)
        self.assertEqual(params.seed, 42)
        self.assertEqual(params.image_size, (768, 768))
        self.assertEqual(params.eta, 0.5)

    def test_conditioning_scale_range(self):
        """Test conditioning scale must be between 0.0 and 2.0."""
        # Valid boundary values
        GenerationParameters(conditioning_scale=0.0)
        GenerationParameters(conditioning_scale=2.0)

        # Invalid values
        with self.assertRaises(ValueError):
            GenerationParameters(conditioning_scale=-0.1)
        with self.assertRaises(ValueError):
            GenerationParameters(conditioning_scale=2.1)

    def test_num_inference_steps_range(self):
        """Test num_inference_steps must be between 1 and 1000."""
        GenerationParameters(num_inference_steps=1)
        GenerationParameters(num_inference_steps=1000)

        with self.assertRaises(ValueError):
            GenerationParameters(num_inference_steps=0)
        with self.assertRaises(ValueError):
            GenerationParameters(num_inference_steps=1001)

    def test_guidance_scale_non_negative(self):
        """Test guidance_scale must be non-negative."""
        GenerationParameters(guidance_scale=0.0)
        GenerationParameters(guidance_scale=20.0)

        with self.assertRaises(ValueError):
            GenerationParameters(guidance_scale=-1.0)

    def test_image_size_must_be_multiple_of_8(self):
        """Test image dimensions must be multiples of 8."""
        GenerationParameters(image_size=(512, 512))
        GenerationParameters(image_size=(768, 512))

        with self.assertRaises(ValueError):
            GenerationParameters(image_size=(513, 512))
        with self.assertRaises(ValueError):
            GenerationParameters(image_size=(512, 511))

    def test_image_size_minimum(self):
        """Test image dimensions must be at least 64."""
        GenerationParameters(image_size=(64, 64))

        with self.assertRaises(ValueError):
            GenerationParameters(image_size=(56, 56))

    def test_eta_range(self):
        """Test eta must be between 0.0 and 1.0."""
        GenerationParameters(eta=0.0)
        GenerationParameters(eta=1.0)

        with self.assertRaises(ValueError):
            GenerationParameters(eta=-0.1)
        with self.assertRaises(ValueError):
            GenerationParameters(eta=1.1)


class TestConditioningStrengthSchedule(unittest.TestCase):
    """Tests for ConditioningStrengthSchedule."""

    def test_constant_schedule(self):
        """Test constant schedule returns same value at all steps."""
        schedule = ConditioningStrengthSchedule(
            start_scale=0.8, end_scale=0.8, schedule_type="constant"
        )
        for step in range(10):
            self.assertAlmostEqual(schedule.get_scale_at_step(step, 10), 0.8)

    def test_linear_schedule(self):
        """Test linear interpolation between start and end scales."""
        schedule = ConditioningStrengthSchedule(
            start_scale=1.0, end_scale=0.5, schedule_type="linear"
        )
        # At step 0, should be start_scale
        self.assertAlmostEqual(schedule.get_scale_at_step(0, 10), 1.0)
        # At last step, should be end_scale
        self.assertAlmostEqual(schedule.get_scale_at_step(9, 10), 0.5)
        # At midpoint, should be average
        self.assertAlmostEqual(schedule.get_scale_at_step(4, 9), 0.75, places=5)

    def test_cosine_schedule(self):
        """Test cosine schedule provides smooth interpolation."""
        schedule = ConditioningStrengthSchedule(
            start_scale=1.0, end_scale=0.0, schedule_type="cosine"
        )
        # At step 0, should be start_scale
        self.assertAlmostEqual(schedule.get_scale_at_step(0, 10), 1.0)
        # At last step, should be end_scale
        self.assertAlmostEqual(schedule.get_scale_at_step(9, 10), 0.0, places=5)
        # Cosine should be smoother than linear at midpoint
        mid_value = schedule.get_scale_at_step(4, 9)
        self.assertTrue(0.0 < mid_value < 1.0)

    def test_single_step(self):
        """Test schedule with total_steps=1 returns start_scale."""
        schedule = ConditioningStrengthSchedule(
            start_scale=0.7, end_scale=0.3, schedule_type="linear"
        )
        self.assertAlmostEqual(schedule.get_scale_at_step(0, 1), 0.7)

    def test_invalid_schedule_type(self):
        """Test invalid schedule type raises ValueError."""
        with self.assertRaises(ValueError):
            ConditioningStrengthSchedule(schedule_type="invalid")

    def test_scale_validation(self):
        """Test scale values must be in valid range."""
        with self.assertRaises(ValueError):
            ConditioningStrengthSchedule(start_scale=-0.1)
        with self.assertRaises(ValueError):
            ConditioningStrengthSchedule(end_scale=2.1)


class TestSchedulerManager(unittest.TestCase):
    """Tests for SchedulerManager."""

    def test_initialization_with_string(self):
        """Test initialization with string scheduler type."""
        manager = SchedulerManager(scheduler_type="ddim")
        self.assertEqual(manager.scheduler_type, SchedulerType.DDIM)

    def test_initialization_with_enum(self):
        """Test initialization with enum scheduler type."""
        manager = SchedulerManager(scheduler_type=SchedulerType.PNDM)
        self.assertEqual(manager.scheduler_type, SchedulerType.PNDM)

    def test_create_ddim_scheduler(self):
        """Test DDIM scheduler creation."""
        manager = SchedulerManager(scheduler_type=SchedulerType.DDIM)
        scheduler = manager.create_scheduler()
        self.assertIsNotNone(scheduler)

    def test_create_pndm_scheduler(self):
        """Test PNDM scheduler creation."""
        manager = SchedulerManager(scheduler_type=SchedulerType.PNDM)
        scheduler = manager.create_scheduler()
        self.assertIsNotNone(scheduler)

    def test_create_euler_scheduler(self):
        """Test Euler scheduler creation."""
        manager = SchedulerManager(scheduler_type=SchedulerType.EULER)
        scheduler = manager.create_scheduler()
        self.assertIsNotNone(scheduler)

    def test_set_timesteps(self):
        """Test setting inference timesteps."""
        manager = SchedulerManager(scheduler_type=SchedulerType.DDIM)
        manager.set_timesteps(20, device="cpu")
        timesteps = manager.get_timesteps()
        self.assertEqual(len(timesteps), 20)

    def test_get_timesteps_without_setting(self):
        """Test getting timesteps reflects scheduler state."""
        manager = SchedulerManager(scheduler_type=SchedulerType.DDIM)
        manager.create_scheduler()
        # DDIM scheduler in diffusers initializes with default timesteps,
        # so we verify that set_timesteps changes the count properly
        manager.set_timesteps(25, device="cpu")
        timesteps = manager.get_timesteps()
        self.assertEqual(len(timesteps), 25)

    def test_scheduler_step(self):
        """Test performing a scheduler step."""
        manager = SchedulerManager(scheduler_type=SchedulerType.DDIM)
        manager.set_timesteps(20, device="cpu")
        timesteps = manager.get_timesteps()

        # Create dummy model output and sample
        model_output = torch.randn(1, 4, 64, 64)
        sample = torch.randn(1, 4, 64, 64)

        result = manager.step(
            model_output=model_output,
            timestep=timesteps[0],
            sample=sample,
            eta=0.0,
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.prev_sample.shape, sample.shape)

    def test_scale_model_input(self):
        """Test model input scaling."""
        manager = SchedulerManager(scheduler_type=SchedulerType.EULER)
        manager.set_timesteps(20, device="cpu")
        timesteps = manager.get_timesteps()

        sample = torch.randn(1, 4, 64, 64)
        scaled = manager.scale_model_input(sample, timesteps[0])
        self.assertEqual(scaled.shape, sample.shape)

    def test_init_noise_sigma(self):
        """Test getting initial noise sigma."""
        manager = SchedulerManager(scheduler_type=SchedulerType.DDIM)
        manager.create_scheduler()
        sigma = manager.init_noise_sigma
        self.assertIsInstance(sigma, (int, float, torch.Tensor))

    def test_get_scheduler_lazy_creation(self):
        """Test get_scheduler creates scheduler on first call."""
        manager = SchedulerManager(scheduler_type=SchedulerType.DDIM)
        self.assertIsNone(manager._scheduler)
        scheduler = manager.get_scheduler()
        self.assertIsNotNone(scheduler)
        self.assertIsNotNone(manager._scheduler)


class TestBatchInferenceManager(unittest.TestCase):
    """Tests for BatchInferenceManager."""

    def test_default_initialization(self):
        """Test default initialization values."""
        manager = BatchInferenceManager()
        self.assertEqual(manager.max_batch_size, 4)
        self.assertEqual(manager.gpu_memory_limit_mb, 13000.0)
        self.assertTrue(manager.enable_memory_optimization)

    def test_compute_optimal_batch_size_512(self):
        """Test optimal batch size for 512x512 images."""
        manager = BatchInferenceManager(max_batch_size=4, gpu_memory_limit_mb=13000.0)
        batch_size = manager.compute_optimal_batch_size((512, 512), 8)
        self.assertGreaterEqual(batch_size, 1)
        self.assertLessEqual(batch_size, 4)

    def test_compute_optimal_batch_size_large_image(self):
        """Test optimal batch size for large images is smaller."""
        manager = BatchInferenceManager(max_batch_size=4, gpu_memory_limit_mb=13000.0)
        batch_512 = manager.compute_optimal_batch_size((512, 512), 8)
        batch_1024 = manager.compute_optimal_batch_size((1024, 1024), 8)
        self.assertGreaterEqual(batch_512, batch_1024)

    def test_compute_optimal_batch_size_no_optimization(self):
        """Test batch size without memory optimization uses max_batch_size."""
        manager = BatchInferenceManager(
            max_batch_size=4, enable_memory_optimization=False
        )
        batch_size = manager.compute_optimal_batch_size((512, 512), 8)
        self.assertEqual(batch_size, 4)

    def test_compute_optimal_batch_size_fewer_images(self):
        """Test batch size doesn't exceed number of images."""
        manager = BatchInferenceManager(max_batch_size=4)
        batch_size = manager.compute_optimal_batch_size((512, 512), 2)
        self.assertLessEqual(batch_size, 2)

    def test_create_batch_schedule(self):
        """Test batch schedule creation."""
        manager = BatchInferenceManager(max_batch_size=4, enable_memory_optimization=False)
        schedule = manager.create_batch_schedule(10, (512, 512))
        self.assertEqual(sum(schedule), 10)
        self.assertTrue(all(s <= 4 for s in schedule))

    def test_create_batch_schedule_empty(self):
        """Test batch schedule with zero images."""
        manager = BatchInferenceManager()
        schedule = manager.create_batch_schedule(0, (512, 512))
        self.assertEqual(schedule, [])

    def test_prepare_batch_inputs_single_prompt(self):
        """Test preparing batch inputs with a single prompt."""
        manager = BatchInferenceManager()
        params = GenerationParameters(seed=42)
        condition_map = torch.randn(3, 512, 512)

        inputs = manager.prepare_batch_inputs(
            prompts="a cat",
            condition_maps=condition_map,
            params=params,
            batch_size=2,
            batch_index=0,
        )

        self.assertEqual(len(inputs["prompts"]), 2)
        self.assertEqual(inputs["prompts"], ["a cat", "a cat"])
        self.assertEqual(inputs["condition_maps"].shape[0], 2)
        self.assertEqual(inputs["seed"], 42)

    def test_prepare_batch_inputs_multiple_prompts(self):
        """Test preparing batch inputs with multiple prompts."""
        manager = BatchInferenceManager()
        params = GenerationParameters(seed=10)
        prompts = ["a cat", "a dog", "a bird", "a fish"]
        condition_maps = torch.randn(4, 3, 512, 512)

        inputs = manager.prepare_batch_inputs(
            prompts=prompts,
            condition_maps=condition_maps,
            params=params,
            batch_size=2,
            batch_index=1,
        )

        self.assertEqual(inputs["prompts"], ["a bird", "a fish"])
        self.assertEqual(inputs["condition_maps"].shape[0], 2)
        self.assertEqual(inputs["seed"], 11)  # seed + batch_index

    def test_get_memory_estimate(self):
        """Test memory estimation."""
        manager = BatchInferenceManager()
        mem_1 = manager.get_memory_estimate((512, 512), 1)
        mem_2 = manager.get_memory_estimate((512, 512), 2)
        self.assertGreater(mem_2, mem_1)


class TestCreateGenerator(unittest.TestCase):
    """Tests for create_generator function."""

    def test_none_seed_returns_none(self):
        """Test that None seed returns None generator."""
        gen = create_generator(None)
        self.assertIsNone(gen)

    def test_valid_seed_returns_generator(self):
        """Test that a valid seed returns a Generator."""
        gen = create_generator(42)
        self.assertIsInstance(gen, torch.Generator)

    def test_reproducibility(self):
        """Test that same seed produces same random values."""
        gen1 = create_generator(42)
        gen2 = create_generator(42)
        val1 = torch.randn(3, generator=gen1)
        val2 = torch.randn(3, generator=gen2)
        self.assertTrue(torch.allclose(val1, val2))


class TestPrepareLatents(unittest.TestCase):
    """Tests for prepare_latents function."""

    def test_generates_random_latents(self):
        """Test generating random latents with correct shape."""
        latents = prepare_latents(
            batch_size=2,
            num_channels=4,
            height=64,
            width=64,
            dtype=torch.float32,
            device="cpu",
        )
        self.assertEqual(latents.shape, (2, 4, 64, 64))

    def test_uses_provided_latents(self):
        """Test using pre-computed latents."""
        provided = torch.ones(1, 4, 64, 64)
        latents = prepare_latents(
            batch_size=1,
            num_channels=4,
            height=64,
            width=64,
            dtype=torch.float32,
            device="cpu",
            latents=provided,
            scheduler_init_noise_sigma=1.0,
        )
        self.assertTrue(torch.allclose(latents, provided))

    def test_shape_mismatch_raises_error(self):
        """Test that mismatched latent shape raises ValueError."""
        wrong_shape = torch.ones(1, 4, 32, 32)
        with self.assertRaises(ValueError):
            prepare_latents(
                batch_size=1,
                num_channels=4,
                height=64,
                width=64,
                dtype=torch.float32,
                device="cpu",
                latents=wrong_shape,
            )

    def test_scales_by_init_noise_sigma(self):
        """Test latents are scaled by scheduler init_noise_sigma."""
        gen = create_generator(42)
        latents = prepare_latents(
            batch_size=1,
            num_channels=4,
            height=64,
            width=64,
            dtype=torch.float32,
            device="cpu",
            generator=gen,
            scheduler_init_noise_sigma=2.0,
        )
        # Regenerate without scaling to compare
        gen2 = create_generator(42)
        unscaled = torch.randn(1, 4, 64, 64, generator=gen2)
        expected = unscaled * 2.0
        self.assertTrue(torch.allclose(latents, expected))

    def test_reproducibility_with_generator(self):
        """Test reproducibility with same generator seed."""
        latents1 = prepare_latents(
            batch_size=1, num_channels=4, height=64, width=64,
            dtype=torch.float32, device="cpu",
            generator=create_generator(123),
        )
        latents2 = prepare_latents(
            batch_size=1, num_channels=4, height=64, width=64,
            dtype=torch.float32, device="cpu",
            generator=create_generator(123),
        )
        self.assertTrue(torch.allclose(latents1, latents2))


class TestApplyConditioningScale(unittest.TestCase):
    """Tests for apply_conditioning_scale function."""

    def test_scale_1_returns_same(self):
        """Test that scale=1.0 returns the same outputs."""
        outputs = {
            "down_block_res_samples": [torch.randn(1, 320, 64, 64)],
            "mid_block_res_sample": torch.randn(1, 1280, 8, 8),
        }
        scaled = apply_conditioning_scale(outputs, 1.0)
        self.assertIs(scaled, outputs)

    def test_scale_0_zeros_outputs(self):
        """Test that scale=0.0 zeros all outputs."""
        outputs = {
            "down_block_res_samples": [torch.ones(1, 320, 64, 64)],
            "mid_block_res_sample": torch.ones(1, 1280, 8, 8),
        }
        scaled = apply_conditioning_scale(outputs, 0.0)
        self.assertTrue(torch.allclose(scaled["down_block_res_samples"][0], torch.zeros(1, 320, 64, 64)))
        self.assertTrue(torch.allclose(scaled["mid_block_res_sample"], torch.zeros(1, 1280, 8, 8)))

    def test_scale_half(self):
        """Test that scale=0.5 halves all outputs."""
        outputs = {
            "down_block_res_samples": [torch.ones(1, 320, 64, 64) * 2.0],
            "mid_block_res_sample": torch.ones(1, 1280, 8, 8) * 4.0,
        }
        scaled = apply_conditioning_scale(outputs, 0.5)
        self.assertTrue(torch.allclose(scaled["down_block_res_samples"][0], torch.ones(1, 320, 64, 64)))
        self.assertTrue(torch.allclose(scaled["mid_block_res_sample"], torch.ones(1, 1280, 8, 8) * 2.0))


class TestApplyScheduledConditioning(unittest.TestCase):
    """Tests for apply_scheduled_conditioning function."""

    def test_constant_schedule_same_at_all_steps(self):
        """Test constant schedule applies same scale at all steps."""
        schedule = ConditioningStrengthSchedule(
            start_scale=0.8, schedule_type="constant"
        )
        outputs = {
            "down_block_res_samples": [torch.ones(1, 320, 64, 64)],
            "mid_block_res_sample": torch.ones(1, 1280, 8, 8),
        }

        for step in range(10):
            scaled = apply_scheduled_conditioning(outputs, schedule, step, 10)
            expected = torch.ones(1, 320, 64, 64) * 0.8
            self.assertTrue(
                torch.allclose(scaled["down_block_res_samples"][0], expected, atol=1e-6)
            )

    def test_linear_schedule_varies_across_steps(self):
        """Test linear schedule produces different scales at different steps."""
        schedule = ConditioningStrengthSchedule(
            start_scale=1.0, end_scale=0.0, schedule_type="linear"
        )
        outputs = {
            "down_block_res_samples": [torch.ones(1, 320, 64, 64)],
            "mid_block_res_sample": torch.ones(1, 1280, 8, 8),
        }

        # First step should be full strength
        scaled_first = apply_scheduled_conditioning(outputs, schedule, 0, 10)
        self.assertTrue(
            torch.allclose(scaled_first["down_block_res_samples"][0], torch.ones(1, 320, 64, 64))
        )

        # Last step should be zero
        scaled_last = apply_scheduled_conditioning(outputs, schedule, 9, 10)
        self.assertTrue(
            torch.allclose(scaled_last["down_block_res_samples"][0], torch.zeros(1, 320, 64, 64), atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
