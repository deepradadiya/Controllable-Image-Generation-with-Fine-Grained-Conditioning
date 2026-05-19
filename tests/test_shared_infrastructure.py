"""
Integration Tests for Shared Training Infrastructure.

Tests the shared utilities in training/utils.py that are used by all three
training scripts (depth, pose, edge):

1. Optimizer configuration matches spec (AdamW, lr=1e-5, betas, weight_decay, eps)
2. Checkpoint save and load produces identical model state
3. Checkpoint max retention (only 3 most recent kept)
4. Cosine LR schedule decreases over time after warmup
5. Cosine LR schedule warmup behavior

**Validates: Requirements 10.1, 10.2, 10.3**
"""

import os
import sys
import tempfile

import pytest
import torch
import torch.nn as nn

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.utils import (
    load_checkpoint,
    save_checkpoint,
    setup_optimizer,
    setup_scheduler,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_model():
    """A simple nn.Linear model for testing checkpoint and optimizer behavior."""
    model = nn.Linear(10, 5)
    return model


@pytest.fixture
def optimizer_and_model(simple_model):
    """Set up optimizer using the shared setup_optimizer function."""
    optimizer = setup_optimizer(simple_model.parameters())
    return optimizer, simple_model


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Optimizer Configuration Matches Spec
# **Validates: Requirement 10.1**
# ─────────────────────────────────────────────────────────────────────────────


class TestOptimizerConfigMatchesSpec:
    """
    Verify that setup_optimizer produces an AdamW optimizer with the exact
    hyperparameters specified in the requirements:
    - lr=1e-5
    - betas=(0.9, 0.999)
    - weight_decay=1e-2
    - eps=1e-8

    **Validates: Requirements 10.1**
    """

    def test_optimizer_config_matches_spec(self, simple_model):
        """Verify all optimizer hyperparameters match the spec exactly."""
        optimizer = setup_optimizer(simple_model.parameters())

        # Should be AdamW
        assert isinstance(optimizer, torch.optim.AdamW), (
            f"Expected AdamW optimizer, got {type(optimizer).__name__}"
        )

        # Check hyperparameters in param_groups
        assert len(optimizer.param_groups) == 1
        pg = optimizer.param_groups[0]

        assert pg["lr"] == 1e-5, f"Expected lr=1e-5, got {pg['lr']}"
        assert pg["betas"] == (0.9, 0.999), f"Expected betas=(0.9, 0.999), got {pg['betas']}"
        assert pg["weight_decay"] == 1e-2, f"Expected weight_decay=1e-2, got {pg['weight_decay']}"
        assert pg["eps"] == 1e-8, f"Expected eps=1e-8, got {pg['eps']}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Checkpoint Save and Load Roundtrip
# **Validates: Requirement 10.3**
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckpointSaveLoadRoundtrip:
    """
    Verify that saving a checkpoint and loading it into a fresh model
    produces identical model state, optimizer state, and scheduler state.

    **Validates: Requirements 10.3**
    """

    def test_checkpoint_save_load_roundtrip(self, simple_model):
        """Save checkpoint, load into fresh model, verify state_dict matches."""
        optimizer = setup_optimizer(simple_model.parameters())
        scheduler = setup_scheduler(optimizer, num_training_steps=1000, warmup_steps=100)

        # Simulate a few optimizer steps to create non-trivial state
        for _ in range(5):
            dummy_loss = (simple_model(torch.randn(2, 10)) ** 2).sum()
            optimizer.zero_grad()
            dummy_loss.backward()
            optimizer.step()
            scheduler.step()

        # Record the state before saving
        original_model_state = {k: v.clone() for k, v in simple_model.state_dict().items()}
        original_step = 42

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save checkpoint
            save_checkpoint(
                model=simple_model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=original_step,
                output_dir=tmpdir,
                max_checkpoints=3,
            )

            # Create a fresh model and optimizer/scheduler
            fresh_model = nn.Linear(10, 5)
            fresh_optimizer = setup_optimizer(fresh_model.parameters())
            fresh_scheduler = setup_scheduler(fresh_optimizer, num_training_steps=1000, warmup_steps=100)

            # Load checkpoint
            checkpoint_dir = os.path.join(tmpdir, f"checkpoint-{original_step}")
            loaded_step = load_checkpoint(
                model=fresh_model,
                optimizer=fresh_optimizer,
                scheduler=fresh_scheduler,
                checkpoint_path=checkpoint_dir,
            )

            # Verify step matches
            assert loaded_step == original_step, (
                f"Expected step={original_step}, got step={loaded_step}"
            )

            # Verify model state_dict matches exactly
            for key in original_model_state:
                assert torch.equal(
                    fresh_model.state_dict()[key],
                    original_model_state[key],
                ), f"Model state mismatch for key '{key}' after checkpoint load"

            # Verify optimizer state matches
            orig_opt_state = optimizer.state_dict()
            loaded_opt_state = fresh_optimizer.state_dict()
            assert orig_opt_state["param_groups"] == loaded_opt_state["param_groups"], (
                "Optimizer param_groups mismatch after checkpoint load"
            )

            # Verify scheduler state matches
            orig_sched_state = scheduler.state_dict()
            loaded_sched_state = fresh_scheduler.state_dict()
            assert orig_sched_state == loaded_sched_state, (
                "Scheduler state mismatch after checkpoint load"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Checkpoint Max Retention
# **Validates: Requirement 10.3**
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckpointMaxRetention:
    """
    Verify that save_checkpoint retains only the max_checkpoints most recent
    checkpoints and removes older ones.

    **Validates: Requirements 10.3**
    """

    def test_checkpoint_max_retention(self, simple_model):
        """Save 5 checkpoints with max_checkpoints=3, verify only 3 remain."""
        optimizer = setup_optimizer(simple_model.parameters())
        scheduler = setup_scheduler(optimizer, num_training_steps=1000, warmup_steps=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save 5 checkpoints
            for step in [100, 200, 300, 400, 500]:
                save_checkpoint(
                    model=simple_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    output_dir=tmpdir,
                    max_checkpoints=3,
                )

            # List remaining checkpoint directories
            checkpoint_dirs = sorted(
                [d for d in os.listdir(tmpdir) if d.startswith("checkpoint-")]
            )

            # Only 3 most recent should remain
            assert len(checkpoint_dirs) == 3, (
                f"Expected 3 checkpoints, found {len(checkpoint_dirs)}: {checkpoint_dirs}"
            )

            # The 3 most recent should be steps 300, 400, 500
            expected = ["checkpoint-300", "checkpoint-400", "checkpoint-500"]
            assert checkpoint_dirs == expected, (
                f"Expected {expected}, got {checkpoint_dirs}"
            )

            # Verify old checkpoints are actually deleted
            assert not os.path.exists(os.path.join(tmpdir, "checkpoint-100"))
            assert not os.path.exists(os.path.join(tmpdir, "checkpoint-200"))


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Cosine LR Schedule Decreases Over Time
# **Validates: Requirement 10.2**
# ─────────────────────────────────────────────────────────────────────────────


class TestCosineLRScheduleDecreases:
    """
    Verify that the cosine LR schedule increases during warmup and then
    decreases after warmup completes.

    **Validates: Requirements 10.2**
    """

    @pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`")
    def test_cosine_lr_schedule_decreases(self, simple_model):
        """
        After warmup, the learning rate should decrease following a cosine curve.
        Verify LR at step 600 < LR at step 500 (just after warmup peak).
        """
        optimizer = setup_optimizer(simple_model.parameters())
        scheduler = setup_scheduler(optimizer, num_training_steps=2000, warmup_steps=500)

        # Step through warmup to reach peak LR
        for _ in range(500):
            scheduler.step()

        lr_at_warmup_end = optimizer.param_groups[0]["lr"]

        # Step further into cosine decay
        for _ in range(500):
            scheduler.step()

        lr_at_step_1000 = optimizer.param_groups[0]["lr"]

        # Step even further
        for _ in range(500):
            scheduler.step()

        lr_at_step_1500 = optimizer.param_groups[0]["lr"]

        # LR should decrease after warmup
        assert lr_at_step_1000 < lr_at_warmup_end, (
            f"LR should decrease after warmup. "
            f"LR at warmup end: {lr_at_warmup_end:.8f}, "
            f"LR at step 1000: {lr_at_step_1000:.8f}"
        )

        assert lr_at_step_1500 < lr_at_step_1000, (
            f"LR should continue decreasing. "
            f"LR at step 1000: {lr_at_step_1000:.8f}, "
            f"LR at step 1500: {lr_at_step_1500:.8f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Cosine LR Schedule Warmup
# **Validates: Requirement 10.2**
# ─────────────────────────────────────────────────────────────────────────────


class TestCosineLRScheduleWarmup:
    """
    Verify that the LR starts near 0 and increases to the target LR
    during the warmup phase.

    **Validates: Requirements 10.2**
    """

    @pytest.mark.filterwarnings("ignore:Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`")
    def test_cosine_lr_schedule_warmup(self, simple_model):
        """
        Verify LR starts near 0 and reaches target (1e-5) after warmup_steps.
        """
        optimizer = setup_optimizer(simple_model.parameters())
        scheduler = setup_scheduler(optimizer, num_training_steps=2000, warmup_steps=500)

        # LR at the very start (before any step) should be near 0
        initial_lr = optimizer.param_groups[0]["lr"]

        # Step once to get the first scheduled LR
        scheduler.step()
        lr_after_first_step = optimizer.param_groups[0]["lr"]

        # LR should be very small at the start of warmup
        assert lr_after_first_step < 1e-5, (
            f"LR should be less than target at start of warmup. "
            f"Got {lr_after_first_step:.10f}"
        )

        # Step through the rest of warmup
        for _ in range(499):
            scheduler.step()

        lr_at_warmup_end = optimizer.param_groups[0]["lr"]

        # LR should be close to target (1e-5) after warmup
        assert abs(lr_at_warmup_end - 1e-5) < 1e-7, (
            f"LR should reach target 1e-5 after warmup. "
            f"Got {lr_at_warmup_end:.10f}"
        )

        # Verify LR increased during warmup
        assert lr_at_warmup_end > lr_after_first_step, (
            f"LR should increase during warmup. "
            f"Start: {lr_after_first_step:.10f}, End: {lr_at_warmup_end:.10f}"
        )
