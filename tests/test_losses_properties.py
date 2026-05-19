"""
Property-based tests for diffusion loss computation.

**Validates: Requirements 4.2**

Tests the mathematical properties of MSE loss used in diffusion training:
- Non-negativity: MSE >= 0 for any inputs
- Identity: MSE(x, x) == 0
- Symmetry: MSE(a, b) == MSE(b, a)
"""

import torch
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from training.losses import compute_diffusion_loss


# --- Hypothesis strategies for generating random tensors ---

@st.composite
def tensor_shape(draw):
    """Generate valid tensor shapes for diffusion loss inputs.

    Produces shapes (B, C, H, W) with reasonable dimensions.
    """
    batch_size = draw(st.integers(min_value=1, max_value=4))
    channels = draw(st.sampled_from([1, 4, 8]))
    height = draw(st.integers(min_value=2, max_value=16))
    width = draw(st.integers(min_value=2, max_value=16))
    return (batch_size, channels, height, width)





# --- Property Tests ---


class TestDiffusionLossProperties:
    """Property-based tests for compute_diffusion_loss.

    **Validates: Requirements 4.2**
    """

    @given(data=st.data(), step=st.integers(min_value=1, max_value=10000).filter(lambda x: x % 10 != 0))
    @settings(max_examples=100, deadline=None)
    def test_loss_is_non_negative(self, data, step):
        """Property: Loss is non-negative — MSE loss >= 0 for any inputs.

        **Validates: Requirements 4.2**

        MSE is defined as mean((a - b)^2), which is a mean of squared values.
        Squared values are always >= 0, so their mean must also be >= 0.
        """
        shape = data.draw(tensor_shape())
        model_pred = torch.randn(shape)
        noise = torch.randn(shape)
        batch_size = shape[0]
        timesteps = torch.randint(0, 1000, (batch_size,))

        loss = compute_diffusion_loss(model_pred, noise, timesteps, step)

        assert loss.item() >= 0.0, (
            f"Loss should be non-negative but got {loss.item()} "
            f"for shape {shape}"
        )

    @given(data=st.data(), step=st.integers(min_value=1, max_value=10000).filter(lambda x: x % 10 != 0))
    @settings(max_examples=100, deadline=None)
    def test_loss_is_zero_when_pred_equals_target(self, data, step):
        """Property: Loss is zero when pred == target — MSE(x, x) == 0.

        **Validates: Requirements 4.2**

        When the predicted noise exactly matches the actual noise,
        the difference is zero everywhere, so MSE must be exactly 0.
        """
        shape = data.draw(tensor_shape())
        tensor = torch.randn(shape)
        batch_size = shape[0]
        timesteps = torch.randint(0, 1000, (batch_size,))

        loss = compute_diffusion_loss(tensor, tensor, timesteps, step)

        assert loss.item() == 0.0, (
            f"Loss should be exactly 0 when pred == target but got {loss.item()} "
            f"for shape {shape}"
        )

    @given(data=st.data(), step=st.integers(min_value=1, max_value=10000).filter(lambda x: x % 10 != 0))
    @settings(max_examples=100, deadline=None)
    def test_loss_is_symmetric(self, data, step):
        """Property: Loss is symmetric — MSE(a, b) == MSE(b, a).

        **Validates: Requirements 4.2**

        MSE(a, b) = mean((a - b)^2) = mean((b - a)^2) = MSE(b, a)
        since (a - b)^2 == (b - a)^2 for all values.
        """
        shape = data.draw(tensor_shape())
        a = torch.randn(shape)
        b = torch.randn(shape)
        batch_size = shape[0]
        timesteps = torch.randint(0, 1000, (batch_size,))

        loss_ab = compute_diffusion_loss(a, b, timesteps, step)
        loss_ba = compute_diffusion_loss(b, a, timesteps, step)

        assert torch.isclose(loss_ab, loss_ba, atol=1e-6), (
            f"Loss should be symmetric but MSE(a,b)={loss_ab.item()} != "
            f"MSE(b,a)={loss_ba.item()} for shape {shape}"
        )
