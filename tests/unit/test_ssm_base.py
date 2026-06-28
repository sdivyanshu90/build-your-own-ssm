"""Unit tests for :mod:`mamba.core.ssm_base`.

The defining property of an LTI SSM is that its recurrent and convolutional
evaluations are mathematically identical. These tests verify that equivalence,
the kernel shape, and the mode-switching dispatcher, and that the abstract base
cannot be instantiated.
"""

from __future__ import annotations

import pytest
import torch

from mamba.core.ssm_base import ContinuousSSM, SSMBase


class TestContinuousSSM:
    def test_kernel_shape(self) -> None:
        """The convolution kernel has length L."""
        ssm = ContinuousSSM(d_state=16)
        assert ssm.compute_kernel(32).shape == (32,)

    def test_output_shape(self) -> None:
        """Both paths map (batch, L) -> (batch, L)."""
        ssm = ContinuousSSM(d_state=16)
        u = torch.randn(3, 24)
        assert ssm.forward_conv(u).shape == (3, 24)
        assert ssm.forward_recurrent(u).shape == (3, 24)

    def test_conv_recurrent_equivalence(self) -> None:
        """Convolutional and recurrent modes agree for the LTI model."""
        torch.manual_seed(0)
        ssm = ContinuousSSM(d_state=32, dt=0.01)
        u = torch.randn(2, 64)
        y_conv = ssm.forward_conv(u)
        y_rec = ssm.forward_recurrent(u)
        torch.testing.assert_close(y_conv, y_rec, atol=1e-4, rtol=1e-3)

    def test_mode_switching(self) -> None:
        """``forward`` dispatches per the pinned mode / training flag."""
        ssm = ContinuousSSM(d_state=8)
        u = torch.randn(1, 10)
        ssm.set_training_mode("conv")
        torch.testing.assert_close(ssm(u), ssm.forward_conv(u), atol=1e-5, rtol=1e-4)
        ssm.set_training_mode("recurrent")
        torch.testing.assert_close(
            ssm(u), ssm.forward_recurrent(u), atol=1e-5, rtol=1e-4
        )

    def test_invalid_mode_raises(self) -> None:
        """An unknown mode string is rejected."""
        ssm = ContinuousSSM(d_state=8)
        with pytest.raises(ValueError):
            ssm.set_training_mode("bogus")  # type: ignore[arg-type]

    def test_get_ssm_matrices(self) -> None:
        """The discrete-matrix accessor returns the expected keys/shapes."""
        ssm = ContinuousSSM(d_state=8)
        mats = ssm.get_ssm_matrices()
        assert set(mats) == {"A_bar", "B_bar", "C", "D"}
        assert mats["A_bar"].shape == (8, 8)
        assert mats["B_bar"].shape == (8, 1)

    def test_gradients_flow(self) -> None:
        """Training-mode forward produces finite gradients for all params."""
        ssm = ContinuousSSM(d_state=8)
        u = torch.randn(2, 16)
        ssm.train()
        ssm(u).sum().backward()
        for name, p in ssm.named_parameters():
            assert p.grad is not None, f"no grad for {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


class TestSSMBase:
    def test_abstract_cannot_instantiate(self) -> None:
        """SSMBase is abstract and cannot be constructed directly."""
        with pytest.raises(TypeError):
            SSMBase()  # type: ignore[abstract]
