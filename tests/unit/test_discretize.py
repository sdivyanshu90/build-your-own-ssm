"""Unit tests for :mod:`mamba.core.discretize`.

These tests pin down the mathematical contract of each discretization rule:
correct shapes, the continuous limit, stability preservation, and agreement
with the SciPy reference (``scipy.signal.cont2discrete``).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.linalg
import scipy.signal
import torch

from mamba.core.discretize import bilinear, euler, selective_zoh, zoh


def _random_stable_A(n: int, seed: int = 0) -> torch.Tensor:
    """Build a random matrix whose eigenvalues all have negative real part."""
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(n, n, generator=g, dtype=torch.float64)
    # -(M M^T + I) is symmetric negative definite -> real negative eigenvalues.
    return -(M @ M.T + torch.eye(n, dtype=torch.float64))


class TestZOH:
    def test_output_shapes(self) -> None:
        """ZOH preserves the (N, N) / (N, M) shapes of A and B."""
        A = torch.randn(4, 4)
        B = torch.randn(4, 2)
        A_bar, B_bar = zoh(A, B, 0.1)
        assert A_bar.shape == (4, 4)
        assert B_bar.shape == (4, 2)

    def test_continuous_limit_as_delta_to_zero(self) -> None:
        r"""As :math:`\Delta \to 0`, :math:`\bar A \to I` and :math:`\bar B \to 0`."""
        A = _random_stable_A(5).to(torch.float32)
        B = torch.randn(5, 1)
        A_bar, B_bar = zoh(A, B, 1e-7)
        torch.testing.assert_close(A_bar, torch.eye(5), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(B_bar, torch.zeros(5, 1), atol=1e-5, rtol=1e-5)

    def test_stability_preservation(self) -> None:
        """A continuous-stable A maps to a discrete-stable A_bar (|eig| <= 1)."""
        A = _random_stable_A(6)
        B = torch.randn(6, 1, dtype=torch.float64)
        A_bar, _ = zoh(A, B, 0.5)
        eig = torch.linalg.eigvals(A_bar)
        assert torch.all(eig.abs() <= 1.0 + 1e-6)

    def test_zoh_vs_matrix_exponential_reference(self) -> None:
        """Match scipy.signal.cont2discrete (the canonical ZOH reference)."""
        A = _random_stable_A(4).numpy()
        B = np.random.default_rng(0).standard_normal((4, 2))
        C = np.zeros((1, 4))
        D = np.zeros((1, 2))
        dt = 0.3
        Ad, Bd, *_ = scipy.signal.cont2discrete((A, B, C, D), dt, method="zoh")
        A_bar, B_bar = zoh(torch.from_numpy(A), torch.from_numpy(B), dt)
        torch.testing.assert_close(A_bar, torch.from_numpy(Ad), atol=1e-8, rtol=1e-6)
        torch.testing.assert_close(B_bar, torch.from_numpy(Bd), atol=1e-8, rtol=1e-6)

    def test_complex_A(self) -> None:
        """Complex A is handled; A_bar matches scipy.linalg.expm(dt*A)."""
        A = torch.tensor(
            [[-1.0 + 2.0j, 0.0], [0.0, -0.5 - 1.0j]], dtype=torch.complex64
        )
        B = torch.ones(2, 1, dtype=torch.complex64)
        dt = 0.25
        A_bar, B_bar = zoh(A, B, dt)
        ref = scipy.linalg.expm(dt * A.numpy())
        torch.testing.assert_close(
            A_bar, torch.from_numpy(ref).to(torch.complex64), atol=1e-5, rtol=1e-4
        )
        assert B_bar.shape == (2, 1)

    def test_batched_delta(self) -> None:
        """A per-batch step size produces one discretization per delta."""
        A = _random_stable_A(3).to(torch.float32)
        B = torch.randn(3, 1)
        deltas = torch.tensor([0.05, 0.1, 0.2, 0.4])
        A_bar, B_bar = zoh(A, B, deltas)
        assert A_bar.shape == (4, 3, 3)
        assert B_bar.shape == (4, 3, 1)
        for k, dt in enumerate(deltas.tolist()):
            a_k, b_k = zoh(A, B, dt)
            torch.testing.assert_close(A_bar[k], a_k, atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(B_bar[k], b_k, atol=1e-5, rtol=1e-4)


class TestBilinear:
    def test_matches_zoh_for_small_delta(self) -> None:
        """Bilinear and ZOH agree to first order for small steps."""
        A = _random_stable_A(4).to(torch.float32)
        B = torch.randn(4, 1)
        dt = 1e-3
        a_bil, b_bil = bilinear(A, B, dt)
        a_zoh, b_zoh = zoh(A, B, dt)
        torch.testing.assert_close(a_bil, a_zoh, atol=1e-5, rtol=1e-3)
        torch.testing.assert_close(b_bil, b_zoh, atol=1e-5, rtol=1e-3)

    def test_frequency_warping_property(self) -> None:
        """Bilinear maps stable continuous systems to stable discrete ones."""
        A = _random_stable_A(5)
        B = torch.randn(5, 1, dtype=torch.float64)
        A_bar, _ = bilinear(A, B, 2.0)
        eig = torch.linalg.eigvals(A_bar)
        assert torch.all(eig.abs() <= 1.0 + 1e-6)

    def test_algebraic_inverse_relationship(self) -> None:
        r"""Verify :math:`(I - \tfrac{\Delta}{2}A)\bar A = I + \tfrac{\Delta}{2}A`."""
        A = _random_stable_A(4)
        B = torch.randn(4, 1, dtype=torch.float64)
        dt = 0.7
        A_bar, _ = bilinear(A, B, dt)
        eye = torch.eye(4, dtype=torch.float64)
        lhs = (eye - dt / 2 * A) @ A_bar
        rhs = eye + dt / 2 * A
        torch.testing.assert_close(lhs, rhs, atol=1e-9, rtol=1e-7)


class TestEuler:
    def test_closed_form(self) -> None:
        """Euler is exactly A_bar = I + dt A, B_bar = dt B."""
        A = torch.randn(4, 4)
        B = torch.randn(4, 2)
        dt = 0.1
        A_bar, B_bar = euler(A, B, dt)
        torch.testing.assert_close(A_bar, torch.eye(4) + dt * A)
        torch.testing.assert_close(B_bar, dt * B)


class TestSelectiveZOH:
    def test_reduces_to_zoh_when_delta_uniform(self) -> None:
        """With a uniform step and one channel, selective ZOH == plain ZOH."""
        n = 6
        A_diag = -(torch.arange(1, n + 1, dtype=torch.float32))  # (N,)
        A_chan = A_diag.unsqueeze(0)  # (D=1, N)
        B = torch.randn(1, 3, n)  # (batch=1, L=3, N)
        dt = 0.123
        delta = torch.full((1, 3, 1), dt)  # uniform across batch/time/channel
        A_bar, B_bar = selective_zoh(A_chan, B, delta)  # (1, 3, 1, N)

        A_mat = torch.diag(A_diag)
        for t in range(3):
            B_col = B[0, t].unsqueeze(-1)  # (N, 1)
            a_ref, b_ref = zoh(A_mat, B_col, dt)
            torch.testing.assert_close(
                A_bar[0, t, 0], torch.diagonal(a_ref), atol=1e-5, rtol=1e-4
            )
            torch.testing.assert_close(
                B_bar[0, t, 0], b_ref.squeeze(-1), atol=1e-5, rtol=1e-4
            )

    def test_shapes_B_L_D_N(self) -> None:
        """Outputs are shaped (batch, L, D, N)."""
        batch, L, D, N = 2, 5, 4, 8
        A = -torch.rand(D, N) - 0.1
        B = torch.randn(batch, L, N)
        delta = torch.rand(batch, L, D) + 0.01
        A_bar, B_bar = selective_zoh(A, B, delta)
        assert A_bar.shape == (batch, L, D, N)
        assert B_bar.shape == (batch, L, D, N)

    def test_gradient_flows_through_delta(self) -> None:
        """Gradients propagate back to delta, A, and B."""
        batch, L, D, N = 1, 4, 2, 3
        A = (-torch.rand(D, N) - 0.1).requires_grad_(True)
        B = torch.randn(batch, L, N, requires_grad=True)
        delta = (torch.rand(batch, L, D) + 0.01).requires_grad_(True)
        A_bar, B_bar = selective_zoh(A, B, delta)
        (A_bar.sum() + B_bar.sum()).backward()
        assert delta.grad is not None and torch.isfinite(delta.grad).all()
        assert A.grad is not None and torch.isfinite(A.grad).all()
        assert B.grad is not None and torch.isfinite(B.grad).all()
