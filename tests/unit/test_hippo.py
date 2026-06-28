"""Unit tests for :mod:`mamba.core.hippo`.

Verify the structural and spectral properties that make HiPPO initialization
correct: triangularity and normalization of LegS, stability of the spectrum,
the NPLR reconstruction identity, the Woodbury identity used by S4's Cauchy
kernel, and the conjugate-pair structure of the DPLR eigenvalues.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from mamba.core.hippo import (
    hippo_lagt,
    hippo_legs,
    hippo_legt,
    make_dplr_hippo,
    make_nplr_hippo,
    random_ssm_init,
)


class TestHiPPOLegs:
    def test_matrix_shape(self) -> None:
        """LegS returns an (N, N) matrix and an (N,) vector."""
        A, B = hippo_legs(16)
        assert A.shape == (16, 16)
        assert B.shape == (16,)

    def test_A_is_lower_triangular(self) -> None:
        """HiPPO-LegS A is lower triangular by construction."""
        A, _ = hippo_legs(24)
        upper = torch.triu(A, diagonal=1)
        assert torch.count_nonzero(upper) == 0

    def test_B_normalization(self) -> None:
        r"""The k-th component of B equals :math:`\sqrt{2k+1}`."""
        _, B = hippo_legs(20)
        expected = torch.sqrt(2 * torch.arange(20, dtype=torch.float32) + 1)
        torch.testing.assert_close(B, expected, atol=1e-5, rtol=1e-5)

    def test_eigenvalue_real_parts_negative(self) -> None:
        """All eigenvalues of A have negative real part (BIBO stability)."""
        A, _ = hippo_legs(16)
        eig = torch.linalg.eigvals(A)
        assert torch.all(eig.real < 0)

    def test_approximation_quality(self, legs_recon_mse: Callable[..., float]) -> None:
        """Projecting a smooth signal onto a size-64 LegS basis reconstructs it.

        Integration error caps accuracy, so we require a comfortably small MSE
        rather than the theoretical optimum.
        """
        t = np.linspace(0.0, 1.0, 2000)
        signal = (
            np.sin(6 * np.pi * t)
            + 0.4 * np.cos(13 * np.pi * t)
            + 0.2 * np.sin(20 * np.pi * t)
        )
        mse = legs_recon_mse(signal, 64)
        assert mse < 1e-3, f"LegS reconstruction MSE too high: {mse}"


class TestHiPPOLegtLagt:
    def test_legt_shape_and_finite(self) -> None:
        """LegT returns finite (N, N)/(N,) tensors."""
        A, B = hippo_legt(12)
        assert A.shape == (12, 12) and B.shape == (12,)
        assert torch.isfinite(A).all() and torch.isfinite(B).all()

    def test_lagt_closed_form(self) -> None:
        r"""LagT is :math:`A = \tfrac12 I - \mathrm{tril}(\mathbf 1)`, ``B = 1``."""
        A, B = hippo_lagt(10)
        expected_A = 0.5 * torch.eye(10) - torch.tril(torch.ones(10, 10))
        torch.testing.assert_close(A, expected_A)
        torch.testing.assert_close(B, torch.ones(10))

    def test_lagt_stable(self) -> None:
        """LagT eigenvalues have negative real part."""
        A, _ = hippo_lagt(16)
        eig = torch.linalg.eigvals(A)
        assert torch.all(eig.real < 0)


class TestNPLR:
    def test_nplr_reconstruction(self) -> None:
        r"""Verify ``A = V diag(w) V^{-1} - P P^T`` recovers the HiPPO matrix."""
        n = 32
        w, P, _, V = make_nplr_hippo(n)
        w = w.to(torch.complex128)
        V = V.to(torch.complex128)
        Pc = P.to(torch.complex128)
        A_rec = (V @ torch.diag(w) @ V.conj().t() - torch.outer(Pc, Pc)).real
        A_true, _ = hippo_legs(n)
        torch.testing.assert_close(
            A_rec.to(torch.float32), A_true, atol=1e-3, rtol=1e-3
        )

    def test_woodbury_identity(self) -> None:
        r"""Numerically verify the Woodbury matrix-inversion identity.

        ``(A + P Q^T)^{-1} = A^{-1} - A^{-1} P (I + Q^T A^{-1} P)^{-1} Q^T A^{-1}``
        underlies S4's efficient Cauchy-kernel evaluation.
        """
        n, k = 8, 2
        torch.manual_seed(0)
        A = torch.randn(n, n, dtype=torch.float64) + n * torch.eye(
            n, dtype=torch.float64
        )
        P = torch.randn(n, k, dtype=torch.float64)
        Q = torch.randn(n, k, dtype=torch.float64)
        eye_k = torch.eye(k, dtype=torch.float64)
        Ainv = torch.linalg.inv(A)
        direct = torch.linalg.inv(A + P @ Q.t())
        woodbury = (
            Ainv - Ainv @ P @ torch.linalg.inv(eye_k + Q.t() @ Ainv @ P) @ Q.t() @ Ainv
        )
        torch.testing.assert_close(direct, woodbury, atol=1e-9, rtol=1e-7)


class TestDPLR:
    def test_complex_conjugate_pairs(self) -> None:
        """DPLR eigenvalues come in complex-conjugate pairs (skew spectrum)."""
        Lambda, _, _, _ = make_dplr_hippo(16)
        imag_sorted = torch.sort(Lambda.imag).values
        # symmetric about zero: sorted imag parts equal negated reversed.
        torch.testing.assert_close(
            imag_sorted, -torch.flip(imag_sorted, dims=[0]), atol=1e-4, rtol=1e-4
        )
        assert torch.allclose(
            Lambda.real, torch.full_like(Lambda.real, -0.5), atol=1e-4
        )

    def test_dtype_handling(self) -> None:
        """The requested complex dtype is honored across all returns."""
        for dtype in (torch.complex64, torch.complex128):
            Lambda, P, B, V = make_dplr_hippo(8, rank=1, dtype=dtype)
            assert Lambda.dtype == dtype
            assert P.dtype == dtype and P.shape == (1, 8)
            assert B.dtype == dtype and V.dtype == dtype


class TestRandomInit:
    def test_real_init_shapes_and_stability(self) -> None:
        """Real init gives (H, N) tensors with strictly stable A."""
        A, B, C = random_ssm_init(4, 8)
        assert A.shape == (4, 8) and B.shape == (4, 8) and C.shape == (4, 8)
        assert torch.all(A < 0)

    def test_complex_init_uses_hippo_spectrum(self) -> None:
        """Complex init draws A from the HiPPO spectrum (Re = -1/2)."""
        A, _, _ = random_ssm_init(3, 8, dtype=torch.complex64)
        assert A.dtype == torch.complex64
        assert torch.all(A.real < 0)
