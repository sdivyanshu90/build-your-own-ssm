"""Property-based tests for SSM invariants that must hold for *all* parameters.

Uses Hypothesis to sample dimensions, step sizes, and shapes, checking the
mathematical guarantees that underpin the architecture: stability under
discretization, linearity of a fixed (non-selective) SSM in its input, the
recurrent/convolutional duality of an LTI model, and the monotone improvement of
HiPPO-LegS reconstruction with state size.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from mamba.core.discretize import zoh
from mamba.core.hippo import hippo_legs
from mamba.core.ssm_base import ContinuousSSM
from mamba.ops.selective_scan_naive import selective_scan_naive

_SETTINGS = settings(deadline=None, max_examples=25)


@_SETTINGS
@given(
    n=st.integers(min_value=2, max_value=24),
    rank=st.integers(min_value=1, max_value=4),
    delta=st.floats(min_value=1e-3, max_value=0.1),
)
def test_stability_preserved_after_discretization(
    n: int, rank: int, delta: float
) -> None:
    """A continuous-stable HiPPO matrix discretizes to a discrete-stable one."""
    del rank  # HiPPO-LegS is rank-1; included to match the documented signature.
    A, _ = hippo_legs(n)
    B = torch.ones(n, 1)
    A_bar, _ = zoh(A, B, delta)
    eig = torch.linalg.eigvals(A_bar)
    assert torch.all(eig.abs() <= 1.0 + 1e-4)


@_SETTINGS
@given(
    batch=st.integers(min_value=1, max_value=4),
    seqlen=st.integers(min_value=2, max_value=32),
)
def test_ssm_linearity(batch: int, seqlen: int) -> None:
    """With fixed (input-independent) delta, B, C, the scan is linear in u."""
    torch.manual_seed(seqlen * 100 + batch)
    d_inner, d_state = 3, 2
    A = -torch.rand(d_inner, d_state) - 0.1
    delta = torch.rand(batch, seqlen, d_inner) + 0.05
    B = torch.randn(batch, seqlen, d_state)
    C = torch.randn(batch, seqlen, d_state)
    u1 = torch.randn(batch, seqlen, d_inner)
    u2 = torch.randn(batch, seqlen, d_inner)
    a, b = 1.7, -0.4

    def run(u: torch.Tensor) -> torch.Tensor:
        return selective_scan_naive(u, delta, A, B, C, D=None, delta_softplus=False)

    lhs = run(a * u1 + b * u2)
    rhs = a * run(u1) + b * run(u2)
    torch.testing.assert_close(lhs, rhs, atol=1e-4, rtol=1e-3)


@_SETTINGS
@given(seqlen=st.integers(min_value=4, max_value=128))
def test_convolutional_recurrent_equivalence(seqlen: int) -> None:
    """An LTI SSM's convolutional and recurrent outputs coincide."""
    torch.manual_seed(seqlen)
    ssm = ContinuousSSM(d_state=16, dt=0.01)
    u = torch.randn(2, seqlen)
    torch.testing.assert_close(
        ssm.forward_conv(u), ssm.forward_recurrent(u), atol=1e-4, rtol=1e-3
    )


def test_hippo_legs_approximation_improves_with_N(
    legs_recon_mse: Callable[..., float],
) -> None:
    """Larger HiPPO-LegS state dimension reconstructs smooth signals better."""
    t = np.linspace(0.0, 1.0, 2000)
    signal = (
        np.sin(6 * np.pi * t)
        + 0.4 * np.cos(13 * np.pi * t)
        + 0.2 * np.sin(20 * np.pi * t)
    )
    mse_small = legs_recon_mse(signal, 16)
    mse_large = legs_recon_mse(signal, 64)
    assert (
        mse_large < mse_small
    ), f"N=64 ({mse_large}) not better than N=16 ({mse_small})"
