"""Property-based equivalence of the naive and parallel selective scans.

Hypothesis randomises batch size, sequence length (including non power-of-two),
channel count, and state size, asserting the two implementations agree on both
the forward output and the gradients for every drawn configuration.
"""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from mamba.ops.selective_scan_naive import selective_scan_naive
from mamba.ops.selective_scan_parallel import selective_scan_parallel


def _inputs(
    batch: int, length: int, d_inner: int, d_state: int, seed: int
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "u": torch.randn(batch, length, d_inner, generator=g, dtype=torch.float64),
        "delta": torch.randn(batch, length, d_inner, generator=g, dtype=torch.float64),
        "A": -torch.rand(d_inner, d_state, generator=g, dtype=torch.float64) - 0.1,
        "B": torch.randn(batch, length, d_state, generator=g, dtype=torch.float64),
        "C": torch.randn(batch, length, d_state, generator=g, dtype=torch.float64),
        "D": torch.randn(d_inner, generator=g, dtype=torch.float64),
    }


@settings(deadline=None, max_examples=40)
@given(
    batch=st.integers(min_value=1, max_value=3),
    length=st.integers(min_value=1, max_value=130),
    d_inner=st.integers(min_value=1, max_value=6),
    d_state=st.integers(min_value=1, max_value=6),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_naive_parallel_forward_equivalence(
    batch: int, length: int, d_inner: int, d_state: int, seed: int
) -> None:
    """Forward outputs of both scans agree for any drawn shape."""
    ins = _inputs(batch, length, d_inner, d_state, seed)
    y_naive = selective_scan_naive(**ins, delta_softplus=True)
    y_par = selective_scan_parallel(**ins, delta_softplus=True)
    torch.testing.assert_close(y_par, y_naive, atol=1e-8, rtol=1e-6)


@settings(deadline=None, max_examples=20)
@given(
    batch=st.integers(min_value=1, max_value=2),
    length=st.integers(min_value=1, max_value=40),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_naive_parallel_gradient_equivalence(
    batch: int, length: int, seed: int
) -> None:
    """Gradients from both scans agree for any drawn shape."""
    base = _inputs(batch, length, 3, 3, seed)
    grads = {}
    for name, fn in (("naive", selective_scan_naive), ("par", selective_scan_parallel)):
        local = {k: v.clone().requires_grad_(True) for k, v in base.items()}
        fn(**local, delta_softplus=True).sum().backward()
        grads[name] = {k: v.grad for k, v in local.items()}
    for key in base:
        torch.testing.assert_close(
            grads["par"][key], grads["naive"][key], atol=1e-7, rtol=1e-5
        )
