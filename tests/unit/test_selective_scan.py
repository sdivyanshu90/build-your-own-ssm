"""Unit tests for the selective scan operators -- the correctness keystone.

The parallel associative scan only earns the right to replace the sequential
reference if it reproduces it *exactly* (to numerical tolerance) across lengths,
options, and gradients. These tests pin that down, plus the algebraic
properties (associativity, identity) the parallel scan relies on.
"""

from __future__ import annotations

import pytest
import torch

from mamba.ops.selective_scan_naive import selective_scan_naive
from mamba.ops.selective_scan_parallel import (
    _make_scan_op,
    _parallel_prefix_scan,
    selective_scan_parallel,
)


def _make_inputs(
    batch: int = 2,
    length: int = 16,
    d_inner: int = 4,
    d_state: int = 3,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    requires_grad: bool = False,
) -> dict[str, torch.Tensor]:
    """Build a consistent set of random selective-scan inputs."""
    g = torch.Generator().manual_seed(seed)

    def r(*shape: int) -> torch.Tensor:
        t = torch.randn(*shape, generator=g, dtype=dtype)
        return t.requires_grad_(requires_grad)

    u = r(batch, length, d_inner)
    delta = r(batch, length, d_inner)
    # A must be negative for stability.
    A = (-torch.rand(d_inner, d_state, generator=g, dtype=dtype) - 0.1).requires_grad_(
        requires_grad
    )
    B = r(batch, length, d_state)
    C = r(batch, length, d_state)
    D = r(d_inner)
    return {"u": u, "delta": delta, "A": A, "B": B, "C": C, "D": D}


class TestNaiveScan:
    def test_shapes(self) -> None:
        """Output has the same (batch, L, d_inner) shape as the input."""
        ins = _make_inputs()
        y = selective_scan_naive(**ins, delta_softplus=True)
        assert y.shape == ins["u"].shape

    def test_all_zeros_input(self) -> None:
        """Zero input yields zero output (no state, no skip contribution)."""
        ins = _make_inputs()
        ins["u"] = torch.zeros_like(ins["u"])
        y = selective_scan_naive(**ins, delta_softplus=True)
        torch.testing.assert_close(y, torch.zeros_like(y))

    def test_zero_A_is_integrator(self) -> None:
        r"""With continuous ``A = 0`` the discrete :math:`\bar A = 1`: the state
        becomes a running sum (an accumulator), so the C-projected output equals
        the cumulative sum of the per-step additive terms.
        """
        ins = _make_inputs(d_state=1, d_inner=2)
        ins["A"] = torch.zeros_like(ins["A"])
        ins["D"] = None  # isolate the SSM term
        y, last = selective_scan_naive(
            **ins, delta_softplus=True, return_last_state=True
        )
        # last state must equal the sum over time of the additive terms.
        from mamba.ops._scan_common import prepare_scan_inputs

        _, deltaB_u, _ = prepare_scan_inputs(
            ins["u"], ins["delta"], ins["A"], ins["B"], ins["C"], None, None, True
        )
        torch.testing.assert_close(last, deltaB_u.sum(dim=1), atol=1e-5, rtol=1e-4)

    def test_strongly_negative_A_is_memoryless(self) -> None:
        r"""As :math:`A \to -\infty`, :math:`\bar A \to 0`: the state forgets the
        past and ``h_t`` depends only on the current input.
        """
        ins = _make_inputs(d_state=2, d_inner=2)
        ins["A"] = torch.full_like(ins["A"], -1e4)
        ins["delta"] = torch.ones_like(ins["delta"])  # ensure Δ·A very negative
        y, last = selective_scan_naive(
            **ins, delta_softplus=False, return_last_state=True
        )
        from mamba.ops._scan_common import prepare_scan_inputs

        _, deltaB_u, _ = prepare_scan_inputs(
            ins["u"], ins["delta"], ins["A"], ins["B"], ins["C"], None, None, False
        )
        # last state ~ additive term at the final step only.
        torch.testing.assert_close(last, deltaB_u[:, -1], atol=1e-4, rtol=1e-3)

    def test_gradient_check(self) -> None:
        """``torch.autograd.gradcheck`` passes for the naive scan (float64)."""
        ins = _make_inputs(
            batch=1,
            length=4,
            d_inner=2,
            d_state=2,
            dtype=torch.float64,
            requires_grad=True,
        )

        def fn(u, delta, A, B, C, D):  # type: ignore[no-untyped-def]
            return selective_scan_naive(u, delta, A, B, C, D, delta_softplus=True)

        assert torch.autograd.gradcheck(
            fn,
            (ins["u"], ins["delta"], ins["A"], ins["B"], ins["C"], ins["D"]),
            atol=1e-4,
            rtol=1e-3,
        )


class TestParallelScan:
    def test_shapes(self) -> None:
        """Parallel scan preserves the input shape."""
        ins = _make_inputs()
        y = selective_scan_parallel(**ins, delta_softplus=True)
        assert y.shape == ins["u"].shape

    def test_matches_naive_random_input(self) -> None:
        """Critical: parallel matches naive on random input to atol=1e-4."""
        ins = _make_inputs(batch=3, length=64, d_inner=8, d_state=5, seed=7)
        y_naive = selective_scan_naive(**ins, delta_softplus=True)
        y_par = selective_scan_parallel(**ins, delta_softplus=True)
        torch.testing.assert_close(y_par, y_naive, atol=1e-4, rtol=1e-3)

    def test_matches_naive_length_1(self) -> None:
        """Single-timestep sequences agree."""
        ins = _make_inputs(length=1)
        torch.testing.assert_close(
            selective_scan_parallel(**ins, delta_softplus=True),
            selective_scan_naive(**ins, delta_softplus=True),
            atol=1e-4,
            rtol=1e-3,
        )

    @pytest.mark.parametrize("length", [2, 4, 8, 16, 64, 256])
    def test_matches_naive_length_power_of_2(self, length: int) -> None:
        """Agreement holds at power-of-two lengths (the easy case for scans)."""
        ins = _make_inputs(length=length, seed=length)
        torch.testing.assert_close(
            selective_scan_parallel(**ins, delta_softplus=True),
            selective_scan_naive(**ins, delta_softplus=True),
            atol=1e-4,
            rtol=1e-3,
        )

    @pytest.mark.parametrize("length", [3, 7, 17, 100, 1000])
    def test_matches_naive_length_non_power_of_2(self, length: int) -> None:
        """Agreement holds at non-power-of-two lengths (the tricky case)."""
        ins = _make_inputs(length=length, seed=length)
        torch.testing.assert_close(
            selective_scan_parallel(**ins, delta_softplus=True),
            selective_scan_naive(**ins, delta_softplus=True),
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradient_check(self) -> None:
        """``gradcheck`` passes for the parallel scan (float64)."""
        ins = _make_inputs(
            batch=1,
            length=5,
            d_inner=2,
            d_state=2,
            dtype=torch.float64,
            requires_grad=True,
        )

        def fn(u, delta, A, B, C, D):  # type: ignore[no-untyped-def]
            return selective_scan_parallel(u, delta, A, B, C, D, delta_softplus=True)

        assert torch.autograd.gradcheck(
            fn,
            (ins["u"], ins["delta"], ins["A"], ins["B"], ins["C"], ins["D"]),
            atol=1e-4,
            rtol=1e-3,
        )

    def test_gradients_match_naive(self) -> None:
        """Gradients from both scans agree, not just the forward outputs."""
        ins = _make_inputs(
            batch=2, length=32, d_inner=4, d_state=3, dtype=torch.float64
        )
        grads = {}
        for name, fn in (
            ("naive", selective_scan_naive),
            ("par", selective_scan_parallel),
        ):
            local = {k: v.clone().requires_grad_(True) for k, v in ins.items()}
            y = fn(**local, delta_softplus=True)
            y.sum().backward()
            grads[name] = {k: v.grad.clone() for k, v in local.items()}
        for k in ins:
            torch.testing.assert_close(
                grads["par"][k], grads["naive"][k], atol=1e-6, rtol=1e-5
            )

    def test_delta_softplus_option(self) -> None:
        """The softplus option is applied identically by both scans."""
        ins = _make_inputs(length=20)
        torch.testing.assert_close(
            selective_scan_parallel(**ins, delta_softplus=True),
            selective_scan_naive(**ins, delta_softplus=True),
            atol=1e-4,
            rtol=1e-3,
        )

    def test_delta_bias_option(self) -> None:
        """A delta bias is honored identically by both scans."""
        ins = _make_inputs(length=20, d_inner=4)
        bias = torch.randn(4)
        torch.testing.assert_close(
            selective_scan_parallel(**ins, delta_bias=bias, delta_softplus=True),
            selective_scan_naive(**ins, delta_bias=bias, delta_softplus=True),
            atol=1e-4,
            rtol=1e-3,
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_memory_usage_linear_in_L(self) -> None:
        """Peak memory grows roughly linearly (not quadratically) with L."""

        def peak_for(length: int) -> int:
            ins = _make_inputs(
                batch=1, length=length, d_inner=16, d_state=8, dtype=torch.float32
            )
            ins = {k: v.cuda() for k, v in ins.items()}
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            selective_scan_parallel(**ins, delta_softplus=True)
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated()

        m1 = peak_for(512)
        m2 = peak_for(2048)  # 4x the length
        # Linear -> ~4x; reject quadratic (~16x). Allow generous slack.
        assert m2 > m1
        assert m2 < 8 * m1, f"memory grew {m2 / m1:.1f}x for 4x length (super-linear)"


class TestScanAssociativity:
    """Property tests for the affine-map operator underlying the parallel scan."""

    def test_operator_is_associative(self) -> None:
        r"""``(p1 ⊕ p2) ⊕ p3 == p1 ⊕ (p2 ⊕ p3)`` for random affine maps."""
        op = _make_scan_op()
        torch.manual_seed(0)
        shape = (4, 5)
        p1 = (torch.randn(shape), torch.randn(shape))
        p2 = (torch.randn(shape), torch.randn(shape))
        p3 = (torch.randn(shape), torch.randn(shape))
        left = op(op(p1, p2), p3)
        right = op(p1, op(p2, p3))
        torch.testing.assert_close(left[0], right[0], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(left[1], right[1], atol=1e-6, rtol=1e-5)

    def test_identity_element(self) -> None:
        """``(1, 0)`` is a two-sided identity for the operator."""
        op = _make_scan_op()
        torch.manual_seed(1)
        shape = (3, 3)
        a, b = torch.randn(shape), torch.randn(shape)
        ident = (torch.ones(shape), torch.zeros(shape))
        for combined in (op(ident, (a, b)), op((a, b), ident)):
            torch.testing.assert_close(combined[0], a)
            torch.testing.assert_close(combined[1], b)

    def test_prefix_scan_matches_python_reduction(self) -> None:
        """The vectorized scan equals a literal left-fold with the operator."""
        op = _make_scan_op()
        torch.manual_seed(2)
        batch, length, dim = 2, 11, 4
        a = torch.rand(batch, length, dim) * 0.9
        b = torch.randn(batch, length, dim)
        _, states = _parallel_prefix_scan(a, b)
        # Reference: sequential fold.
        ref = []
        acc = (torch.ones(batch, dim), torch.zeros(batch, dim))
        for t in range(length):
            acc = op(acc, (a[:, t], b[:, t]))
            ref.append(acc[1])
        ref_states = torch.stack(ref, dim=1)
        torch.testing.assert_close(states, ref_states, atol=1e-5, rtol=1e-4)
