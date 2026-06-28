"""Unit tests for :class:`mamba.core.selective_ssm.SelectiveSSM`.

Verify the S6 contract: correct shapes, finite outputs, gradient flow to every
learnable parameter, the positivity of :math:`\\Delta`, the negativity (hence
stability) of ``A``, causality, and -- most importantly -- that step-by-step
recurrent decoding reproduces the parallel scan.
"""

from __future__ import annotations

import torch

from mamba.core.selective_ssm import SelectiveSSM


def _ssm(d_inner: int = 16, d_state: int = 8) -> SelectiveSSM:
    torch.manual_seed(0)
    return SelectiveSSM(d_inner=d_inner, d_state=d_state)


class TestSelectiveSSM:
    def test_forward_output_shape(self) -> None:
        """Output preserves the (batch, L, d_inner) shape."""
        ssm = _ssm()
        u = torch.randn(2, 16, 16)
        assert ssm(u).shape == (2, 16, 16)

    def test_no_nan_no_inf_forward(self) -> None:
        """Forward output is everywhere finite."""
        ssm = _ssm()
        y = ssm(torch.randn(2, 32, 16))
        assert torch.isfinite(y).all()

    def test_gradient_flow_through_all_params(self) -> None:
        """A_log, D, x_proj and dt_proj all receive gradients."""
        ssm = _ssm()
        ssm(torch.randn(2, 12, 16)).sum().backward()
        for name in ("A_log", "D", "x_proj.weight", "dt_proj.weight", "dt_proj.bias"):
            p = dict(ssm.named_parameters())[name]
            assert p.grad is not None, f"no grad for {name}"
            assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"

    def test_dt_positivity_after_softplus(self) -> None:
        r""":math:`\Delta` is strictly positive (softplus output)."""
        ssm = _ssm()
        delta, _, _ = ssm._project(torch.randn(2, 16, 16))
        assert torch.all(delta > 0)

    def test_A_log_negative_ensures_stability(self) -> None:
        """``A = -exp(A_log)`` is strictly negative -> stable poles."""
        ssm = _ssm()
        assert torch.all(ssm._A() < 0)

    def test_b_c_shapes(self) -> None:
        """B and C are (batch, L, N), not (batch, L, d_inner, N)."""
        ssm = _ssm(d_inner=16, d_state=8)
        _, B, C = ssm._project(torch.randn(2, 10, 16))
        assert B.shape == (2, 10, 8)
        assert C.shape == (2, 10, 8)

    def test_parallel_vs_recurrent_equivalence(self) -> None:
        """Parallel scan equals step-by-step recurrence from a zero state."""
        ssm = _ssm(d_inner=16, d_state=8).eval()
        u = torch.randn(2, 24, 16)
        y_par = ssm(u)
        h = torch.zeros(2, 16, 8)
        outs = []
        for t in range(u.shape[1]):
            y_t, h = ssm.step(u[:, t], h)
            outs.append(y_t)
        y_rec = torch.stack(outs, dim=1)
        torch.testing.assert_close(y_par, y_rec, atol=1e-4, rtol=1e-3)

    def test_training_inference_mode_switch(self) -> None:
        """Fast-path and reference-path scans agree (mode switch is lossless)."""
        torch.manual_seed(0)
        fast = SelectiveSSM(d_inner=16, d_state=8, use_fast_path=True)
        slow = SelectiveSSM(d_inner=16, d_state=8, use_fast_path=False)
        slow.load_state_dict(fast.state_dict())
        u = torch.randn(2, 20, 16)
        torch.testing.assert_close(fast(u), slow(u), atol=1e-4, rtol=1e-3)

    def test_causal_masking(self) -> None:
        """Outputs up to position t0 do not depend on inputs after t0."""
        ssm = _ssm().eval()
        u = torch.randn(1, 16, 16)
        t0 = 8
        y = ssm(u)
        u2 = u.clone()
        u2[:, t0:] += 3.0
        y2 = ssm(u2)
        torch.testing.assert_close(y[:, :t0], y2[:, :t0], atol=1e-5, rtol=1e-4)

    def test_different_sequence_lengths(self, seqlen_cases: int) -> None:
        """Forward works across a spread of sequence lengths."""
        ssm = _ssm()
        u = torch.randn(2, seqlen_cases, 16)
        assert ssm(u).shape == (2, seqlen_cases, 16)

    def test_return_last_state(self) -> None:
        """The optional last-state return matches the manual recurrence."""
        ssm = _ssm(d_inner=8, d_state=4).eval()
        u = torch.randn(2, 10, 8)
        _, last = ssm(u, return_last_state=True)
        h = torch.zeros(2, 8, 4)
        for t in range(u.shape[1]):
            _, h = ssm.step(u[:, t], h)
        torch.testing.assert_close(last, h, atol=1e-4, rtol=1e-3)
