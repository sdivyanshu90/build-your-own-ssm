"""Unit tests for :class:`mamba.ops.causal_conv1d.CausalConv1d`.

Confirm shape preservation, strict causality (no dependence on the future), and
that the rolling-buffer single-step path reproduces the full convolution.
"""

from __future__ import annotations

import torch

from mamba.ops.causal_conv1d import CausalConv1d


class TestCausalConv1d:
    def test_output_shape(self) -> None:
        """Output shape equals input shape ``(batch, d_inner, L)``."""
        conv = CausalConv1d(d_inner=8, d_conv=4)
        x = torch.randn(2, 8, 16)
        assert conv(x).shape == (2, 8, 16)

    def test_no_future_leakage(self) -> None:
        """Perturbing input at position ``t`` leaves outputs ``< t`` unchanged."""
        torch.manual_seed(0)
        conv = CausalConv1d(d_inner=4, d_conv=3)
        x = torch.randn(1, 4, 12)
        y = conv(x)
        x2 = x.clone()
        x2[:, :, 6:] += 5.0  # change the future only
        y2 = conv(x2)
        torch.testing.assert_close(y[:, :, :6], y2[:, :, :6])

    def test_step_matches_forward(self) -> None:
        """Rolling-buffer ``step`` reproduces the batched ``forward``."""
        torch.manual_seed(1)
        d_inner, d_conv, length = 6, 4, 10
        conv = CausalConv1d(d_inner=d_inner, d_conv=d_conv)
        x = torch.randn(2, d_inner, length)
        y_full = conv(x)

        conv_state = torch.zeros(2, d_inner, d_conv)
        outs = []
        for t in range(length):
            out_t, conv_state = conv.step(x[:, :, t], conv_state)
            outs.append(out_t)
        y_step = torch.stack(outs, dim=-1)
        torch.testing.assert_close(y_step, y_full, atol=1e-5, rtol=1e-4)

    def test_left_padding_only(self) -> None:
        """The first output depends only on the first input (left pad works)."""
        conv = CausalConv1d(d_inner=2, d_conv=4, bias=False)
        x = torch.zeros(1, 2, 5)
        x[:, :, 0] = 1.0
        y = conv(x)
        # out[0] uses only x[0]; equals the last kernel tap times x[0].
        expected0 = conv.conv.weight.squeeze(1)[:, -1]
        torch.testing.assert_close(y[0, :, 0], expected0, atol=1e-6, rtol=1e-5)

    def test_allocate_inference_cache(self) -> None:
        """The allocated buffer has shape ``(batch, d_inner, d_conv)`` of zeros."""
        conv = CausalConv1d(d_inner=5, d_conv=3)
        cache = conv.allocate_inference_cache(batch_size=4, dtype=torch.float32)
        assert cache.shape == (4, 5, 3)
        assert torch.count_nonzero(cache) == 0
