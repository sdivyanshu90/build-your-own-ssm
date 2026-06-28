"""Unit tests for :class:`mamba.layers.mamba_block.MambaBlock` and the
pre-norm :class:`mamba.layers.residual.ResidualBlock`.

The decisive test here is :meth:`test_inference_step_matches_forward`: the
incremental decoder must reproduce the parallel forward pass exactly, otherwise
recurrent generation would diverge from training-time scoring.
"""

from __future__ import annotations

import pytest
import torch

from mamba.config import MambaConfig
from mamba.layers.mamba_block import MambaBlock
from mamba.layers.residual import ResidualBlock


@pytest.fixture
def cfg() -> MambaConfig:
    return MambaConfig(d_model=32, n_layers=2, d_state=8, vocab_size=64)


class TestMambaBlock:
    def test_output_shape_preserved(self, cfg: MambaConfig) -> None:
        """Input and output shapes match (batch, L, d_model)."""
        block = MambaBlock(cfg)
        x = torch.randn(2, 16, cfg.d_model)
        assert block(x).shape == (2, 16, cfg.d_model)

    def test_causal_property(self, cfg: MambaConfig) -> None:
        """Block output up to t0 is independent of inputs after t0."""
        block = MambaBlock(cfg).eval()
        x = torch.randn(1, 16, cfg.d_model)
        t0 = 8
        y = block(x)
        x2 = x.clone()
        x2[:, t0:] += 2.0
        y2 = block(x2)
        torch.testing.assert_close(y[:, :t0], y2[:, :t0], atol=1e-5, rtol=1e-4)

    def test_inference_step_matches_forward(self, cfg: MambaConfig) -> None:
        """T calls to step() reproduce a single forward() over T tokens."""
        block = MambaBlock(cfg).eval()
        length = 20
        x = torch.randn(2, length, cfg.d_model)
        y_full = block(x)

        conv_state, ssm_state = block.allocate_inference_cache(2, length)
        outs = []
        for t in range(length):
            out_t, conv_state, ssm_state = block.step(x[:, t], conv_state, ssm_state)
            outs.append(out_t)
        y_step = torch.stack(outs, dim=1)
        torch.testing.assert_close(y_step, y_full, atol=1e-4, rtol=1e-3)

    def test_inference_cache_allocation(self, cfg: MambaConfig) -> None:
        """Allocated caches have the expected conv/SSM shapes."""
        block = MambaBlock(cfg)
        conv_state, ssm_state = block.allocate_inference_cache(4, 10)
        assert conv_state.shape == (4, cfg.d_inner, cfg.d_conv)
        assert ssm_state.shape == (4, cfg.d_inner, cfg.d_state)

    def test_parameter_count(self, cfg: MambaConfig) -> None:
        """Total parameters equal the analytic sum of every sub-module."""
        block = MambaBlock(cfg)
        D, E, K, N = cfg.d_model, cfg.expand, cfg.d_conv, cfg.d_state
        ED = E * D
        R = cfg.dt_rank_int
        expected = (
            2 * ED * D  # in_proj weight (no bias)
            + ED * K
            + ED  # conv weight + conv bias
            + ED * (R + 2 * N)  # x_proj weight (no bias)
            + R * ED
            + ED  # dt_proj weight + bias
            + ED * N  # A_log
            + ED  # D skip
            + ED * D  # out_proj weight (no bias)
        )
        actual = sum(p.numel() for p in block.parameters())
        assert actual == expected, f"expected {expected}, got {actual}"

    def test_no_weight_matrix_for_time(self, cfg: MambaConfig) -> None:
        """Mamba has no positional/time embedding parameters."""
        block = MambaBlock(cfg)
        names = [n.lower() for n, _ in block.named_parameters()]
        assert not any("pos" in n or "time_embed" in n for n in names)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_low_precision_forward(self, cfg: MambaConfig, dtype: torch.dtype) -> None:
        """Forward is finite in float16 and bfloat16."""
        block = MambaBlock(cfg).to(dtype)
        x = torch.randn(2, 16, cfg.d_model, dtype=dtype)
        y = block(x)
        assert y.dtype == dtype
        assert torch.isfinite(y).all()

    def test_gradient_checkpointing_compatibility(self, cfg: MambaConfig) -> None:
        """The block runs under torch.utils.checkpoint and backprops."""
        from torch.utils.checkpoint import checkpoint

        block = MambaBlock(cfg)
        x = torch.randn(2, 12, cfg.d_model, requires_grad=True)
        y = checkpoint(block, x, use_reentrant=False)
        y.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


class TestResidualBlock:
    def test_output_shape(self, cfg: MambaConfig) -> None:
        """Residual block preserves the residual-stream shape."""
        block = ResidualBlock(cfg, layer_idx=0)
        x = torch.randn(2, 16, cfg.d_model)
        assert block(x).shape == (2, 16, cfg.d_model)

    def test_residual_connection(self, cfg: MambaConfig) -> None:
        """Zeroing the mixer output projection -> pure residual pass-through."""
        block = ResidualBlock(cfg, layer_idx=0).eval()
        with torch.no_grad():
            block.mixer.out_proj.weight.zero_()
        x = torch.randn(2, 8, cfg.d_model)
        torch.testing.assert_close(block(x), x, atol=1e-6, rtol=1e-5)

    def test_inference_cache_delegation(self, cfg: MambaConfig) -> None:
        """Residual block delegates cache allocation to its mixer."""
        block = ResidualBlock(cfg, layer_idx=0)
        conv_state, ssm_state = block.allocate_inference_cache(2, 10)
        assert conv_state.shape == (2, cfg.d_inner, cfg.d_conv)
        assert ssm_state.shape == (2, cfg.d_inner, cfg.d_state)
