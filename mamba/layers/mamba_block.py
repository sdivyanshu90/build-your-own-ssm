r"""The Mamba block -- the architectural unit repeated to build the model.

Each block expands the residual stream, runs a gated selective SSM on it, and
projects back down::

    input (B, L, D)
        │
      in_proj           Linear(D -> 2*D_inner, no bias)
        │  split
        ├─ x branch:  CausalConv1d -> SiLU -> SelectiveSSM
        └─ z branch:  SiLU                          (the gate)
        │
      x * z             element-wise gating
      out_proj          Linear(D_inner -> D, no bias)
        │
      output (B, L, D)

Normalization and the residual add are *not* here -- they live in
:class:`~mamba.layers.residual.ResidualBlock` (pre-norm). Keeping the block
norm-free makes the training :meth:`forward` and the incremental :meth:`step`
produce identical values, which the test suite verifies.

References
----------
[Gu & Dao, 2023] "Mamba: Linear-Time Sequence Modeling with Selective State
    Spaces", Figure 3 (block diagram).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mamba.config import MambaConfig
from mamba.core.selective_ssm import SelectiveSSM
from mamba.ops.causal_conv1d import CausalConv1d

if TYPE_CHECKING:
    from mamba.utils.generation import InferenceParams

__all__ = ["MambaBlock"]


class MambaBlock(nn.Module):
    """A single gated selective-SSM block.

    Args
    ----
    config:
        The :class:`~mamba.config.MambaConfig` providing all dimensions.
    layer_idx:
        Optional index identifying this block within the stack; used to key its
        entry in an inference cache. Falls back to ``config.layer_idx``.

    Attributes
    ----------
    in_proj : nn.Linear
        ``D -> 2 * d_inner`` (produces the SSM input and the gate).
    conv1d : CausalConv1d
        Depthwise causal convolution over the SSM-input branch.
    ssm : SelectiveSSM
        The S6 selective state space model.
    out_proj : nn.Linear
        ``d_inner -> D``.
    """

    def __init__(self, config: MambaConfig, layer_idx: Optional[int] = None) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx if layer_idx is not None else config.layer_idx
        self.d_model = config.d_model
        self.d_inner = config.d_inner
        self.d_conv = config.d_conv

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=config.bias)
        self.conv1d = CausalConv1d(self.d_inner, self.d_conv, bias=config.conv_bias)
        self.ssm = SelectiveSSM(
            d_inner=self.d_inner,
            d_state=config.d_state,
            dt_rank=config.dt_rank_int,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            dt_init=config.dt_init,
            dt_scale=config.dt_scale,
            dt_init_floor=config.dt_init_floor,
            use_fast_path=config.use_fast_path,
            bias=config.bias,
        )
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=config.bias)

    def forward(
        self,
        hidden_states: Tensor,
        inference_params: "Optional[InferenceParams]" = None,
    ) -> Tensor:
        """Apply the block to a sequence.

        Args
        ----
        hidden_states:
            Input ``(batch, L, d_model)``.
        inference_params:
            Optional incremental-decoding cache. When supplied, a single-token
            call (``L == 1`` with ``seqlen_offset > 0``) routes through
            :meth:`step`; a longer call is treated as a prefill that also seeds
            the cache.

        Returns
        -------
        Tensor
            Output ``(batch, L, d_model)``.
        """
        batch, length, _ = hidden_states.shape

        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(
                inference_params, batch, hidden_states.dtype, hidden_states.device
            )
            if inference_params.seqlen_offset > 0:
                out, conv_state, ssm_state = self.step(
                    hidden_states.squeeze(1), conv_state, ssm_state
                )
                self._store_states(inference_params, conv_state, ssm_state)
                return out.unsqueeze(1)

        xz = self.in_proj(hidden_states)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # (B, L, d_inner) each

        x_t = x.transpose(1, 2)  # (B, d_inner, L) for the conv

        if inference_params is not None:
            # Seed the conv buffer with the trailing d_conv inputs (left-padded).
            if length >= self.d_conv:
                conv_state = x_t[..., -self.d_conv :].clone()
            else:
                conv_state = F.pad(x_t, (self.d_conv - length, 0)).clone()

        x_t = self.conv1d(x_t)  # (B, d_inner, L)
        x = x_t.transpose(1, 2)  # (B, L, d_inner)
        x = F.silu(x)

        if inference_params is not None:
            ssm_out = self.ssm(x, return_last_state=True)
            assert isinstance(ssm_out, tuple)
            y, ssm_state = ssm_out
        else:
            y = self.ssm(x)

        y = y * F.silu(z)
        out = self.out_proj(y)

        if inference_params is not None:
            self._store_states(inference_params, conv_state, ssm_state)
        return out

    def step(
        self, hidden_states: Tensor, conv_state: Tensor, ssm_state: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        r"""Single-token :math:`O(1)` inference step.

        Args
        ----
        hidden_states:
            Current token, shape ``(batch, d_model)``.
        conv_state:
            Causal-conv rolling buffer ``(batch, d_inner, d_conv)``.
        ssm_state:
            SSM hidden state ``(batch, d_inner, d_state)``.

        Returns
        -------
        out : Tensor
            Output token ``(batch, d_model)``.
        conv_state : Tensor
            Updated conv buffer.
        ssm_state : Tensor
            Updated SSM state.

        Notes
        -----
        Produces exactly what :meth:`forward` would for the same token given the
        same history (verified to ``atol=1e-4`` in the tests).
        """
        xz = self.in_proj(hidden_states)  # (B, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # (B, d_inner) each
        x_conv, conv_state = self.conv1d.step(x, conv_state)  # (B, d_inner)
        x_conv = F.silu(x_conv)
        y, ssm_state = self.ssm.step(x_conv, ssm_state)  # (B, d_inner)
        y = y * F.silu(z)
        out = self.out_proj(y)
        return out, conv_state, ssm_state

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> tuple[Tensor, Tensor]:
        """Allocate the (conv buffer, SSM state) pair for incremental decoding.

        Args
        ----
        batch_size:
            Number of sequences decoded in parallel.
        max_seqlen:
            Maximum decode length (unused for sizing -- Mamba's state is
            constant in length -- but kept for API parity with attention caches).
        dtype:
            Cache dtype.
        device:
            Cache device.

        Returns
        -------
        conv_state : Tensor
            ``(batch_size, d_inner, d_conv)`` zeros.
        ssm_state : Tensor
            ``(batch_size, d_inner, d_state)`` zeros.
        """
        conv_state = self.conv1d.allocate_inference_cache(batch_size, dtype, device)
        ssm_state = self.ssm.allocate_inference_cache(batch_size, dtype, device)
        return conv_state, ssm_state

    def _get_states_from_cache(
        self,
        inference_params: "InferenceParams",
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> tuple[Tensor, Tensor]:
        """Fetch (or lazily allocate) this layer's cached states."""
        key = self.layer_idx if self.layer_idx is not None else id(self)
        cache = inference_params.key_value_memory_dict
        if key not in cache:
            cache[key] = self.allocate_inference_cache(
                batch_size, inference_params.max_seqlen, dtype, device
            )
        states: tuple[Tensor, Tensor] = cache[key]
        return states

    def _store_states(
        self,
        inference_params: "InferenceParams",
        conv_state: Tensor,
        ssm_state: Tensor,
    ) -> None:
        """Write back this layer's updated states into the cache."""
        key = self.layer_idx if self.layer_idx is not None else id(self)
        inference_params.key_value_memory_dict[key] = (conv_state, ssm_state)
