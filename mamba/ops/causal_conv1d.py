r"""Causal depthwise 1-D convolution.

Mamba applies a short depthwise convolution to each channel before the
selective SSM, mixing a few neighbouring timesteps. To preserve autoregressive
causality the convolution may only look *backwards*: output ``t`` depends on
inputs ``t, t-1, ..., t-(K-1)`` and never on the future. This is achieved by
left-padding the sequence by ``K - 1`` and discarding the trailing samples.

During incremental (token-by-token) generation the same arithmetic is performed
with an :math:`O(K)` rolling buffer instead of a full convolution.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["CausalConv1d"]


class CausalConv1d(nn.Module):
    r"""Depthwise 1-D convolution with no future leakage.

    Args
    ----
    d_inner:
        Number of channels :math:`D`. The convolution is depthwise
        (``groups = d_inner``), so each channel is filtered independently.
    d_conv:
        Kernel width :math:`K`.
    bias:
        Whether to include a per-channel bias.

    Attributes
    ----------
    conv : nn.Conv1d
        The underlying grouped convolution with weight shape ``(D, 1, K)``.

    Notes
    -----
    Operates on the channels-first layout ``(batch, d_inner, L)`` used
    internally by :class:`~mamba.layers.mamba_block.MambaBlock`. The left pad of
    ``K - 1`` followed by truncation to the original length is what makes the
    operator causal.
    """

    def __init__(self, d_inner: int, d_conv: int, bias: bool = True) -> None:
        super().__init__()
        if d_inner <= 0 or d_conv <= 0:
            raise ValueError(
                f"d_inner and d_conv must be positive, got {d_inner}, {d_conv}"
            )
        self.d_inner = d_inner
        self.d_conv = d_conv
        self.conv = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""Apply the causal convolution to a full sequence.

        Args
        ----
        x:
            Input of shape ``(batch, d_inner, L)``.

        Returns
        -------
        Tensor
            Output of shape ``(batch, d_inner, L)``.

        Raises
        ------
        ValueError
            If ``x`` is not 3-D or its channel dim does not match ``d_inner``.
        """
        if x.ndim != 3:
            raise ValueError(f"x must be (batch, d_inner, L), got {tuple(x.shape)}")
        if x.shape[1] != self.d_inner:
            raise ValueError(f"x channels {x.shape[1]} != d_inner {self.d_inner}")
        length = x.shape[-1]
        # padding=K-1 pads both ends; keep only the first L outputs -> causal.
        out: Tensor = self.conv(x)
        return out[..., :length]

    def step(self, x_t: Tensor, conv_state: Tensor) -> tuple[Tensor, Tensor]:
        r"""Single-timestep convolution using a rolling buffer.

        Args
        ----
        x_t:
            Current input, shape ``(batch, d_inner)``.
        conv_state:
            Buffer of the most recent ``d_conv - 1`` inputs followed by the slot
            for the current input, shape ``(batch, d_inner, d_conv)``. It is
            rolled in place: the oldest column is dropped and ``x_t`` appended.

        Returns
        -------
        out : Tensor
            Convolved output for this timestep, shape ``(batch, d_inner)``.
        conv_state : Tensor
            The updated buffer, shape ``(batch, d_inner, d_conv)``.

        Raises
        ------
        ValueError
            If ``conv_state`` has the wrong trailing dimension.

        Notes
        -----
        Produces exactly the same value as the last position of
        :meth:`forward` applied to the corresponding window, but in
        :math:`O(d\_conv)` time and memory.
        """
        if conv_state.shape[-1] != self.d_conv:
            raise ValueError(
                f"conv_state last dim {conv_state.shape[-1]} != d_conv {self.d_conv}"
            )
        # Drop the oldest sample and append the new one.
        conv_state = conv_state.roll(shifts=-1, dims=-1)
        conv_state = conv_state.clone()
        conv_state[:, :, -1] = x_t
        weight = self.conv.weight.squeeze(1)  # (d_inner, d_conv)
        out = torch.sum(conv_state * weight, dim=-1)  # (batch, d_inner)
        if self.conv.bias is not None:
            out = out + self.conv.bias
        return out, conv_state

    def allocate_inference_cache(
        self, batch_size: int, dtype: torch.dtype, device: torch.device | str = "cpu"
    ) -> Tensor:
        """Allocate a zeroed rolling buffer for incremental decoding.

        Args
        ----
        batch_size:
            Number of sequences decoded in parallel.
        dtype:
            Buffer dtype.
        device:
            Buffer device.

        Returns
        -------
        Tensor
            Zeros of shape ``(batch_size, d_inner, d_conv)``.
        """
        return torch.zeros(
            batch_size, self.d_inner, self.d_conv, dtype=dtype, device=device
        )
