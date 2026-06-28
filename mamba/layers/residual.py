r"""Pre-norm residual wrapper around a Mamba block.

Mamba uses the now-standard *pre-norm* residual arrangement:

.. math::

    y = x + \mathrm{Block}(\mathrm{Norm}(x)).

Normalizing the input to each block (rather than the output) keeps the residual
stream un-normalized, which stabilises the gradients of deep stacks. Keeping the
norm and the skip connection in this wrapper -- not inside
:class:`~mamba.layers.mamba_block.MambaBlock` -- lets the block's training and
single-step inference paths stay byte-for-byte equivalent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor, nn

from mamba.config import MambaConfig
from mamba.layers.mamba_block import MambaBlock
from mamba.layers.norms import RMSNorm

if TYPE_CHECKING:
    from mamba.utils.generation import InferenceParams

__all__ = ["ResidualBlock"]


class ResidualBlock(nn.Module):
    """A Mamba block with pre-normalization, a residual connection and dropout.

    Args
    ----
    config:
        The model configuration.
    layer_idx:
        Index of this block within the stack (keys the inference cache).
    dropout:
        Dropout probability applied to the block output before the residual add.

    Attributes
    ----------
    norm : RMSNorm
        Pre-normalization applied to the block input.
    mixer : MambaBlock
        The selective-SSM block.
    """

    def __init__(
        self, config: MambaConfig, layer_idx: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.norm = RMSNorm(config.d_model)
        self.mixer = MambaBlock(config, layer_idx=layer_idx)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        hidden_states: Tensor,
        inference_params: "Optional[InferenceParams]" = None,
    ) -> Tensor:
        r"""Apply pre-norm, the mixer, dropout, and the residual add.

        Args
        ----
        hidden_states:
            Input ``(batch, L, d_model)``.
        inference_params:
            Optional incremental-decoding cache passed through to the mixer.

        Returns
        -------
        Tensor
            Output ``(batch, L, d_model)``, the residual stream after this block.
        """
        residual = hidden_states
        normed = self.norm(hidden_states)
        out = self.mixer(normed, inference_params=inference_params)
        result: Tensor = residual + self.dropout(out)
        return result

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> tuple[Tensor, Tensor]:
        """Delegate cache allocation to the wrapped :class:`MambaBlock`."""
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype, device
        )
