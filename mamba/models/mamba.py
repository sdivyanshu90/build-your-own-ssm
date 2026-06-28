r"""The Mamba backbone: token embedding, a stack of residual blocks, final norm.

Structurally this mirrors a GPT-2 transformer trunk -- embed, apply ``n_layers``
identical mixing blocks, normalise -- but every block is a linear-time selective
SSM rather than quadratic self-attention, and there is **no positional
encoding**: a continuous-time SSM is inherently sequential and time-aware.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch
from torch import Tensor, nn

from mamba.config import MambaConfig
from mamba.layers.norms import RMSNorm
from mamba.layers.residual import ResidualBlock

if TYPE_CHECKING:
    from mamba.utils.generation import InferenceParams

__all__ = ["MambaModel"]


class MambaModel(nn.Module):
    """Full Mamba backbone producing hidden states (no LM head).

    Args
    ----
    config:
        Model configuration.

    Attributes
    ----------
    embedding : nn.Embedding
        Token embedding ``(padded_vocab_size, d_model)``.
    layers : nn.ModuleList
        ``n_layers`` :class:`~mamba.layers.residual.ResidualBlock` modules.
    norm_f : RMSNorm
        Final normalization applied to the residual stream.
    """

    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.padded_vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [ResidualBlock(config, layer_idx=i) for i in range(config.n_layers)]
        )
        self.norm_f = RMSNorm(config.d_model)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Standard GPT-style initialization of linear/embedding weights."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: Tensor,
        inference_params: "Optional[InferenceParams]" = None,
    ) -> Tensor:
        """Embed tokens and run them through the block stack.

        Args
        ----
        input_ids:
            Integer token ids of shape ``(batch, L)``.
        inference_params:
            Optional incremental-decoding cache.

        Returns
        -------
        Tensor
            Final hidden states ``(batch, L, d_model)``.
        """
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, inference_params=inference_params)
        out: Tensor = self.norm_f(hidden)
        return out

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> dict[int, tuple[Tensor, Tensor]]:
        """Allocate per-layer (conv, SSM) state caches.

        Args
        ----
        batch_size, max_seqlen, dtype, device:
            Passed to each block's allocator.

        Returns
        -------
        dict
            Maps ``layer_idx`` to its ``(conv_state, ssm_state)`` pair.
        """
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype, device)
            for i, layer in enumerate(self.layers)
        }

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "MambaModel":
        """Load a backbone from a local checkpoint file.

        Args
        ----
        model_name_or_path:
            Path to a ``.pt`` checkpoint produced by
            :func:`mamba.utils.checkpoint.save_checkpoint` (must contain a
            ``"config"`` entry).
        device, dtype:
            Target device and dtype for the loaded model.

        Returns
        -------
        MambaModel
            The instantiated, weight-loaded backbone.

        Raises
        ------
        FileNotFoundError
            If the path does not exist (remote hub loading is not implemented).
        """
        path = Path(model_name_or_path)
        if not path.exists():
            raise FileNotFoundError(
                f"{model_name_or_path!r} not found; remote hub loading is not "
                "implemented in this from-scratch build."
            )
        ckpt = torch.load(path, map_location=device)
        config = ckpt["config"]
        model = cls(config).to(device=device, dtype=dtype)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        # The checkpoint may belong to an LM head model; keep backbone keys.
        backbone_state = {
            k[len("backbone.") :]: v
            for k, v in state.items()
            if k.startswith("backbone.")
        }
        model.load_state_dict(backbone_state or state, strict=False)
        return model
