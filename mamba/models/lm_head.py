r"""Language-model head wrapping the Mamba backbone.

Adds a token-prediction projection on top of :class:`~mamba.models.mamba.MambaModel`
and ties it to the input embedding (the standard weight-sharing trick that saves
``vocab_size * d_model`` parameters and tends to improve perplexity). Also
exposes :meth:`generate` for autoregressive decoding via the recurrent
:math:`O(1)`-per-step path -- no KV cache required.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

import torch
from torch import Tensor, nn

from mamba.config import MambaConfig
from mamba.models.mamba import MambaModel
from mamba.utils.training import compute_loss

if TYPE_CHECKING:
    from mamba.utils.generation import InferenceParams

__all__ = ["MambaLMHeadModel", "CausalLMOutput", "load_pretrained"]


class CausalLMOutput(NamedTuple):
    """Return type of :meth:`MambaLMHeadModel.forward`.

    Attributes
    ----------
    logits : Tensor
        Next-token logits ``(batch, L, vocab)``.
    loss : Tensor or None
        Cross-entropy loss if ``labels`` were supplied, else ``None``.
    """

    logits: Tensor
    loss: Optional[Tensor]


class MambaLMHeadModel(nn.Module):
    """Mamba backbone + tied language-model head.

    Args
    ----
    config:
        Model configuration.

    Attributes
    ----------
    backbone : MambaModel
        The embedding + block stack + final norm.
    lm_head : nn.Linear
        Projection to vocabulary logits, weight-tied to the embedding.
    """

    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = MambaModel(config)
        self.lm_head = nn.Linear(config.d_model, config.padded_vocab_size, bias=False)
        # Weight tying: share the embedding matrix with the output projection.
        self.lm_head.weight = self.backbone.embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        inference_params: "Optional[InferenceParams]" = None,
    ) -> CausalLMOutput:
        """Compute next-token logits and (optionally) the LM loss.

        Args
        ----
        input_ids:
            Token ids ``(batch, L)``.
        labels:
            Optional targets ``(batch, L)``; if given, the causal LM loss is
            computed with a one-position shift.
        inference_params:
            Optional incremental-decoding cache.

        Returns
        -------
        CausalLMOutput
            Named tuple of ``(logits, loss)``.
        """
        hidden = self.backbone(input_ids, inference_params=inference_params)
        logits = self.lm_head(hidden)
        loss = compute_loss(logits, labels) if labels is not None else None
        return CausalLMOutput(logits=logits, loss=loss)

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> dict[int, tuple[Tensor, Tensor]]:
        """Delegate cache allocation to the backbone."""
        return self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype, device
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        eos_token_id: Optional[int] = None,
        streamer: Optional[Any] = None,
    ) -> Tensor:
        """Autoregressively generate continuations (see :func:`mamba.utils.generation.generate`)."""
        from mamba.utils.generation import generate as _generate

        return _generate(
            self,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            eos_token_id=eos_token_id,
            streamer=streamer,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "MambaLMHeadModel":
        """Build an LM-head model from a local checkpoint (see :func:`load_pretrained`)."""
        return load_pretrained(model_name_or_path, device=device, dtype=dtype)


def load_pretrained(
    model_name_or_path: str,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> MambaLMHeadModel:
    """Load a :class:`MambaLMHeadModel` from a local checkpoint file.

    Args
    ----
    model_name_or_path:
        Path to a ``.pt`` checkpoint containing ``"config"`` and a state dict.
    device, dtype:
        Target device and dtype.

    Returns
    -------
    MambaLMHeadModel
        The loaded model.

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
    config: MambaConfig = ckpt["config"]
    model = MambaLMHeadModel(config).to(device=device, dtype=dtype)
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state)
    return model
