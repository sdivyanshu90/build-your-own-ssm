r"""Autoregressive generation for Mamba.

Because Mamba carries a *fixed-size* recurrent state -- not a growing KV cache --
decoding is :math:`O(1)` time and memory per token regardless of how much text
has already been produced. The :class:`InferenceParams` cache holds, per layer,
the convolution rolling buffer and the SSM hidden state; both have shapes
independent of the sequence length.

This module implements greedy decoding plus the usual stochastic controls:
temperature, top-k, nucleus (top-p), and a repetition penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["InferenceParams", "generate"]


@dataclass
class InferenceParams:
    """Recurrent-state cache threaded through the block stack during decoding.

    Attributes
    ----------
    max_seqlen:
        Maximum total sequence length (prompt + generated).
    max_batch_size:
        Maximum batch size the cache is sized for.
    seqlen_offset:
        Number of tokens already consumed. ``0`` marks the prefill pass; any
        positive value selects the single-token recurrent path in each block.
    key_value_memory_dict:
        Maps ``layer_idx`` to that layer's ``(conv_state, ssm_state)`` pair. The
        name mirrors the attention-cache convention even though Mamba stores SSM
        state, not keys/values.
    """

    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    key_value_memory_dict: dict[int, Any] = field(default_factory=dict)


class _Streamer(Protocol):
    """Minimal streamer protocol: receive each new token batch, then finish."""

    def put(self, token: Tensor) -> None: ...

    def end(self) -> None: ...


def apply_repetition_penalty(
    logits: Tensor, generated: Tensor, penalty: float
) -> Tensor:
    r"""Penalise the logits of already-generated tokens.

    Args
    ----
    logits:
        Next-token logits ``(batch, vocab)``.
    generated:
        Token ids produced so far ``(batch, T)``.
    penalty:
        Penalty factor (``> 1`` discourages repetition). Positive logits are
        divided by it and negative logits multiplied, following [Keskar et al.,
        2019].

    Returns
    -------
    Tensor
        The adjusted logits (a new tensor).
    """
    logits = logits.clone()
    score = torch.gather(logits, 1, generated)
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(1, generated, score)
    return logits


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Mask all but the ``k`` highest logits per row (set to ``-inf``).

    Args
    ----
    logits:
        ``(batch, vocab)`` logits.
    k:
        Number of tokens to keep.

    Returns
    -------
    Tensor
        Filtered logits.
    """
    k = min(k, logits.size(-1))
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Nucleus filtering: keep the smallest set of tokens with cumulative
    probability ``>= p`` (always keeping at least one).

    Args
    ----
    logits:
        ``(batch, vocab)`` logits.
    p:
        Cumulative-probability threshold in ``(0, 1]``.

    Returns
    -------
    Tensor
        Filtered logits.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_remove = cumulative > p
    # Shift right so the first token over the threshold is kept.
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False
    remove = sorted_remove.scatter(-1, sorted_indices, sorted_remove)
    return logits.masked_fill(remove, float("-inf"))


def _sample_next(
    logits: Tensor,
    do_sample: bool,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> Tensor:
    """Select the next token id ``(batch,)`` from per-row logits.

    Greedy (argmax) when ``do_sample`` is ``False`` or ``temperature == 0``;
    otherwise temperature-scaled sampling with optional top-k / top-p filtering.
    """
    if (not do_sample) or temperature == 0.0:
        return torch.argmax(logits, dim=-1)
    logits = logits / temperature
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def generate(
    model: Any,
    input_ids: Tensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    do_sample: bool = True,
    num_return_sequences: int = 1,
    eos_token_id: Optional[int] = None,
    streamer: Optional[_Streamer] = None,
) -> Tensor:
    """Generate continuations using the recurrent (constant-memory) path.

    Args
    ----
    model:
        A :class:`~mamba.models.lm_head.MambaLMHeadModel` (anything returning an
        object with a ``logits`` attribute and accepting ``inference_params``).
    input_ids:
        Prompt token ids ``(batch, prompt_len)``.
    max_new_tokens:
        Number of tokens to generate.
    temperature:
        Softmax temperature; ``0`` forces greedy decoding.
    top_k:
        If set, restrict sampling to the ``top_k`` most likely tokens.
    top_p:
        If set, restrict sampling to the nucleus with cumulative probability
        ``top_p``.
    repetition_penalty:
        Penalty (``> 1``) discouraging repeats; ``1.0`` disables it.
    do_sample:
        Sample (``True``) or take the argmax (``False``).
    num_return_sequences:
        Number of independent continuations per prompt (the batch is expanded).
    eos_token_id:
        If set, a sequence stops contributing new tokens after emitting it;
        generation ends early once all sequences are finished.
    streamer:
        Optional object receiving each new token batch via ``put`` then ``end``.

    Returns
    -------
    Tensor
        Token ids ``(batch * num_return_sequences, prompt_len + generated)``.

    Raises
    ------
    ValueError
        If ``input_ids`` is not a 2-D integer tensor.
    """
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be (batch, L), got {tuple(input_ids.shape)}")
    model.eval()
    device = input_ids.device

    if num_return_sequences > 1:
        input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
    batch, prompt_len = input_ids.shape

    inference_params = InferenceParams(
        max_seqlen=prompt_len + max_new_tokens, max_batch_size=batch
    )

    # Prefill: consume the prompt and seed the per-layer caches.
    logits = model(input_ids, inference_params=inference_params).logits[:, -1, :]
    inference_params.seqlen_offset = prompt_len

    generated = input_ids
    finished = torch.zeros(batch, dtype=torch.bool, device=device)

    for step in range(max_new_tokens):
        step_logits = logits
        if repetition_penalty != 1.0:
            step_logits = apply_repetition_penalty(
                step_logits, generated, repetition_penalty
            )
        next_token = _sample_next(step_logits, do_sample, temperature, top_k, top_p)

        if eos_token_id is not None:
            # Finished sequences keep emitting EOS so the batch stays aligned.
            next_token = torch.where(
                finished, torch.full_like(next_token, eos_token_id), next_token
            )

        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        if streamer is not None:
            streamer.put(next_token)

        if eos_token_id is not None:
            finished = finished | (next_token == eos_token_id)
            if bool(torch.all(finished)):
                break

        if step < max_new_tokens - 1:
            logits = model(
                next_token.unsqueeze(1), inference_params=inference_params
            ).logits[:, -1, :]
            inference_params.seqlen_offset += 1

    if streamer is not None:
        streamer.end()
    return generated
