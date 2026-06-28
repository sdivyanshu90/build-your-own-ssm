r"""Training utilities: optimizer/scheduler construction, clipping, and loss.

Encodes the recipe from the Mamba paper: AdamW with decoupled weight decay, the
decay applied only to matrix (>= 2-D) parameters -- never to biases, norm gains,
or the SSM's ``A_log`` / ``D`` -- a linear warmup followed by cosine decay, and
global-norm gradient clipping.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "build_optimizer",
    "build_scheduler",
    "clip_grad_norm_",
    "compute_loss",
]


def _is_no_decay(name: str, param: Tensor) -> bool:
    """Return whether a parameter should be excluded from weight decay.

    Args
    ----
    name:
        Fully-qualified parameter name.
    param:
        The parameter tensor.

    Returns
    -------
    bool
        ``True`` for 1-D parameters (biases, norm gains) and for the SSM's
        ``A_log`` and ``D``, which must not be decayed.
    """
    return param.ndim < 2 or name.endswith("A_log") or name.endswith(".D")


def build_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Build an AdamW optimizer with a decay / no-decay parameter split.

    Args
    ----
    model:
        The model whose parameters are optimized.
    lr:
        Peak learning rate.
    weight_decay:
        Decoupled weight decay applied to matrix parameters only.
    betas:
        AdamW moment coefficients (the paper uses ``(0.9, 0.95)``).
    eps:
        AdamW numerical epsilon.

    Returns
    -------
    torch.optim.AdamW
        The configured optimizer with two parameter groups.
    """
    decay: list[Tensor] = []
    no_decay: list[Tensor] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (no_decay if _is_no_decay(name, param) else decay).append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    r"""Linear warmup followed by cosine decay to ``min_lr_ratio * lr``.

    Args
    ----
    optimizer:
        The optimizer to schedule.
    warmup_steps:
        Number of steps over which the LR ramps linearly from 0 to peak.
    total_steps:
        Total training steps (warmup + decay).
    min_lr_ratio:
        Floor of the cosine decay as a fraction of the peak LR.

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
        The scheduler. SSMs tolerate larger learning rates than Transformers,
        so the peak LR passed to :func:`build_optimizer` is typically higher.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def clip_grad_norm_(model: nn.Module, max_norm: float) -> Tensor:
    """Clip gradients by global norm and return the (pre-clip) total norm.

    Args
    ----
    model:
        The model whose gradients are clipped in place.
    max_norm:
        Maximum allowed global gradient norm.

    Returns
    -------
    Tensor
        The total norm of the gradients before clipping (useful for logging).
    """
    total_norm: Tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return total_norm


def compute_loss(logits: Tensor, labels: Tensor, ignore_index: int = -100) -> Tensor:
    """Causal language-modeling cross-entropy with a one-position shift.

    Args
    ----
    logits:
        Predicted logits ``(batch, L, vocab)``.
    labels:
        Target token ids ``(batch, L)``. Position ``t`` of ``logits`` is trained
        to predict ``labels[t+1]``.
    ignore_index:
        Label value to skip in the loss (e.g. padding).

    Returns
    -------
    Tensor
        Scalar mean cross-entropy over the non-ignored positions.

    Raises
    ------
    ValueError
        If ``logits`` is not 3-D or its leading dims disagree with ``labels``.
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must be (batch, L, vocab), got {tuple(logits.shape)}")
    if labels.shape != logits.shape[:2]:
        raise ValueError(
            f"labels {tuple(labels.shape)} must match logits batch/length "
            f"{tuple(logits.shape[:2])}"
        )
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
