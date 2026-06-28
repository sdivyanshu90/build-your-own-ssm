r"""Normalization layers used by the Mamba stack.

Mamba (like LLaMA and many modern LMs) uses :class:`RMSNorm` rather than
LayerNorm: it rescales by the root-mean-square of the activations without
subtracting the mean or learning a bias, which is slightly cheaper and works as
well in practice.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["RMSNorm", "LayerNorm"]


class RMSNorm(nn.Module):
    r"""Root-mean-square layer normalization (no mean subtraction, no bias).

    Args
    ----
    d_model:
        Feature dimension normalised over (the last axis).
    eps:
        Numerical floor added inside the square root.

    Attributes
    ----------
    weight : nn.Parameter
        ``(d_model,)`` learnable per-feature gain, initialised to ones.

    Notes
    -----
    Computes

    .. math::

        y = \frac{x}{\sqrt{\tfrac{1}{d}\sum_i x_i^2 + \varepsilon}} \odot w.

    The reduction is performed in ``float32`` regardless of the input dtype so
    that ``float16``/``bfloat16`` activations do not overflow when squared.
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        """Normalise the last dimension of ``x``.

        Args
        ----
        x:
            Input whose last dimension equals ``d_model``.

        Returns
        -------
        Tensor
            Normalised tensor of the same shape and dtype as ``x``.
        """
        in_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        result: Tensor = x_normed.to(in_dtype) * self.weight
        return result


class LayerNorm(nn.Module):
    """Thin wrapper around :class:`torch.nn.LayerNorm` for API consistency.

    Args
    ----
    d_model:
        Feature dimension normalised over.
    eps:
        Numerical floor.
    bias:
        Whether to learn a bias term.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Apply layer normalization to the last dimension of ``x``."""
        out: Tensor = self.norm(x)
        return out
