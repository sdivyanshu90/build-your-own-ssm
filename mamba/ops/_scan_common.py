r"""Shared helpers for the selective-scan operators.

Both the sequential reference scan (:mod:`mamba.ops.selective_scan_naive`) and
the parallel associative scan (:mod:`mamba.ops.selective_scan_parallel`) must
apply *exactly* the same pre-processing -- step-size activation, discretization,
and output projection -- so that their results agree to numerical tolerance.
Factoring that shared logic here is what guarantees the equivalence the test
suite checks.

The selective recurrence implemented downstream is

.. math::

    h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t\,u_t, \qquad
    y_t = \langle C_t, h_t \rangle + D\,u_t,

with all quantities indexed by (batch ``b``, time ``t``, channel ``d``, state
``n``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from mamba.core.discretize import selective_zoh

__all__ = ["preprocess_delta", "prepare_scan_inputs", "project_output"]


def _validate_shapes(
    u: Tensor, delta: Tensor, A: Tensor, B: Tensor, C: Tensor, D: Tensor | None
) -> None:
    """Raise a descriptive ``ValueError`` if the scan inputs are inconsistent.

    Args
    ----
    u, delta:
        Both must be ``(batch, L, d_inner)``.
    A:
        ``(d_inner, d_state)`` diagonal continuous state matrix.
    B, C:
        Both must be ``(batch, L, d_state)``.
    D:
        ``(d_inner,)`` skip connection, or ``None``.

    Raises
    ------
    ValueError
        On any rank or dimension mismatch.
    """
    if u.ndim != 3:
        raise ValueError(f"u must be (batch, L, d_inner), got {tuple(u.shape)}")
    if delta.shape != u.shape:
        raise ValueError(
            f"delta shape {tuple(delta.shape)} must equal u shape {tuple(u.shape)}"
        )
    if A.ndim != 2:
        raise ValueError(f"A must be (d_inner, d_state), got {tuple(A.shape)}")
    d_inner, d_state = A.shape
    if u.shape[-1] != d_inner:
        raise ValueError(f"u channel dim {u.shape[-1]} != A d_inner {d_inner}")
    for name, t in (("B", B), ("C", C)):
        if t.ndim != 3 or t.shape[0] != u.shape[0] or t.shape[1] != u.shape[1]:
            raise ValueError(
                f"{name} must be (batch, L, d_state), got {tuple(t.shape)}"
            )
        if t.shape[-1] != d_state:
            raise ValueError(f"{name} state dim {t.shape[-1]} != A d_state {d_state}")
    if D is not None and (D.ndim != 1 or D.shape[0] != d_inner):
        raise ValueError(f"D must be ({d_inner},) or None, got {tuple(D.shape)}")


def preprocess_delta(
    delta: Tensor,
    delta_bias: Tensor | None,
    delta_softplus: bool,
) -> Tensor:
    r"""Apply the optional bias and softplus activation to the raw step size.

    Args
    ----
    delta:
        Raw step size of shape ``(batch, L, d_inner)``.
    delta_bias:
        Optional per-channel bias ``(d_inner,)`` added *before* the softplus.
    delta_softplus:
        If ``True`` apply ``softplus`` so that the resulting :math:`\Delta` is
        guaranteed positive.

    Returns
    -------
    Tensor
        The activated step size, same shape as ``delta``.

    Notes
    -----
    This mirrors Mamba's ``Delta = softplus(dt_proj(x) + dt_bias)``: the bias is
    the only learnable parameter folded into the scan, and softplus enforces
    :math:`\Delta > 0` which is required for a stable discretization.
    """
    if delta_bias is not None:
        delta = delta + delta_bias.to(delta.dtype)
    if delta_softplus:
        delta = F.softplus(delta)
    return delta


def prepare_scan_inputs(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None,
    delta_bias: Tensor | None,
    delta_softplus: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Discretize the continuous SSM and fold the input into ``B̄``.

    Args
    ----
    u, delta, A, B, C, D, delta_bias, delta_softplus:
        See :func:`mamba.ops.selective_scan_naive.selective_scan_naive`.

    Returns
    -------
    A_bar : Tensor
        Discrete per-step multipliers, shape ``(batch, L, d_inner, d_state)``.
    deltaB_u : Tensor
        Discrete per-step additive term :math:`\bar{B}_t \odot u_t`, shape
        ``(batch, L, d_inner, d_state)``.
    delta : Tensor
        The activated step size (returned for reuse/debugging).

    Raises
    ------
    ValueError
        If the input shapes are inconsistent.
    """
    _validate_shapes(u, delta, A, B, C, D)
    delta = preprocess_delta(delta, delta_bias, delta_softplus)
    A_bar, B_bar = selective_zoh(A, B, delta)  # (b, L, d, n) each
    deltaB_u = B_bar * u.unsqueeze(-1).to(B_bar.dtype)  # (b, L, d, n)
    return A_bar, deltaB_u, delta


def project_output(states: Tensor, C: Tensor, D: Tensor | None, u: Tensor) -> Tensor:
    r"""Map per-step states to outputs: :math:`y_t = \langle C_t, h_t\rangle + D u_t`.

    Args
    ----
    states:
        Hidden states for every step, shape ``(batch, L, d_inner, d_state)``.
    C:
        Output projection, shape ``(batch, L, d_state)``.
    D:
        Optional skip connection ``(d_inner,)``.
    u:
        The block input ``(batch, L, d_inner)`` (for the skip term).

    Returns
    -------
    Tensor
        Outputs ``y`` of shape ``(batch, L, d_inner)``. If ``states`` is complex
        the real part is taken (the SSM output is real-valued).
    """
    # y[b, l, d] = sum_n states[b, l, d, n] * C[b, l, n]
    y = torch.einsum("bldn,bln->bld", states, C.to(states.dtype))
    if y.is_complex():
        y = y.real
    if D is not None:
        y = y + u.to(y.dtype) * D.to(y.dtype)
    # The scan computes in (at least) float32 for stability; hand the result
    # back in the caller's dtype so the surrounding fp16/bf16 graph is intact.
    return y.to(u.dtype)
