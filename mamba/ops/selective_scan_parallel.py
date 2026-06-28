r"""Parallel (associative) implementation of the selective scan.

The selective recurrence :math:`h_t = \bar A_t h_{t-1} + \bar B_t u_t` is a
*first-order linear recurrence*. Such recurrences are prefix-scans under the
associative operator

.. math::

    (a_1, b_1) \oplus (a_2, b_2) = (a_2 a_1,\; a_2 b_1 + b_2),

which composes "multiply by :math:`a` then add :math:`b`" affine maps. Because
:math:`\oplus` is associative (proved below and tested in
``tests/property``), the scan can be evaluated in :math:`O(\log L)` parallel
depth instead of :math:`O(L)` sequential steps -- the key to running Mamba
efficiently on parallel hardware.

This module implements a work-efficient *log-depth* prefix scan in pure
PyTorch. It is built entirely from ``mul``/``add``/``cat`` so ordinary autograd
produces exact gradients (verified with :func:`torch.autograd.gradcheck`); no
custom backward or CUDA kernel is required. For very long sequences the scan
can additionally be wrapped in :func:`torch.utils.checkpoint.checkpoint` to
trade compute for :math:`O(L)` memory.

References
----------
[Blelloch, 1990] "Prefix sums and their applications."
[Gu & Dao, 2023] "Mamba: Linear-Time Sequence Modeling with Selective State
    Spaces."
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from mamba.ops._scan_common import prepare_scan_inputs, project_output

__all__ = ["selective_scan_parallel"]

# A scan state is the pair ``(multiplier a, addend b)`` of an affine map
# ``h -> a * h + b``.
ScanState = tuple[Tensor, Tensor]


def _make_scan_op() -> Callable[[ScanState, ScanState], ScanState]:
    r"""Return the associative operator :math:`\oplus` over affine maps.

    Returns
    -------
    Callable
        A function ``op(left, right)`` implementing
        ``(a_l, b_l) ⊕ (a_r, b_r) = (a_r a_l, a_r b_l + b_r)``, i.e. the
        composition "apply the left map, then the right map".

    Notes
    -----
    The identity element is ``(1, 0)`` (the identity affine map). Associativity
    follows from the associativity of affine-map composition; both properties
    are checked in :mod:`tests.property.test_ssm_properties`.
    """

    def op(left: ScanState, right: ScanState) -> ScanState:
        a_left, b_left = left
        a_right, b_right = right
        return a_right * a_left, a_right * b_left + b_right

    return op


def _parallel_prefix_scan(a: Tensor, b: Tensor) -> ScanState:
    r"""Inclusive prefix scan of the affine maps ``(a_t, b_t)`` along dim 1.

    Implements the Hillis-Steele log-depth scan: at iteration ``k`` every
    position ``t`` is combined with the partial result ``2^k`` positions to its
    left, so after :math:`\lceil\log_2 L\rceil` iterations each position holds
    the running composition ``(a_1,b_1) ⊕ ... ⊕ (a_t,b_t)``.

    Args
    ----
    a:
        Per-step multipliers, shape ``(batch, L, ...)``.
    b:
        Per-step addends, shape ``(batch, L, ...)`` matching ``a``.

    Returns
    -------
    a_cum : Tensor
        Cumulative multipliers (prefix products), same shape as ``a``.
    b_cum : Tensor
        Cumulative addends -- with a zero initial state these are exactly the
        hidden states :math:`h_t`, same shape as ``b``.

    Notes
    -----
    The scan handles arbitrary (non power-of-two) ``L`` with no padding because
    the shift-and-combine update is well defined for every length. Complexity is
    :math:`O(\log L)` depth and :math:`O(L \log L)` total work.
    """
    length = a.shape[1]
    a_cum = a
    b_cum = b
    shift = 1
    while shift < length:
        # Build the partial result ``shift`` positions to the left, padding the
        # first ``shift`` positions with the identity map (a=1, b=0).
        a_prev = torch.cat(
            [torch.ones_like(a_cum[:, :shift]), a_cum[:, : length - shift]], dim=1
        )
        b_prev = torch.cat(
            [torch.zeros_like(b_cum[:, :shift]), b_cum[:, : length - shift]], dim=1
        )
        # combine(prev, cur): use the *current* multiplier for both updates,
        # so compute the new addend before overwriting the multiplier.
        b_cum = a_cum * b_prev + b_cum
        a_cum = a_cum * a_prev
        shift *= 2
    return a_cum, b_cum


def selective_scan_parallel(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Tensor | None = None,
    delta_bias: Tensor | None = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    r"""Hardware-efficient parallel selective scan.

    Numerically equivalent to
    :func:`mamba.ops.selective_scan_naive.selective_scan_naive` but evaluated
    with a log-depth associative scan instead of a sequential loop.

    Args
    ----
    u, delta, A, B, C, D, delta_bias, delta_softplus, return_last_state:
        Identical in meaning and shape to
        :func:`mamba.ops.selective_scan_naive.selective_scan_naive`.

    Returns
    -------
    y : Tensor
        Output sequence, shape ``(batch, L, d_inner)``.
    last_state : Tensor, optional
        Final hidden state ``(batch, d_inner, d_state)`` if
        ``return_last_state`` is ``True``.

    Raises
    ------
    ValueError
        If the input shapes are inconsistent.

    Notes
    -----
    The associative operator and its proof of associativity live in
    :func:`_make_scan_op`. The scan itself (:func:`_parallel_prefix_scan`)
    materializes the full ``(batch, L, d_inner, d_state)`` state, giving
    :math:`O(B L D N)` memory -- the price of a pure-PyTorch parallel scan.
    A fused CUDA kernel (or :func:`torch.utils.checkpoint`) is required to reach
    the :math:`O(B D N)` memory of the reference loop.
    """
    A_bar, deltaB_u, _ = prepare_scan_inputs(
        u, delta, A, B, C, D, delta_bias, delta_softplus
    )  # (b, L, d, n) each

    _, states = _parallel_prefix_scan(A_bar, deltaB_u)  # (b, L, d, n)

    y = project_output(states, C, D, u)
    if return_last_state:
        return y, states[:, -1]
    return y
