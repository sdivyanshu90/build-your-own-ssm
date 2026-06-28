r"""Reference (sequential) implementation of the selective scan.

This is the *ground truth* against which the parallel scan is validated. It is
deliberately the most literal possible transcription of the recurrence

.. math::

    h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t \odot u_t, \qquad
    y_t = \langle C_t, h_t \rangle + D\,u_t,

using an explicit Python loop over the sequence. It is :math:`O(L)` sequential
steps and materializes the running state of shape ``(batch, d_inner, d_state)``
only -- never the full ``(batch, L, d_inner, d_state)`` tensor -- so its peak
state memory is :math:`O(B\,D\,N)`.

References
----------
[Gu & Dao, 2023] "Mamba: Linear-Time Sequence Modeling with Selective State
    Spaces", Algorithm 2 (selective scan).
"""

from __future__ import annotations

import torch
from torch import Tensor

from mamba.ops._scan_common import prepare_scan_inputs, project_output

__all__ = ["selective_scan_naive"]


def selective_scan_naive(
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
    r"""Sequential selective scan; correct but :math:`O(L)` in depth.

    Args
    ----
    u:
        Input sequence, shape ``(batch, L, d_inner)``.
    delta:
        Raw (pre-activation) step size, shape ``(batch, L, d_inner)``.
    A:
        Diagonal continuous state matrix, shape ``(d_inner, d_state)``. Entries
        should have negative real part for stability.
    B:
        Input-dependent input projection, shape ``(batch, L, d_state)``.
    C:
        Input-dependent output projection, shape ``(batch, L, d_state)``.
    D:
        Optional skip connection, shape ``(d_inner,)``.
    delta_bias:
        Optional bias added to ``delta`` before the softplus, shape
        ``(d_inner,)``.
    delta_softplus:
        Whether to apply ``softplus`` to ``delta`` (recommended; keeps
        :math:`\Delta > 0`).
    return_last_state:
        If ``True`` also return the final hidden state of shape
        ``(batch, d_inner, d_state)``.

    Returns
    -------
    y : Tensor
        Output sequence, shape ``(batch, L, d_inner)``.
    last_state : Tensor, optional
        Returned only if ``return_last_state`` is ``True``.

    Raises
    ------
    ValueError
        If the input shapes are inconsistent (see
        :func:`mamba.ops._scan_common.prepare_scan_inputs`).

    Notes
    -----
    Must agree with
    :func:`mamba.ops.selective_scan_parallel.selective_scan_parallel` to within
    ``atol=1e-4`` -- this function defines the semantics that the parallel scan
    must reproduce.
    """
    batch, length, d_inner = u.shape
    d_state = A.shape[1]

    A_bar, deltaB_u, _ = prepare_scan_inputs(
        u, delta, A, B, C, D, delta_bias, delta_softplus
    )  # (b, L, d, n) each

    h = torch.zeros(
        batch, d_inner, d_state, dtype=A_bar.dtype, device=u.device
    )  # (b, d, n)
    states = []
    for t in range(length):
        # h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ u_t   (the additive term already holds u).
        h = A_bar[:, t] * h + deltaB_u[:, t]
        states.append(h)
    states_t = torch.stack(states, dim=1)  # (b, L, d, n)

    y = project_output(states_t, C, D, u)
    if return_last_state:
        return y, h
    return y
