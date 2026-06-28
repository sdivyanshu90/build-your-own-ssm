r"""Discretization of continuous-time state space models.

A continuous-time, single-input single-output (or multi-channel) linear state
space model evolves as

.. math::

    x'(t) = A\,x(t) + B\,u(t), \qquad y(t) = C\,x(t) + D\,u(t).

To run such a system on a digital computer we *discretize* it with a fixed (or,
in Mamba, input-dependent) step size :math:`\Delta`, producing a recurrence

.. math::

    x_k = \bar{A}\,x_{k-1} + \bar{B}\,u_k, \qquad y_k = C\,x_k + D\,u_k.

This module implements three classical discretization rules -- zero-order hold
(ZOH), the bilinear / Tustin transform, and forward Euler -- plus the batched,
diagonal ZOH used by Mamba's selective scan.

Design choices
--------------
* **Numerical robustness.** ZOH is computed with the *augmented matrix*
  identity rather than an explicit ``A^{-1}``. For the block matrix
  :math:`M = \begin{bmatrix} A & B \\ 0 & 0 \end{bmatrix}` one has

  .. math::

      \exp(\Delta M) = \begin{bmatrix} \bar{A} & \bar{B} \\ 0 & I \end{bmatrix},
      \quad \bar{A} = e^{\Delta A},\;
      \bar{B} = \Big(\textstyle\int_0^\Delta e^{\tau A}\,d\tau\Big) B .

  This is well defined even when ``A`` is singular (e.g. ``A = 0``), unlike the
  textbook formula :math:`\bar{B} = A^{-1}(\bar{A}-I)B`.
* **Compute dtype.** Matrix exponentials and linear solves are performed in
  ``float32`` (or ``complex64`` for complex ``A``) regardless of the input
  dtype, then cast back. This keeps ``float16``/``bfloat16`` callers safe.

References
----------
[Gu et al., 2021] "Efficiently Modeling Long Sequences with Structured State
    Spaces" (S4) -- discretization background.
[Gu & Dao, 2023] "Mamba: Linear-Time Sequence Modeling with Selective State
    Spaces" -- the selective (input-dependent) ZOH.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["zoh", "bilinear", "euler", "selective_zoh", "phi"]


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Return the float/complex dtype used for the numerically sensitive math.

    Args
    ----
    dtype:
        The dtype of the user-supplied tensors.

    Returns
    -------
    torch.dtype
        The smallest float/complex type with at least single precision:
        ``float16``/``bfloat16``/``float32`` are promoted to ``float32``,
        ``float64`` is kept, and complex inputs keep their precision.
    """
    if dtype.is_complex:
        return torch.complex64 if dtype == torch.complex64 else torch.complex128
    if dtype == torch.float64:
        return torch.float64
    return torch.float32


def _check_square(A: Tensor) -> int:
    """Validate that ``A`` is a (possibly batched) square matrix.

    Args
    ----
    A:
        Tensor of shape ``(..., N, N)``.

    Returns
    -------
    int
        The state dimension ``N``.

    Raises
    ------
    ValueError
        If ``A`` has fewer than two dimensions or its last two dims differ.
    """
    if A.ndim < 2:
        raise ValueError(f"A must have shape (..., N, N), got ndim={A.ndim}")
    if A.shape[-1] != A.shape[-2]:
        raise ValueError(f"A must be square in its last two dims, got {tuple(A.shape)}")
    return A.shape[-1]


def _as_delta(delta: float | Tensor, A: Tensor) -> Tensor:
    """Broadcast a scalar or per-batch step size to ``A``'s batch shape.

    Args
    ----
    delta:
        Either a Python float / 0-dim tensor (one step for the whole batch) or
        a tensor broadcastable to ``A.shape[:-2]`` (one step per batch element).
    A:
        The continuous state matrix of shape ``(..., N, N)``.

    Returns
    -------
    Tensor
        ``delta`` with two trailing singleton dims appended so it multiplies
        the ``(..., N, N)`` generator by broadcasting. A 1-D ``delta`` of length
        ``K`` against an unbatched ``A`` therefore *introduces* a leading batch
        of size ``K`` in the result.
    """
    if not isinstance(delta, Tensor):
        delta = torch.as_tensor(delta, dtype=torch.float32, device=A.device)
    else:
        delta = delta.to(device=A.device)
    return delta[..., None, None]


def zoh(A: Tensor, B: Tensor, delta: float | Tensor) -> tuple[Tensor, Tensor]:
    r"""Zero-order-hold discretization.

    Holds the input :math:`u` constant across each step of width
    :math:`\Delta`, giving the *exact* solution of the linear ODE over the
    interval.

    Args
    ----
    A:
        Continuous state matrix of shape ``(..., N, N)``. Real or complex.
    B:
        Continuous input matrix of shape ``(..., N, M)`` (a 1-D ``(N,)`` tensor
        is treated as a single input column ``(N, 1)``).
    delta:
        Step size. A scalar, 0-dim tensor, or a tensor broadcastable to the
        batch dimensions ``A.shape[:-2]``.

    Returns
    -------
    A_bar : Tensor
        Discrete state matrix :math:`\bar{A} = e^{\Delta A}`, shape
        ``(..., N, N)``.
    B_bar : Tensor
        Discrete input matrix, shape ``(..., N, M)``.

    Raises
    ------
    ValueError
        If ``A`` is not square or ``B``'s leading dimension does not match ``N``.

    Notes
    -----
    Computed via the augmented-matrix exponential

    .. math::

        \exp\!\left(\Delta\begin{bmatrix} A & B \\ 0 & 0\end{bmatrix}\right)
        = \begin{bmatrix} \bar{A} & \bar{B} \\ 0 & I \end{bmatrix},

    which avoids inverting ``A`` and therefore handles singular ``A`` (e.g.
    ``A = 0``) exactly. In the limit :math:`\Delta \to 0`, ``A_bar`` tends to
    the identity and ``B_bar`` to zero.
    """
    n = _check_square(A)
    squeeze_b = B.ndim == A.ndim - 1
    if squeeze_b:
        B = B.unsqueeze(-1)
    if B.shape[-2] != n:
        raise ValueError(f"B must have {n} rows to match A, got {tuple(B.shape)}")
    m = B.shape[-1]

    in_dtype = A.dtype
    cdtype = _compute_dtype(in_dtype)
    Ac = A.to(cdtype)
    Bc = B.to(cdtype)
    d = _as_delta(delta, A).to(cdtype)

    batch_shape = torch.broadcast_shapes(Ac.shape[:-2], Bc.shape[:-2])  # type: ignore[no-untyped-call]
    Ac = torch.broadcast_to(Ac, (*batch_shape, n, n))
    Bc = torch.broadcast_to(Bc, (*batch_shape, n, m))

    # Build the (N+M) x (N+M) augmented generator and exponentiate.
    top = torch.cat([Ac, Bc], dim=-1)  # (..., N, N+M)
    bottom = torch.zeros(
        (*batch_shape, m, n + m), dtype=cdtype, device=A.device
    )  # (..., M, N+M)
    M = torch.cat([top, bottom], dim=-2)  # (..., N+M, N+M)
    expM = torch.matrix_exp(d * M)

    A_bar = expM[..., :n, :n]
    B_bar = expM[..., :n, n:]
    if squeeze_b:
        B_bar = B_bar.squeeze(-1)
    return A_bar.to(in_dtype), B_bar.to(in_dtype)


def bilinear(A: Tensor, B: Tensor, delta: float | Tensor) -> tuple[Tensor, Tensor]:
    r"""Bilinear (Tustin) discretization.

    The trapezoidal rule applied to the state ODE. Maps the stable left half of
    the continuous ``s``-plane onto the unit disc of the discrete ``z``-plane,
    so stability is preserved exactly.

    Args
    ----
    A:
        Continuous state matrix ``(..., N, N)``.
    B:
        Continuous input matrix ``(..., N, M)`` (``(N,)`` allowed).
    delta:
        Step size; scalar or broadcastable to ``A.shape[:-2]``.

    Returns
    -------
    A_bar : Tensor
        :math:`\bar{A} = (I - \tfrac{\Delta}{2}A)^{-1}(I + \tfrac{\Delta}{2}A)`.
    B_bar : Tensor
        :math:`\bar{B} = (I - \tfrac{\Delta}{2}A)^{-1}\,\Delta B`.

    Raises
    ------
    ValueError
        If ``A`` is not square or ``B``'s leading dimension mismatches ``N``.

    Notes
    -----
    For small :math:`\Delta` the bilinear and ZOH maps agree to first order;
    they diverge for large steps where bilinear introduces frequency warping.
    """
    n = _check_square(A)
    squeeze_b = B.ndim == A.ndim - 1
    if squeeze_b:
        B = B.unsqueeze(-1)
    if B.shape[-2] != n:
        raise ValueError(f"B must have {n} rows to match A, got {tuple(B.shape)}")

    in_dtype = A.dtype
    cdtype = _compute_dtype(in_dtype)
    Ac = A.to(cdtype)
    Bc = B.to(cdtype)
    d = _as_delta(delta, A).to(cdtype)

    eye = torch.eye(n, dtype=cdtype, device=A.device)
    half = d / 2
    left = eye - half * Ac
    right = eye + half * Ac
    A_bar = torch.linalg.solve(left, right)
    B_bar = torch.linalg.solve(left, d * Bc)
    if squeeze_b:
        B_bar = B_bar.squeeze(-1)
    return A_bar.to(in_dtype), B_bar.to(in_dtype)


def euler(A: Tensor, B: Tensor, delta: float | Tensor) -> tuple[Tensor, Tensor]:
    r"""Forward (explicit) Euler discretization.

    The cheapest first-order rule. Only conditionally stable: for an eigenvalue
    :math:`\lambda` of ``A`` the discrete pole is :math:`1 + \Delta\lambda`,
    which can leave the unit disc for large :math:`\Delta`.

    Args
    ----
    A:
        Continuous state matrix ``(..., N, N)``.
    B:
        Continuous input matrix ``(..., N, M)`` (``(N,)`` allowed).
    delta:
        Step size; scalar or broadcastable to ``A.shape[:-2]``.

    Returns
    -------
    A_bar : Tensor
        :math:`\bar{A} = I + \Delta A`.
    B_bar : Tensor
        :math:`\bar{B} = \Delta B`.

    Raises
    ------
    ValueError
        If ``A`` is not square or ``B``'s leading dimension mismatches ``N``.
    """
    n = _check_square(A)
    squeeze_b = B.ndim == A.ndim - 1
    if squeeze_b:
        B = B.unsqueeze(-1)
    if B.shape[-2] != n:
        raise ValueError(f"B must have {n} rows to match A, got {tuple(B.shape)}")

    in_dtype = A.dtype
    cdtype = _compute_dtype(in_dtype)
    Ac = A.to(cdtype)
    Bc = B.to(cdtype)
    d = _as_delta(delta, A).to(cdtype)

    eye = torch.eye(n, dtype=cdtype, device=A.device)
    A_bar = eye + d * Ac
    B_bar = d * Bc
    if squeeze_b:
        B_bar = B_bar.squeeze(-1)
    return A_bar.to(in_dtype), B_bar.to(in_dtype)


def phi(x: Tensor) -> Tensor:
    r"""Numerically stable :math:`\varphi(x) = (e^x - 1) / x` with :math:`\varphi(0)=1`.

    Args
    ----
    x:
        Real or complex tensor.

    Returns
    -------
    Tensor
        ``expm1(x) / x`` away from zero, and the truncated Taylor series
        :math:`1 + x/2` near zero, avoiding the ``0/0`` singularity (and the
        ``NaN`` gradients it would otherwise create).

    Notes
    -----
    :math:`\varphi` is exactly the factor relating the ZOH discrete input matrix
    to the Euler one for diagonal systems:
    :math:`\bar{B} = \varphi(\Delta A)\,\Delta B`.
    """
    eps = torch.finfo(torch.float32).eps
    small = x.abs() < eps
    x_safe = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1.0 + x / 2.0, torch.expm1(x) / x_safe)


def selective_zoh(A: Tensor, B: Tensor, delta: Tensor) -> tuple[Tensor, Tensor]:
    r"""Batched, input-dependent zero-order-hold for the Mamba selective scan.

    Each of the ``D`` channels owns an independent *diagonal* state matrix, and
    the step size :math:`\Delta` varies with batch, time, and channel. This is
    the discretization at the heart of the S6 selective SSM.

    Args
    ----
    A:
        Diagonal continuous state entries, shape ``(D, N)``. Real or complex.
        Entry ``A[d, n]`` is the ``n``-th pole of channel ``d``.
    B:
        Input-dependent (and therefore time-varying) input vector, shape
        ``(batch, L, N)``, shared across the ``D`` channels.
    delta:
        Per-(batch, time, channel) step size, shape ``(batch, L, D)``. Must be
        positive.

    Returns
    -------
    A_bar : Tensor
        Discrete state factors, shape ``(batch, L, D, N)``, equal to
        :math:`\exp(\Delta \odot A)`.
    B_bar : Tensor
        Discrete input factors, shape ``(batch, L, D, N)``, equal to
        :math:`\varphi(\Delta A)\,\Delta\,B`.

    Raises
    ------
    ValueError
        If the shapes are inconsistent (``A`` not 2-D, trailing ``N`` mismatch,
        or ``delta``/``B`` not 3-D).

    Notes
    -----
    For a *uniform* step size and a single channel this reduces exactly to
    :func:`zoh` applied to ``diag(A)``. Mamba's reference code approximates
    :math:`\bar{B} \approx \Delta B` (i.e. :math:`\varphi \equiv 1`); here we
    keep the exact :math:`\varphi` factor so that the discretization is
    consistent with :func:`zoh` and the recurrent/parallel scans agree.
    """
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D (D, N), got shape {tuple(A.shape)}")
    if B.ndim != 3:
        raise ValueError(f"B must be 3-D (batch, L, N), got shape {tuple(B.shape)}")
    if delta.ndim != 3:
        raise ValueError(
            f"delta must be 3-D (batch, L, D), got shape {tuple(delta.shape)}"
        )
    d_inner, n = A.shape
    if B.shape[-1] != n:
        raise ValueError(f"B last dim {B.shape[-1]} must match A state dim N={n}")
    if delta.shape[-1] != d_inner:
        raise ValueError(
            f"delta last dim {delta.shape[-1]} must match A channel dim D={d_inner}"
        )

    in_dtype = A.dtype
    cdtype = _compute_dtype(in_dtype)  # float32, float64, or complex
    Ac = A.to(cdtype)  # (D, N)
    # Keep float64 precision when requested (e.g. for autograd gradcheck).
    delta_dtype = torch.float64 if cdtype == torch.float64 else torch.float32
    delta_f = delta.to(delta_dtype)  # (batch, L, D)

    # x = Δ ⊙ A broadcast to (batch, L, D, N).
    x = delta_f[..., None].to(cdtype) * Ac  # (b, L, D, N)
    A_bar = torch.exp(x)
    # B̄ = φ(ΔA) · Δ · B, with B broadcast over the channel axis D.
    B_bar = phi(x) * delta_f[..., None].to(cdtype) * B[:, :, None, :].to(cdtype)
    return A_bar, B_bar
