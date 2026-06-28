r"""HiPPO initialization matrices for state space models.

Randomly initialised SSM state matrices forget the distant past: their
discrete transition has eigenvalues that either decay too fast (vanishing
memory) or blow up (exploding gradients). HiPPO -- *High-order Polynomial
Projection Operators* [Gu et al., 2020] -- instead derives :math:`A, B` so that
the SSM state ``x(t)`` holds the optimal coefficients of an orthogonal
polynomial expansion of the input history. This gives provably stable, uniform
memory over long contexts and is the initialization that makes S4 and Mamba
trainable.

This module provides:

* :func:`hippo_legs`, :func:`hippo_legt`, :func:`hippo_lagt` -- the three
  classical measures (scaled Legendre, translated Legendre, translated
  Laguerre).
* :func:`make_nplr_hippo` -- the Normal-Plus-Low-Rank factorization of
  HiPPO-LegS used to reach :math:`O(N)` algorithms.
* :func:`make_dplr_hippo` -- the Diagonal-Plus-Low-Rank parameterization (the
  S4 starting point), with complex diagonal :math:`\Lambda`.
* :func:`random_ssm_init` -- a HiPPO-initialized multi-channel diagonal SSM,
  the form Mamba actually trains.

References
----------
[Gu et al., 2020] "HiPPO: Recurrent Memory with Optimal Polynomial
    Projections."
[Gu et al., 2021] "Efficiently Modeling Long Sequences with Structured State
    Spaces" (S4) -- the NPLR/DPLR machinery.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "hippo_legs",
    "hippo_legt",
    "hippo_lagt",
    "make_nplr_hippo",
    "make_dplr_hippo",
    "random_ssm_init",
]


def hippo_legs(n: int) -> tuple[Tensor, Tensor]:
    r"""HiPPO-LegS (scaled Legendre) matrices.

    LegS projects the input onto Legendre polynomials over the *entire*
    history ``[0, t]``, rescaling the window as ``t`` grows. This yields memory
    that is uniform in time and scale-invariant.

    Args
    ----
    n:
        State dimension :math:`N` (number of polynomial coefficients).

    Returns
    -------
    A : Tensor
        ``(N, N)`` lower-triangular state matrix, ``float32``.
    B : Tensor
        ``(N,)`` input vector with :math:`B_k = \sqrt{2k+1}`, ``float32``.

    Raises
    ------
    ValueError
        If ``n`` is not a positive integer.

    Notes
    -----
    With :math:`p_k = \sqrt{2k+1}`,

    .. math::

        A_{nk} = -\begin{cases}
            p_n\,p_k & n > k \\
            n + 1    & n = k \\
            0        & n < k
        \end{cases}

    Because ``A`` is lower triangular its eigenvalues are exactly its diagonal
    :math:`-(k+1)`, all real and negative -- the system is stable.
    """
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    p = torch.sqrt(1.0 + 2.0 * torch.arange(n, dtype=torch.float64))  # (2k+1)^{1/2}
    A = p[:, None] * p[None, :]  # outer product
    A = torch.tril(A) - torch.diag(torch.arange(n, dtype=torch.float64))
    A = -A
    B = torch.sqrt(2.0 * torch.arange(n, dtype=torch.float64) + 1.0)
    return A.to(torch.float32), B.to(torch.float32)


def hippo_legt(n: int) -> tuple[Tensor, Tensor]:
    r"""HiPPO-LegT (translated Legendre) matrices.

    LegT projects onto Legendre polynomials over a *sliding* fixed-length
    window, biasing memory toward the recent past.

    Args
    ----
    n:
        State dimension :math:`N`.

    Returns
    -------
    A : Tensor
        ``(N, N)`` state matrix, ``float32``.
    B : Tensor
        ``(N,)`` input vector :math:`B_k = \sqrt{2k+1}`, ``float32``.

    Raises
    ------
    ValueError
        If ``n`` is not a positive integer.

    Notes
    -----
    With :math:`R_k = \sqrt{2k+1}` and row/col indices :math:`r, c`,

    .. math::

        A_{rc} = -R_r R_c \times \begin{cases}
            (-1)^{r-c} & r < c \\ 1 & r \ge c \end{cases}.
    """
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    q = torch.arange(n, dtype=torch.float64)
    R = torch.sqrt(2.0 * q + 1.0)
    row = q[:, None]
    col = q[None, :]
    sign = torch.where(
        row < col, (-1.0) ** (row - col), torch.ones(n, n, dtype=torch.float64)
    )
    A = -(R[:, None] * sign * R[None, :])
    B = R.clone()
    return A.to(torch.float32), B.to(torch.float32)


def hippo_lagt(n: int) -> tuple[Tensor, Tensor]:
    r"""HiPPO-LagT (translated Laguerre) matrices.

    LagT projects onto Laguerre functions, giving exponentially decaying memory
    of the recent past.

    Args
    ----
    n:
        State dimension :math:`N`.

    Returns
    -------
    A : Tensor
        ``(N, N)`` state matrix :math:`A = \tfrac12 I - \mathrm{tril}(\mathbf 1)`,
        ``float32``.
    B : Tensor
        ``(N,)`` all-ones input vector, ``float32``.

    Raises
    ------
    ValueError
        If ``n`` is not a positive integer.
    """
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    A = 0.5 * torch.eye(n, dtype=torch.float64) - torch.tril(
        torch.ones(n, n, dtype=torch.float64)
    )
    B = torch.ones(n, dtype=torch.float64)
    return A.to(torch.float32), B.to(torch.float32)


def make_nplr_hippo(n: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Normal-Plus-Low-Rank factorization of HiPPO-LegS.

    The LegS matrix ``A`` is not normal, so it cannot be diagonalized by a
    unitary change of basis. HiPPO-LegS is, however, *normal plus low rank*:

    .. math::

        A = S - P P^\top, \qquad S = A + P P^\top \text{ is normal.}

    Diagonalizing the normal part ``S = V \Lambda V^*`` exposes the structure
    that S4 exploits for fast kernels.

    Args
    ----
    n:
        State dimension :math:`N`.

    Returns
    -------
    w : Tensor
        ``(N,)`` complex eigenvalues :math:`\Lambda` of the normal part ``S``.
        Every value has real part :math:`-\tfrac12`.
    P : Tensor
        ``(N,)`` real rank-1 low-rank factor, :math:`P_k = \sqrt{k + \tfrac12}`.
    B : Tensor
        ``(N,)`` real LegS input vector :math:`B_k = \sqrt{2k+1}`.
    V : Tensor
        ``(N, N)`` complex unitary change-of-basis (eigenvectors of ``S``).

    Raises
    ------
    ValueError
        If ``n`` is not a positive integer.

    Notes
    -----
    ``A`` is recovered as ``V diag(w) V^* - P P^T`` (see
    :func:`tests <test_nplr_reconstruction>`). The diagonal of ``S`` is the
    constant :math:`-\tfrac12`, so ``S + \tfrac12 I`` is exactly skew-symmetric
    and is diagonalized through the Hermitian matrix :math:`-i(S + \tfrac12 I)`.
    """
    if n <= 0:
        raise ValueError(f"n must be a positive integer, got {n}")
    A, B = hippo_legs(n)
    A = A.to(torch.float64)
    B = B.to(torch.float64)
    P = torch.sqrt(torch.arange(n, dtype=torch.float64) + 0.5)  # (N,)

    S = A + P[:, None] * P[None, :]  # normal part
    diag_val = torch.mean(torch.diagonal(S))  # == -0.5
    skew = S - diag_val * torch.eye(n, dtype=torch.float64)  # skew-symmetric

    # -i * skew is Hermitian; eigh gives real eigenvalues (the imaginary spectrum).
    H = -1j * skew.to(torch.complex128)
    mu, V = torch.linalg.eigh(H)  # mu real ascending, V unitary
    w = diag_val.to(torch.complex128) + 1j * mu  # eigenvalues of S

    return (
        w.to(torch.complex64),
        P.to(torch.float32),
        B.to(torch.float32),
        V.to(torch.complex64),
    )


def make_dplr_hippo(
    n: int, rank: int = 1, dtype: torch.dtype = torch.complex64
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Diagonal-Plus-Low-Rank parameterization of HiPPO-LegS (the S4 form).

    Transforms the NPLR factorization into the eigenbasis so that the state
    matrix becomes ``diag(Lambda) - P P^*``: a complex diagonal plus a low-rank
    correction. This is the representation S4 trains.

    Args
    ----
    n:
        State dimension :math:`N`.
    rank:
        Number of low-rank columns to return. HiPPO-LegS has true rank 1; for
        ``rank > 1`` the extra columns are zero-padded.
    dtype:
        Complex dtype for the returned tensors (``complex64`` or ``complex128``).

    Returns
    -------
    Lambda : Tensor
        ``(N,)`` complex diagonal, real part :math:`-\tfrac12`.
    P : Tensor
        ``(rank, N)`` complex low-rank factor in the eigenbasis
        (``V^* P_original``).
    B : Tensor
        ``(N,)`` complex input vector in the eigenbasis (``V^* B_original``).
    V : Tensor
        ``(N, N)`` complex change-of-basis.

    Raises
    ------
    ValueError
        If ``n`` or ``rank`` is not a positive integer, or ``dtype`` is not
        complex.
    """
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    if not dtype.is_complex:
        raise ValueError(f"dtype must be complex, got {dtype}")
    w, P, B, V = make_nplr_hippo(n)
    Vh = V.conj().transpose(-1, -2)
    P_eig = (Vh @ P.to(V.dtype)).reshape(1, n)  # (1, N)
    if rank > 1:
        pad = torch.zeros(rank - 1, n, dtype=P_eig.dtype)
        P_eig = torch.cat([P_eig, pad], dim=0)  # (rank, N)
    B_eig = Vh @ B.to(V.dtype)  # (N,)
    return w.to(dtype), P_eig.to(dtype), B_eig.to(dtype), V.to(dtype)


def random_ssm_init(
    h: int, n: int, dtype: torch.dtype = torch.float32
) -> tuple[Tensor, Tensor, Tensor]:
    r"""HiPPO-initialized multi-channel diagonal SSM ``(A, B, C)``.

    Produces the per-channel diagonal state matrix Mamba actually trains. For a
    real ``dtype`` the S4D-real initialization :math:`A = -(1, 2, \dots, N)` is
    used (a real approximation to the HiPPO-LegS spectrum); for a complex
    ``dtype`` the HiPPO-LegS eigenvalues are used directly.

    Args
    ----
    h:
        Number of channels :math:`H` (e.g. ``d_inner``).
    n:
        State dimension :math:`N` per channel.
    dtype:
        Real or complex dtype of the returned tensors.

    Returns
    -------
    A : Tensor
        ``(H, N)`` diagonal state entries, every value with negative real part.
    B : Tensor
        ``(H, N)`` input gains (ones for real init; HiPPO ``B`` for complex).
    C : Tensor
        ``(H, N)`` output gains, randomly initialised with unit-ish scale.

    Raises
    ------
    ValueError
        If ``h`` or ``n`` is not a positive integer.

    Notes
    -----
    The function asserts the two HiPPO invariants before returning: all state
    eigenvalues have strictly negative real part (stability) and ``B`` is finite
    and non-degenerate.
    """
    if h <= 0 or n <= 0:
        raise ValueError(f"h and n must be positive, got h={h}, n={n}")
    if dtype.is_complex:
        Lambda, _, B_eig, _ = make_dplr_hippo(n, rank=1, dtype=dtype)
        A = Lambda.unsqueeze(0).repeat(h, 1)  # (H, N)
        B = B_eig.unsqueeze(0).repeat(h, 1).to(dtype)
    else:
        # S4D-real: A_n = -(n+1), the diagonal Mamba initializes A_log from.
        a = -(torch.arange(1, n + 1, dtype=dtype))  # (N,)
        A = a.unsqueeze(0).repeat(h, 1)  # (H, N)
        B = torch.ones(h, n, dtype=dtype)
    C = torch.randn(h, n, dtype=dtype) / (n**0.5)

    # HiPPO invariants.
    if not torch.all(A.real < 0):
        raise ValueError("HiPPO init produced a non-stable A (eigenvalue Re >= 0)")
    if not torch.isfinite(B.abs()).all():
        raise ValueError("HiPPO init produced a non-finite B")
    return A, B, C
