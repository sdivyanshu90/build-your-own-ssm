r"""The S6 selective state space model -- the core of Mamba.

S4 is *linear time-invariant*: its :math:`A, B, C` are fixed, so it processes
every token with the same dynamics and cannot decide, based on content, what to
remember or ignore. Mamba's S6 layer breaks this by making the step size
:math:`\Delta` and the projections :math:`B, C` **functions of the input**:

.. math::

    \Delta_t = \mathrm{softplus}(\mathrm{dt\_proj}(\mathrm{lowrank}(x_t))),\quad
    B_t = \mathrm{Lin}_B(x_t),\quad C_t = \mathrm{Lin}_C(x_t).

Input dependence destroys the convolutional shortcut S4 relied on, so the
recurrence must be evaluated with the associative *selective scan*
(:mod:`mamba.ops.selective_scan_parallel`). The state matrix ``A`` stays static
and diagonal; only :math:`\Delta, B, C` move with the data.

References
----------
[Gu & Dao, 2023] "Mamba: Linear-Time Sequence Modeling with Selective State
    Spaces", section 3 (the selection mechanism) and section 3.6 (parameter
    initialization).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mamba.core.discretize import selective_zoh
from mamba.ops.selective_scan_naive import selective_scan_naive
from mamba.ops.selective_scan_parallel import selective_scan_parallel

__all__ = ["SelectiveSSM"]


class SelectiveSSM(nn.Module):
    r"""S6: input-dependent (selective) state space model.

    Makes ``B``, ``C`` and :math:`\Delta` input-dependent via learned
    projections, breaking the LTI constraint of S4.

    Input shape:  ``(batch, L, d_inner)``
    Output shape: ``(batch, L, d_inner)``

    Args
    ----
    d_inner:
        Inner channel dimension :math:`D` (the SSM runs one diagonal system per
        channel).
    d_state:
        SSM state dimension :math:`N`.
    dt_rank:
        Rank of the low-rank :math:`\Delta` projection.
    dt_min, dt_max:
        Range for initialising the softplus bias so initial steps fall
        (log-uniformly) in ``[dt_min, dt_max]``.
    dt_init:
        ``"random"`` or ``"constant"`` initialisation of ``dt_proj.weight``.
    dt_scale:
        Multiplier on the ``dt_proj.weight`` init scale.
    dt_init_floor:
        Lower clamp on the initial :math:`\Delta` samples.
    use_fast_path:
        Use the parallel associative scan (``True``) or the sequential
        reference scan (``False``) in :meth:`forward`.
    bias:
        Whether ``x_proj`` carries a bias (the paper uses ``False``).

    Attributes
    ----------
    A_log : nn.Parameter
        ``(d_inner, d_state)``; the state matrix is ``A = -exp(A_log)``, which
        is strictly negative and therefore stable.
    D : nn.Parameter
        ``(d_inner,)`` skip connection.
    x_proj : nn.Linear
        Projects ``x_t`` to ``(Δ_raw, B_t, C_t)``.
    dt_proj : nn.Linear
        Expands the low-rank :math:`\Delta` to ``d_inner`` channels.
    """

    def __init__(
        self,
        d_inner: int,
        d_state: int = 16,
        dt_rank: Optional[int] = None,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        use_fast_path: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_inner <= 0 or d_state <= 0:
            raise ValueError("d_inner and d_state must be positive")
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank if dt_rank is not None else math.ceil(d_inner / 16)
        self.use_fast_path = use_fast_path

        # x_proj: x_t -> (Δ_raw, B_t, C_t). No bias (paper default).
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=bias)
        # dt_proj: low-rank Δ -> per-channel Δ. Bias carries the dt init.
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        self._initialize_dt_proj(
            self.dt_proj,
            self.dt_rank,
            dt_min,
            dt_max,
            dt_init,
            dt_scale,
            dt_init_floor,
        )

        # A initialised from the S4D-real HiPPO spectrum: A = -(1, 2, ..., N).
        a = torch.arange(1, d_state + 1, dtype=torch.float32)  # (N,)
        a = a.unsqueeze(0).repeat(d_inner, 1)  # (d_inner, N)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(d_inner))

    @staticmethod
    def _initialize_dt_proj(
        dt_proj: nn.Linear,
        dt_rank: int,
        dt_min: float,
        dt_max: float,
        dt_init: str,
        dt_scale: float,
        dt_init_floor: float,
    ) -> None:
        r"""Initialise ``dt_proj`` exactly as in the Mamba paper (section 3.6).

        Args
        ----
        dt_proj:
            The ``Linear(dt_rank, d_inner)`` to initialise in place.
        dt_rank, dt_min, dt_max, dt_init, dt_scale, dt_init_floor:
            See the class docstring.

        Raises
        ------
        ValueError
            If ``dt_init`` is not ``"random"`` or ``"constant"``.

        Notes
        -----
        The weight scale is :math:`\mathrm{dt\_scale}\cdot \mathrm{dt\_rank}^{-1/2}`.
        The bias is set so that :math:`\mathrm{softplus}(\text{bias})` is
        log-uniform in ``[dt_min, dt_max]``, using the inverse softplus
        :math:`b = d + \log(-\mathrm{expm1}(-d))`.
        """
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise ValueError(f"dt_init must be random|constant, got {dt_init!r}")

        d_inner = dt_proj.bias.shape[0]
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: b such that softplus(b) == dt.
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

    def _A(self) -> Tensor:
        """Return the (stable, negative) diagonal state matrix ``A = -exp(A_log)``."""
        return -torch.exp(self.A_log)

    def _project(self, u: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        r"""Compute the input-dependent ``(Δ, B, C)`` from the input.

        Args
        ----
        u:
            Input of shape ``(batch, L, d_inner)``.

        Returns
        -------
        delta : Tensor
            Positive step size ``(batch, L, d_inner)`` (softplus applied).
        B : Tensor
            ``(batch, L, d_state)``.
        C : Tensor
            ``(batch, L, d_state)``.
        """
        x_dbl = self.x_proj(u)  # (batch, L, dt_rank + 2N)
        delta_raw, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta_raw))  # (batch, L, d_inner), > 0
        return delta, B.contiguous(), C.contiguous()

    def _forward_parallel(
        self,
        u: Tensor,
        delta: Tensor,
        A: Tensor,
        B: Tensor,
        C: Tensor,
        return_last_state: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Parallel (training) path via the selective scan.

        Args
        ----
        u, delta, B, C:
            Input and input-dependent parameters (already softplus-activated
            ``delta``).
        A:
            Diagonal state matrix ``(d_inner, d_state)``.
        return_last_state:
            If ``True`` also return the final hidden state ``(batch, d_inner,
            d_state)`` (used to seed incremental decoding after a prefill).

        Returns
        -------
        Tensor or tuple
            Output ``(batch, L, d_inner)``, optionally paired with the final
            state.
        """
        scan = selective_scan_parallel if self.use_fast_path else selective_scan_naive
        return scan(
            u,
            delta,
            A,
            B,
            C,
            self.D,
            delta_softplus=False,
            return_last_state=return_last_state,
        )

    def _forward_recurrent(
        self,
        u_t: Tensor,
        delta_t: Tensor,
        A: Tensor,
        B_t: Tensor,
        C_t: Tensor,
        h: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""Single-step recurrence for :math:`O(1)` inference.

        Args
        ----
        u_t:
            Current input ``(batch, d_inner)``.
        delta_t:
            Current (positive) step size ``(batch, d_inner)``.
        A:
            Diagonal state matrix ``(d_inner, d_state)``.
        B_t, C_t:
            Current projections ``(batch, d_state)``.
        h:
            Previous hidden state ``(batch, d_inner, d_state)``.

        Returns
        -------
        y_t : Tensor
            Output for this step ``(batch, d_inner)``.
        h_new : Tensor
            Updated hidden state ``(batch, d_inner, d_state)``.

        Notes
        -----
        Uses :func:`mamba.core.discretize.selective_zoh` with ``L = 1`` so the
        discretization is bit-for-bit identical to the parallel path, which is
        what makes recurrent decoding match parallel scoring.
        """
        A_bar, B_bar = selective_zoh(
            A, B_t.unsqueeze(1), delta_t.unsqueeze(1)
        )  # (batch, 1, d_inner, d_state)
        A_bar = A_bar[:, 0]  # (batch, d_inner, d_state)
        deltaB_u = B_bar[:, 0] * u_t.unsqueeze(-1)  # (batch, d_inner, d_state)
        h_new = A_bar * h.to(A_bar.dtype) + deltaB_u
        y_t = torch.einsum("bdn,bn->bd", h_new, C_t.to(h_new.dtype))
        if y_t.is_complex():
            y_t = y_t.real
        y_t = y_t + u_t.to(y_t.dtype) * self.D.to(y_t.dtype)
        return y_t.to(u_t.dtype), h_new

    def forward(
        self,
        u: Tensor,
        inference_params: Optional[Any] = None,
        return_last_state: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Run the selective SSM over a full sequence (training / prefill).

        Args
        ----
        u:
            Input ``(batch, L, d_inner)``.
        inference_params:
            Unused here; single-token decoding goes through :meth:`step`. Present
            for API symmetry with the rest of the stack.
        return_last_state:
            If ``True`` also return the final hidden state, for seeding decode.

        Returns
        -------
        Tensor or tuple
            Output ``(batch, L, d_inner)``, optionally paired with the final
            hidden state ``(batch, d_inner, d_state)``.

        Raises
        ------
        ValueError
            If ``u`` is not 3-D with channel dimension ``d_inner``.
        """
        if u.ndim != 3 or u.shape[-1] != self.d_inner:
            raise ValueError(
                f"u must be (batch, L, {self.d_inner}), got {tuple(u.shape)}"
            )
        delta, B, C = self._project(u)
        return self._forward_parallel(
            u, delta, self._A(), B, C, return_last_state=return_last_state
        )

    def step(self, u_t: Tensor, ssm_state: Tensor) -> tuple[Tensor, Tensor]:
        r"""Advance one token, returning the output and the new state.

        Args
        ----
        u_t:
            Current input ``(batch, d_inner)``.
        ssm_state:
            Previous hidden state ``(batch, d_inner, d_state)``.

        Returns
        -------
        y_t : Tensor
            Output ``(batch, d_inner)``.
        ssm_state : Tensor
            Updated hidden state ``(batch, d_inner, d_state)``.
        """
        delta, B, C = self._project(u_t.unsqueeze(1))  # add length axis
        return self._forward_recurrent(
            u_t, delta[:, 0], self._A(), B[:, 0], C[:, 0], ssm_state
        )

    def allocate_inference_cache(
        self, batch_size: int, dtype: torch.dtype, device: torch.device | str = "cpu"
    ) -> Tensor:
        """Allocate a zeroed SSM hidden state for incremental decoding.

        Args
        ----
        batch_size:
            Number of sequences decoded in parallel.
        dtype:
            State dtype.
        device:
            State device.

        Returns
        -------
        Tensor
            Zeros of shape ``(batch_size, d_inner, d_state)``.
        """
        return torch.zeros(
            batch_size, self.d_inner, self.d_state, dtype=dtype, device=device
        )
