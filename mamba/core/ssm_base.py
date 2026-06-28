r"""Abstract base class and a concrete linear time-invariant (LTI) SSM.

A *time-invariant* state space model (as used by S4, and the pedagogical
stepping stone to Mamba) has fixed matrices :math:`A, B, C, D`. Such a model can
be evaluated two equivalent ways:

* **Recurrently** -- step the state :math:`h_t = \bar A h_{t-1} + \bar B u_t`;
  :math:`O(1)` memory and ideal for autoregressive inference.
* **Convolutionally** -- the output equals the input convolved with the SSM
  kernel :math:`\bar K = (C\bar B, C\bar A\bar B, C\bar A^2\bar B, \dots)`;
  fully parallel and ideal for training.

This module provides :class:`SSMBase`, which fixes that two-mode contract, and
:class:`ContinuousSSM`, a single-input single-output LTI model that discretizes
its continuous parameters on the fly and switches modes automatically.

References
----------
[Gu et al., 2021] "Efficiently Modeling Long Sequences with Structured State
    Spaces" (S4).
"""

from __future__ import annotations

import abc
from typing import Literal

import torch
from torch import Tensor, nn

from mamba.core.discretize import zoh
from mamba.core.hippo import hippo_legs

__all__ = ["SSMBase", "ContinuousSSM"]

Mode = Literal["auto", "conv", "recurrent"]


class SSMBase(nn.Module, abc.ABC):
    """Abstract base fixing the recurrent/convolutional duality of an SSM.

    Subclasses must implement :meth:`compute_kernel`, :meth:`forward_recurrent`
    and :meth:`forward_conv`. The base supplies mode management and a dispatcher
    :meth:`forward` that picks convolutional mode while training and recurrent
    mode otherwise (unless a mode is pinned via :meth:`set_training_mode`).
    """

    def __init__(self) -> None:
        super().__init__()
        self._mode: Mode = "auto"

    def set_training_mode(self, mode: Mode) -> None:
        """Pin the evaluation mode.

        Args
        ----
        mode:
            ``"auto"`` (convolution while ``training``, recurrence otherwise),
            ``"conv"`` (always convolutional), or ``"recurrent"`` (always
            recurrent).

        Raises
        ------
        ValueError
            If ``mode`` is not one of the three accepted strings.
        """
        if mode not in ("auto", "conv", "recurrent"):
            raise ValueError(f"mode must be auto|conv|recurrent, got {mode!r}")
        self._mode = mode

    def _use_conv(self) -> bool:
        """Return whether the current settings select convolutional mode."""
        if self._mode == "conv":
            return True
        if self._mode == "recurrent":
            return False
        return self.training

    @abc.abstractmethod
    def compute_kernel(self, length: int) -> Tensor:
        """Return the SSM convolution kernel of the given length."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward_recurrent(self, u: Tensor) -> Tensor:
        """Evaluate the SSM by stepping the recurrence."""
        raise NotImplementedError

    @abc.abstractmethod
    def forward_conv(self, u: Tensor) -> Tensor:
        """Evaluate the SSM as a convolution with its kernel."""
        raise NotImplementedError

    def forward(self, u: Tensor) -> Tensor:
        """Dispatch to convolutional or recurrent evaluation.

        Args
        ----
        u:
            Input sequence; shape defined by the concrete subclass.

        Returns
        -------
        Tensor
            The SSM output, same shape as ``u``.
        """
        return self.forward_conv(u) if self._use_conv() else self.forward_recurrent(u)


class ContinuousSSM(SSMBase):
    r"""A single-input single-output LTI SSM with HiPPO initialization.

    Holds continuous-time parameters :math:`A` (full ``N \times N``), :math:`B`,
    :math:`C` and a scalar :math:`D`, plus a learnable log step size. The
    discrete matrices are recomputed (via :func:`mamba.core.discretize.zoh`)
    every forward call so a single parameter set serves both modes.

    Args
    ----
    d_state:
        State dimension :math:`N`.
    dt:
        Initial discretization step size :math:`\Delta` (stored in log space).

    Attributes
    ----------
    A, B, C, D, log_dt : nn.Parameter
        Continuous parameters; ``A`` is initialised to HiPPO-LegS.

    Notes
    -----
    Operates on inputs of shape ``(batch, L)`` and returns ``(batch, L)``. The
    convolutional and recurrent paths are mathematically identical for this LTI
    model; the test suite asserts they agree to ``atol=1e-4``.
    """

    def __init__(self, d_state: int = 64, dt: float = 0.01) -> None:
        super().__init__()
        if d_state <= 0:
            raise ValueError(f"d_state must be positive, got {d_state}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.d_state = d_state
        A0, B0 = hippo_legs(d_state)  # (N, N), (N,)
        self.A = nn.Parameter(A0)  # (N, N)
        self.B = nn.Parameter(B0.unsqueeze(-1))  # (N, 1)
        self.C = nn.Parameter(torch.randn(1, d_state) / (d_state**0.5))  # (1, N)
        self.D = nn.Parameter(torch.zeros(1))
        self.log_dt = nn.Parameter(torch.log(torch.tensor(dt)))

    @property
    def dt(self) -> Tensor:
        """The current (positive) step size :math:`\\Delta = e^{\\log\\Delta}`."""
        return torch.exp(self.log_dt)

    def _discretize(self) -> tuple[Tensor, Tensor]:
        """Return the discrete ``(A_bar, B_bar)`` for the current parameters."""
        return zoh(self.A, self.B, self.dt)  # (N, N), (N, 1)

    def compute_kernel(self, length: int) -> Tensor:
        r"""SSM convolution kernel :math:`\bar K_k = C \bar A^k \bar B`.

        Args
        ----
        length:
            Kernel length :math:`L`.

        Returns
        -------
        Tensor
            Real kernel of shape ``(L,)``.

        Raises
        ------
        ValueError
            If ``length`` is not positive.
        """
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        A_bar, B_bar = self._discretize()  # (N, N), (N, 1)
        v = B_bar  # A_bar^0 B_bar, shape (N, 1)
        taps = []
        for _ in range(length):
            taps.append((self.C @ v).reshape(()))  # scalar C A^k B
            v = A_bar @ v
        return torch.stack(taps)  # (L,)

    def forward_conv(self, u: Tensor) -> Tensor:
        r"""Convolutional evaluation: :math:`y = (\bar K * u) + D u`.

        Args
        ----
        u:
            Input of shape ``(batch, L)``.

        Returns
        -------
        Tensor
            Output of shape ``(batch, L)``.

        Notes
        -----
        The causal convolution is computed with the FFT in :math:`O(L\log L)`.
        """
        if u.ndim != 2:
            raise ValueError(f"u must be (batch, L), got {tuple(u.shape)}")
        length = u.shape[-1]
        kernel = self.compute_kernel(length)  # (L,)
        fft_len = 2 * length
        u_f = torch.fft.rfft(u.float(), n=fft_len, dim=-1)
        k_f = torch.fft.rfft(kernel.float(), n=fft_len)
        y = torch.fft.irfft(u_f * k_f, n=fft_len, dim=-1)[..., :length]
        result: Tensor = (y + u * self.D).to(u.dtype)
        return result

    def forward_recurrent(self, u: Tensor) -> Tensor:
        r"""Recurrent evaluation by stepping :math:`h_t = \bar A h_{t-1}+\bar B u_t`.

        Args
        ----
        u:
            Input of shape ``(batch, L)``.

        Returns
        -------
        Tensor
            Output of shape ``(batch, L)``.
        """
        if u.ndim != 2:
            raise ValueError(f"u must be (batch, L), got {tuple(u.shape)}")
        batch, length = u.shape
        A_bar, B_bar = self._discretize()  # (N, N), (N, 1)
        h = torch.zeros(batch, self.d_state, 1, dtype=A_bar.dtype, device=u.device)
        outs = []
        for t in range(length):
            # h_t = A_bar h_{t-1} + B_bar u_t
            h = A_bar @ h + B_bar * u[:, t].reshape(batch, 1, 1)
            y_t = (self.C @ h).reshape(batch) + self.D * u[:, t]
            outs.append(y_t)
        return torch.stack(outs, dim=1)

    def get_ssm_matrices(self) -> dict[str, Tensor]:
        """Return the cached discrete state-space matrices.

        Returns
        -------
        dict
            ``{"A_bar", "B_bar", "C", "D"}`` -- the discretized matrices used by
            the recurrent path, useful for inspection or external inference.

        Notes
        -----
        Named ``get_ssm_matrices`` (not ``get_state_dict``) to avoid shadowing
        :meth:`torch.nn.Module.state_dict`, which would break checkpointing.
        """
        A_bar, B_bar = self._discretize()
        return {
            "A_bar": A_bar.detach(),
            "B_bar": B_bar.detach(),
            "C": self.C.detach(),
            "D": self.D.detach(),
        }
