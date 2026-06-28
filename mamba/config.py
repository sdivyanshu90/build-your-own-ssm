"""Configuration objects for the Mamba model family.

This module defines :class:`MambaConfig`, a frozen-by-convention dataclass that
collects every hyperparameter required to instantiate a Mamba language model.
The configuration mirrors the reference implementation of
[Gu & Dao, 2023] while documenting the role and sensitivity of each field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Union

__all__ = ["MambaConfig"]


@dataclass
class MambaConfig:
    """Hyperparameters for a Mamba model.

    The defaults reproduce the ``mamba-130m`` configuration from the reference
    release (``d_model=768``, ``n_layers=24``).

    Args
    ----
    d_model:
        Width of the residual stream :math:`D`. Every block reads and writes a
        tensor of shape ``(batch, length, d_model)``.
    n_layers:
        Number of stacked :class:`~mamba.layers.residual.ResidualBlock` modules.
    d_state:
        SSM state dimension :math:`N` (called ``d_state`` or ``N`` in the
        paper). Controls how much history each channel can store. Memory and
        compute scale linearly in ``d_state``.
    d_conv:
        Kernel width of the causal depthwise convolution applied before the
        selective SSM. A small value (default ``4``) provides local context.
    expand:
        Inner expansion factor :math:`E`. The block projects ``d_model`` up to
        ``d_inner = expand * d_model`` for the SSM computation.
    dt_rank:
        Rank of the low-rank projection that produces the input-dependent step
        size :math:`\\Delta`. ``"auto"`` resolves to ``ceil(d_model / 16)``.
    dt_min, dt_max:
        Range used to initialise the softplus bias of ``dt_proj`` so that the
        initial :math:`\\Delta` values are spread log-uniformly in
        ``[dt_min, dt_max]``.
    dt_init:
        ``"random"`` initialises ``dt_proj.weight`` uniformly with a scale of
        ``dt_scale * dt_rank**-0.5``; ``"constant"`` uses that scale exactly.
    dt_scale:
        Multiplier on the ``dt_proj`` weight initialisation scale.
    dt_init_floor:
        Lower clamp on the initial :math:`\\Delta` values, preventing degenerate
        near-zero steps.
    conv_bias:
        Whether the causal convolution has a bias term.
    bias:
        Whether the input/output linear projections have bias terms. The Mamba
        paper uses ``False``.
    use_fast_path:
        When ``True`` the selective SSM uses the parallel associative scan;
        when ``False`` it falls back to the sequential reference scan. Both are
        pure PyTorch in this repository.
    layer_idx:
        Optional index of the block within the stack. Used to key per-layer
        inference caches.
    vocab_size:
        Size of the token vocabulary before padding.
    pad_vocab_size_multiple:
        The vocabulary is padded up to a multiple of this value so the output
        projection has a hardware-friendly shape.

    Notes
    -----
    Two derived quantities are computed in :meth:`__post_init__`:

    * ``d_inner = expand * d_model`` -- the expanded channel count fed to the
      SSM.
    * ``dt_rank`` -- if given as ``"auto"`` it is resolved to
      :math:`\\lceil D / 16 \\rceil`.
    """

    d_model: int = 768
    n_layers: int = 24
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"
    dt_scale: float = 1.0
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False
    use_fast_path: bool = True
    layer_idx: Optional[int] = None
    vocab_size: int = 50277
    pad_vocab_size_multiple: int = 8

    # Derived fields (populated in __post_init__); not part of the public init.
    d_inner: int = field(init=False)

    def __post_init__(self) -> None:
        """Validate inputs and compute derived hyperparameters.

        Raises
        ------
        ValueError
            If any structural hyperparameter is non-positive, if ``dt_min`` is
            not strictly less than ``dt_max``, or if ``dt_init`` is not one of
            the supported strings.
        """
        if self.d_model <= 0:
            raise ValueError(f"d_model must be positive, got {self.d_model}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.d_state <= 0:
            raise ValueError(f"d_state must be positive, got {self.d_state}")
        if self.d_conv <= 0:
            raise ValueError(f"d_conv must be positive, got {self.d_conv}")
        if self.expand <= 0:
            raise ValueError(f"expand must be positive, got {self.expand}")
        if not (self.dt_min < self.dt_max):
            raise ValueError(
                f"dt_min must be < dt_max, got {self.dt_min} >= {self.dt_max}"
            )
        if self.dt_init not in ("random", "constant"):
            raise ValueError(
                f"dt_init must be 'random' or 'constant', got {self.dt_init!r}"
            )

        object.__setattr__(self, "d_inner", self.expand * self.d_model)

        if self.dt_rank == "auto":
            object.__setattr__(self, "dt_rank", math.ceil(self.d_model / 16))
        elif isinstance(self.dt_rank, int):
            if self.dt_rank <= 0:
                raise ValueError(f"dt_rank must be positive, got {self.dt_rank}")
        else:
            raise ValueError(
                f"dt_rank must be a positive int or 'auto', got {self.dt_rank!r}"
            )

    @property
    def padded_vocab_size(self) -> int:
        """Vocabulary size rounded up to ``pad_vocab_size_multiple``."""
        multiple = self.pad_vocab_size_multiple
        if self.vocab_size % multiple == 0:
            return self.vocab_size
        return self.vocab_size + (multiple - self.vocab_size % multiple)

    @property
    def dt_rank_int(self) -> int:
        """``dt_rank`` resolved to a concrete integer.

        Returns
        -------
        int
            The integer rank of the :math:`\\Delta` projection.

        Raises
        ------
        RuntimeError
            If accessed before ``dt_rank`` has been resolved (should not happen
            after ``__post_init__``).
        """
        if isinstance(self.dt_rank, str):
            raise RuntimeError("dt_rank was not resolved to an int")
        return self.dt_rank
