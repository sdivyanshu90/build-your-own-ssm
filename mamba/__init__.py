"""Mamba: a from-first-principles selective state space model.

This package implements the Mamba architecture of [Gu & Dao, 2023] together
with its mathematical antecedents (HiPPO, S4) in pure PyTorch, with no external
state-space-model dependencies.

The public API is intentionally small::

    from mamba import MambaConfig, MambaModel, MambaBlock, SelectiveSSM

References
----------
[Gu & Dao, 2023] A. Gu and T. Dao. "Mamba: Linear-Time Sequence Modeling with
    Selective State Spaces." 2023.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mamba.config import MambaConfig

__all__ = [
    "MambaConfig",
    "MambaModel",
    "MambaLMHeadModel",
    "MambaBlock",
    "SelectiveSSM",
    "load_pretrained",
]

__version__ = "0.1.0"

if TYPE_CHECKING:
    from mamba.core.selective_ssm import SelectiveSSM
    from mamba.layers.mamba_block import MambaBlock
    from mamba.models.lm_head import MambaLMHeadModel, load_pretrained
    from mamba.models.mamba import MambaModel


def __getattr__(name: str) -> Any:
    """Lazily resolve the heavy public symbols on first access (PEP 562).

    Importing the model classes eagerly would force ``torch`` and the entire
    layer stack to load merely to read :class:`MambaConfig`. Deferring keeps
    ``import mamba`` cheap and avoids import cycles during the build.
    """
    if name == "SelectiveSSM":
        from mamba.core.selective_ssm import SelectiveSSM

        return SelectiveSSM
    if name == "MambaBlock":
        from mamba.layers.mamba_block import MambaBlock

        return MambaBlock
    if name == "MambaModel":
        from mamba.models.mamba import MambaModel

        return MambaModel
    if name in ("MambaLMHeadModel", "load_pretrained"):
        from mamba.models import lm_head

        return getattr(lm_head, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
