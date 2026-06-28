r"""Checkpoint saving, loading, and reference-format conversion.

A checkpoint bundles the model weights, the :class:`~mamba.config.MambaConfig`
needed to rebuild the architecture, the optimizer/scheduler state for resuming,
and the step counter. :func:`convert_from_reference` translates key names from
the official ``mamba_ssm`` release into this repository's module layout (mainly
the ``conv1d`` wrapper and the ``ssm`` sub-module).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

__all__ = ["save_checkpoint", "load_checkpoint", "convert_from_reference"]


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    step: int,
    path: str | Path,
) -> None:
    """Write a resumable training checkpoint to ``path``.

    Args
    ----
    model:
        The model whose ``state_dict`` and ``config`` are saved.
    optimizer:
        Optimizer whose state is saved (``None`` to skip).
    scheduler:
        LR scheduler whose state is saved (``None`` to skip).
    step:
        Global step counter stored alongside the weights.
    path:
        Destination file path.

    Notes
    -----
    ``model.config`` (a :class:`MambaConfig`) is stored so the architecture can
    be reconstructed by :func:`mamba.models.lm_head.load_pretrained` without the
    caller re-specifying hyperparameters.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "step": step,
    }
    if hasattr(model, "config"):
        payload["config"] = model.config
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    map_location: torch.device | str = "cpu",
) -> int:
    """Restore model (and optionally optimizer/scheduler) state from a checkpoint.

    Args
    ----
    path:
        Checkpoint file written by :func:`save_checkpoint`.
    model:
        Model to load weights into (in place).
    optimizer:
        Optimizer to restore (if provided and present in the checkpoint).
    scheduler:
        Scheduler to restore (if provided and present in the checkpoint).
    map_location:
        Device mapping for :func:`torch.load`.

    Returns
    -------
    int
        The stored global step (``0`` if absent).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("step", 0))


# Mapping from official ``mamba_ssm`` key fragments to this repo's layout.
_REFERENCE_RENAMES: tuple[tuple[str, str], ...] = (
    ("mixer.conv1d.weight", "mixer.conv1d.conv.weight"),
    ("mixer.conv1d.bias", "mixer.conv1d.conv.bias"),
    ("mixer.x_proj.", "mixer.ssm.x_proj."),
    ("mixer.dt_proj.", "mixer.ssm.dt_proj."),
    ("mixer.A_log", "mixer.ssm.A_log"),
    ("mixer.D", "mixer.ssm.D"),
)


def convert_from_reference(
    reference_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Rename keys from the official Mamba release to this repository's layout.

    Args
    ----
    reference_state_dict:
        A state dict using the official ``mamba_ssm`` naming, e.g.
        ``backbone.layers.0.mixer.conv1d.weight``,
        ``backbone.layers.0.mixer.A_log``.

    Returns
    -------
    dict
        A state dict with keys remapped to match
        :class:`~mamba.models.lm_head.MambaLMHeadModel` (e.g.
        ``...mixer.conv1d.conv.weight``, ``...mixer.ssm.A_log``).

    Notes
    -----
    The two architectures are weight-compatible; only the module nesting
    differs. Tensor values and shapes are passed through unchanged.
    """
    converted: dict[str, torch.Tensor] = {}
    for key, value in reference_state_dict.items():
        new_key = key
        for src, dst in _REFERENCE_RENAMES:
            if src in new_key:
                new_key = new_key.replace(src, dst)
                break
        converted[new_key] = value
    return converted
