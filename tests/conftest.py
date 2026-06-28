"""Shared pytest fixtures for the Mamba test suite.

These fixtures provide deterministic randomness, a small fast model config, a
batched input factory, and parametrization over dtypes and sequence lengths.
A session-scoped check also imports every package module to guarantee the
dependency graph contains no circular imports.
"""

from __future__ import annotations

import importlib
import pkgutil
import random
from typing import Callable

import numpy as np
import pytest
import torch

import mamba
from mamba.config import MambaConfig


@pytest.fixture(autouse=True)
def rng_seed() -> int:
    """Seed ``torch``, ``numpy`` and ``random`` before every test.

    Autouse so that *all* tests are deterministic without having to request the
    fixture explicitly. Returns the seed in case a test wants to re-seed.
    """
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


@pytest.fixture
def device() -> torch.device:
    """The compute device: CUDA when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def small_config() -> MambaConfig:
    """A tiny config (``d_model=64``, ``n_layers=2``, ``d_state=8``) for fast tests."""
    return MambaConfig(
        d_model=64,
        n_layers=2,
        d_state=8,
        vocab_size=256,
        pad_vocab_size_multiple=8,
    )


@pytest.fixture
def batch_input() -> Callable[..., torch.Tensor]:
    """Factory producing a random ``float32`` tensor of shape ``(B, L, D)``.

    Usage::

        def test_x(batch_input):
            x = batch_input(B=2, L=16, D=64)
    """

    def _make(
        B: int = 2,
        L: int = 16,
        D: int = 64,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        return torch.randn(B, L, D, dtype=dtype, device=device)

    return _make


@pytest.fixture(params=[torch.float32, torch.float16, torch.bfloat16])
def dtype_cases(request: pytest.FixtureRequest) -> torch.dtype:
    """Parametrize a test over the three supported floating dtypes."""
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(params=[1, 7, 16, 128, 1024])
def seqlen_cases(request: pytest.FixtureRequest) -> int:
    """Parametrize a test over a spread of sequence lengths."""
    return request.param  # type: ignore[no-any-return]


@pytest.fixture
def legs_recon_mse() -> Callable[..., float]:
    """Return a helper measuring HiPPO-LegS function-reconstruction MSE.

    The helper integrates the scaled LegS ODE ``c' = (1/t)(A c + B u)`` with a
    midpoint (RK2) step to obtain the projection coefficients, reconstructs the
    input from the normalised Legendre basis, and returns the interior MSE.
    Exercising the actual :func:`mamba.core.hippo.hippo_legs` matrices, it
    verifies that the HiPPO basis approximates smooth signals -- with quality
    improving as the state dimension ``N`` grows.
    """
    from numpy.polynomial import legendre as _leg

    from mamba.core.hippo import hippo_legs

    def _mse(signal: np.ndarray, n: int, trim: int = 100) -> float:
        a, b = hippo_legs(n)
        a_np = a.double().numpy()
        b_np = b.double().numpy()
        length = len(signal)
        c = np.zeros(n)
        dt = 1.0 / length
        for k in range(1, length + 1):
            t = k * dt
            f1 = (1.0 / t) * (a_np @ c + b_np * signal[k - 1])
            c_half = c + 0.5 * dt * f1
            f2 = (1.0 / t) * (a_np @ c_half + b_np * signal[k - 1])
            c = c + dt * f2
        x = np.linspace(0.0, 1.0, length)
        xx = 2.0 * x - 1.0
        recon = np.zeros(length)
        for j in range(n):
            coef = np.zeros(j + 1)
            coef[j] = 1.0
            recon += c[j] * np.sqrt(2 * j + 1) * _leg.legval(xx, coef)
        return float(np.mean((recon[trim:-trim] - signal[trim:-trim]) ** 2))

    return _mse


def test_no_circular_imports() -> None:
    """Importing every ``mamba`` submodule must not raise (no import cycles).

    Walks the package tree and imports each module. A circular import would
    surface here as an :class:`ImportError`, failing the test.
    """
    failures: list[str] = []
    for mod in pkgutil.walk_packages(mamba.__path__, prefix="mamba."):
        try:
            importlib.import_module(mod.name)
        except Exception as exc:  # noqa: BLE001 - we want to report any failure
            failures.append(f"{mod.name}: {exc}")
    assert not failures, "module import failures:\n" + "\n".join(failures)
