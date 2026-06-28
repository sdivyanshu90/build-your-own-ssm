"""Micro-benchmarks for the selective scan implementations.

Run explicitly with::

    pytest tests/benchmark/ -m slow --benchmark-only

These compare the sequential reference scan against the parallel associative
scan across sequence lengths. They are tagged ``slow`` and excluded from the
default test run (use ``-m "not slow"`` in CI).
"""

from __future__ import annotations

import pytest
import torch

from mamba.ops.selective_scan_naive import selective_scan_naive
from mamba.ops.selective_scan_parallel import selective_scan_parallel

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _inputs(length: int, batch: int = 1, d_inner: int = 16, d_state: int = 8) -> dict:
    torch.manual_seed(0)
    return {
        "u": torch.randn(batch, length, d_inner, device=_DEVICE),
        "delta": torch.randn(batch, length, d_inner, device=_DEVICE),
        "A": -torch.rand(d_inner, d_state, device=_DEVICE) - 0.1,
        "B": torch.randn(batch, length, d_state, device=_DEVICE),
        "C": torch.randn(batch, length, d_state, device=_DEVICE),
        "D": torch.randn(d_inner, device=_DEVICE),
    }


@pytest.mark.slow
@pytest.mark.benchmark(group="scan")
def test_bench_naive_scan(benchmark: object) -> None:
    """Benchmark the sequential reference scan at L=2048."""
    ins = _inputs(2048)

    def run() -> None:
        selective_scan_naive(**ins, delta_softplus=True)
        if _DEVICE.type == "cuda":
            torch.cuda.synchronize()

    benchmark(run)  # type: ignore[operator]


@pytest.mark.slow
@pytest.mark.benchmark(group="scan")
def test_bench_parallel_scan(benchmark: object) -> None:
    """Benchmark the parallel associative scan at L=2048."""
    ins = _inputs(2048)

    def run() -> None:
        selective_scan_parallel(**ins, delta_softplus=True)
        if _DEVICE.type == "cuda":
            torch.cuda.synchronize()

    benchmark(run)  # type: ignore[operator]


@pytest.mark.slow
@pytest.mark.benchmark(group="scan")
@pytest.mark.parametrize("seqlen", [128, 512, 2048, 8192])
def test_bench_scan_vs_seqlen(benchmark: object, seqlen: int) -> None:
    """Benchmark the parallel scan across a range of sequence lengths."""
    ins = _inputs(seqlen)

    def run() -> None:
        selective_scan_parallel(**ins, delta_softplus=True)
        if _DEVICE.type == "cuda":
            torch.cuda.synchronize()

    benchmark(run)  # type: ignore[operator]
