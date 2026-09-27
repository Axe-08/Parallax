"""Tests for OOM-resilient pools and the vectorized Macro F0.5 threshold scan."""

from __future__ import annotations

import os
import signal
from pathlib import Path

import numpy as np
import pytest

from parallax.metrics.evaluator import evaluate_resolution_predictions, macro_f05_threshold_scan
from parallax.utils.parallel import chunk_bounds, fork_available, resolve_workers, run_pool

_MARKER: Path | None = None


def _square_or_die_once(x: int) -> int:
    """Kill the worker (like the OOM killer) the first time task 3 runs."""
    assert _MARKER is not None
    if x == 3 and not _MARKER.exists():
        _MARKER.write_text("killed")
        os.kill(os.getpid(), signal.SIGKILL)
    return x * x


@pytest.mark.skipif(not fork_available(), reason="fork start method required")
def test_run_pool_retries_after_worker_sigkill(tmp_path: Path) -> None:
    global _MARKER
    _MARKER = tmp_path / "marker"
    out = run_pool(_square_or_die_once, list(range(8)), max_workers=4, label="test")
    assert out == [i * i for i in range(8)]
    assert _MARKER.exists()


def test_chunk_bounds_cover_range() -> None:
    b = chunk_bounds(10, 3)
    assert b[0][0] == 0 and b[-1][1] == 10
    assert sum(e - s for s, e in b) == 10
    assert chunk_bounds(0, 4) == []
    assert len(chunk_bounds(3, 10)) == 3


def test_resolve_workers_clamps() -> None:
    assert resolve_workers(8, n_tasks=3) == 3
    assert resolve_workers(0) == 1
    assert resolve_workers(4, per_worker_gb=1e9) == 1


def test_threshold_scan_matches_reference_evaluator() -> None:
    rng = np.random.default_rng(0)
    gt: dict[str, set[str]] = {}
    s1s: list[str] = []
    cands: list[str] = []
    for i in range(300):
        s1 = f"S1-{i}"
        n_true = int(rng.integers(0, 4))
        gt[s1] = {f"T-{i}-{j}" for j in range(n_true)}
        pool = [f"T-{i}-{j}" for j in range(n_true + 3)]  # includes blocking-missed + distractors
        for c in pool[1:] if n_true > 1 else pool:  # drop one true match sometimes
            s1s.append(s1)
            cands.append(c)
    s1_arr = np.array(s1s)
    is_true = np.array([c in gt[s] for s, c in zip(s1s, cands, strict=True)])
    probs = np.clip(is_true * 0.6 + rng.random(len(s1s)) * 0.5, 0, 1)
    taus = [0.3, 0.5, 0.62, 0.8, 0.95]
    fast = macro_f05_threshold_scan(s1_arr, is_true, probs, gt, taus)
    for tau, f in zip(taus, fast, strict=True):
        preds: dict[str, set[str]] = {s: set() for s in gt}
        for s, c, p in zip(s1s, cands, probs, strict=True):
            if p >= tau:
                preds[s].add(c)
        ref = evaluate_resolution_predictions(gt, preds).macro_f05
        assert f == pytest.approx(ref, abs=1e-12)
