"""
Parallax OOM-Resilient Process Pool Utilities
=============================================
Shared helpers for every multi-process stage of the pipeline:

- Fork-context pools so large read-only state is shared via copy-on-write.
- ``gc.freeze()`` before forking so the cyclic GC does not touch (and therefore
  copy) every inherited object page in the children.
- Workers raise their own ``oom_score_adj`` so the kernel OOM killer picks a
  worker instead of the parent orchestrator; the parent then sees
  ``BrokenProcessPool`` and retries with half the workers.
- Per-worker OpenMP/BLAS thread caps to avoid thread oversubscription.
- Memory-aware worker sizing from currently available RAM.
"""

from __future__ import annotations

import gc
import multiprocessing
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, TypeVar

T = TypeVar("T")
R = TypeVar("R")

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def total_threads() -> int:
    """Total CPU threads the pipeline may use (env PARALLAX_TOTAL_THREADS overrides)."""
    env = os.environ.get("PARALLAX_TOTAL_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return os.cpu_count() or 1


def available_memory_gb() -> float | None:
    """Currently available system RAM in GB (psutil, else /proc/meminfo), or None."""
    try:
        import psutil

        return float(psutil.virtual_memory().available) / (1024**3)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / (1024**2)
    except OSError:
        pass
    return None


def resolve_workers(
    requested: int,
    per_worker_gb: float = 0.0,
    reserve_gb: float = 8.0,
    n_tasks: int | None = None,
) -> int:
    """
    Clamp a requested worker count by task count and by available memory.
    per_worker_gb is the estimated private (non-shared) memory per worker.
    """
    workers = max(1, int(requested))
    if n_tasks is not None:
        workers = max(1, min(workers, n_tasks))
    if per_worker_gb > 0:
        avail = available_memory_gb()
        if avail is not None:
            fit = int(max(avail - reserve_gb, per_worker_gb) // per_worker_gb)
            workers = max(1, min(workers, fit))
    return workers


def chunk_bounds(n_items: int, n_chunks: int) -> list[tuple[int, int]]:
    """Split range(n_items) into up to n_chunks contiguous, near-equal (start, end) bounds."""
    if n_items <= 0:
        return []
    n_chunks = max(1, min(n_chunks, n_items))
    size = -(-n_items // n_chunks)
    return [(i, min(i + size, n_items)) for i in range(0, n_items, size)]


def _worker_init(threads_per_worker: int) -> None:
    """Initializer run in every child process."""
    # Prefer killing workers over the parent when the box runs out of memory.
    try:
        with open("/proc/self/oom_score_adj", "w") as fh:
            fh.write("1000")
    except OSError:
        pass
    t = str(max(1, threads_per_worker))
    for var in _THREAD_ENV_VARS:
        os.environ[var] = t


def fork_available() -> bool:
    return "fork" in multiprocessing.get_all_start_methods()


def run_pool(
    fn: Callable[[T], R],
    tasks: Sequence[T],
    max_workers: int,
    threads_per_worker: int = 1,
    max_retries: int = 3,
    label: str = "pool",
    on_result: Callable[[int, R], None] | None = None,
) -> list[R]:
    """
    Run fn over tasks in a fork-context process pool and return results in task order.

    Tasks must be idempotent. If a worker is killed (typically by the OOM killer), the
    pool is torn down, memory is reclaimed, and the unfinished tasks are retried with
    half as many workers, up to max_retries times.
    """
    n = len(tasks)
    if n == 0:
        return []
    results: list[Any] = [None] * n
    done = [False] * n

    if max_workers <= 1 or n == 1 or not fork_available():
        for i, task in enumerate(tasks):
            results[i] = fn(task)
            if on_result is not None:
                on_result(i, results[i])
        return results

    ctx = multiprocessing.get_context("fork")
    workers = max(1, min(max_workers, n))
    attempt = 0
    while True:
        pending = [i for i in range(n) if not done[i]]
        if not pending:
            return results
        gc.collect()
        gc.freeze()
        try:
            with ProcessPoolExecutor(
                max_workers=min(workers, len(pending)),
                mp_context=ctx,
                initializer=_worker_init,
                initargs=(threads_per_worker,),
            ) as executor:
                futures = {executor.submit(fn, tasks[i]): i for i in pending}
                from concurrent.futures import as_completed

                for fut in as_completed(futures):
                    i = futures[fut]
                    results[i] = fut.result()
                    done[i] = True
                    if on_result is not None:
                        on_result(i, results[i])
            return results
        except BrokenProcessPool:
            attempt += 1
            remaining = sum(1 for d in done if not d)
            if attempt > max_retries or workers <= 1:
                print(
                    f"  ❌ [{label}] Worker killed (likely OOM); retries exhausted "
                    f"with {remaining} tasks left.",
                    flush=True,
                )
                raise
            workers = max(1, workers // 2)
            print(
                f"  ⚠️  [{label}] Worker killed (likely OOM). Retrying {remaining} tasks "
                f"with {workers} workers (attempt {attempt}/{max_retries})...",
                flush=True,
            )
            gc.unfreeze()
            gc.collect()
            time.sleep(2.0)
        finally:
            gc.unfreeze()
