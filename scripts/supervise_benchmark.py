#!/usr/bin/env python
"""
Parallax OOM-Resilient Benchmark Supervisor
===========================================
Runs the benchmark runner as a child process and restarts it from checkpoints when it
dies from memory exhaustion, halving --n-jobs on every OOM restart.

Every stage of the runner is checkpointed (widening, per-country blocking, candidate and
feature caches, per-config sweep results, per-fold CV results), so a restart only redoes
the stage that was interrupted.

Events are appended one per line to --events (default reports/supervisor_events.log):
    <iso-time> START|OOM_RETRY|CRASH|DONE|GAVE_UP key=value ...
so an external watcher can react without parsing the full log.

Usage:
    python scripts/supervise_benchmark.py --n-jobs 80 -- --data-dir data/medium_split_200k \
        --output-dir output_par --reports-dir reports_par
Everything after "--" is passed to parallax.experiments.runner (do not pass --n-jobs there).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

OOM_MARKERS = (
    "MemoryError",
    "BrokenProcessPool",
    "Cannot allocate memory",
    "std::bad_alloc",
    "Killed",
    "out of memory",
)


def emit(events: Path, kind: str, **fields: object) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    extra = " ".join(f"{k}={str(v).replace(chr(10), ' | ')}" for k, v in fields.items())
    line = f"{ts} {kind} {extra}".rstrip()
    with events.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(f"[supervisor] {line}", flush=True)


def run_once(cmd: list[str], log_path: Path, env: dict[str, str]) -> tuple[int, str]:
    """Run cmd, tee output to stdout and log_path, return (returncode, tail text)."""
    tail: deque[str] = deque(maxlen=60)
    with log_path.open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
            tail.append(line)
        rc = proc.wait()
    return rc, "".join(tail)


def main() -> int:
    argv = sys.argv[1:]
    runner_args: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, runner_args = argv[:i], argv[i + 1 :]

    ap = argparse.ArgumentParser(description="OOM-resilient supervisor for the Parallax runner")
    ap.add_argument("--n-jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--min-jobs", type=int, default=4)
    ap.add_argument("--max-restarts", type=int, default=4)
    ap.add_argument("--log", default="reports/supervised_run.log")
    ap.add_argument("--events", default="reports/supervisor_events.log")
    args = ap.parse_args(argv)

    log_path = Path(args.log)
    events = Path(args.events)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    events.parent.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(project_root / "src") + os.pathsep + env.get("PYTHONPATH", "")

    n_jobs = max(1, args.n_jobs)
    restarts = 0
    first = True
    t_start = time.time()
    while True:
        cmd_args = list(runner_args)
        if not first:
            # Never wipe checkpoints on a restart.
            cmd_args = [a for a in cmd_args if a != "--reset-checkpoints"]
        cmd = [sys.executable, "-u", "-m", "parallax.experiments.runner", *cmd_args]
        cmd += ["--n-jobs", str(n_jobs)]
        emit(events, "START", attempt=restarts + 1, n_jobs=n_jobs, pid=os.getpid())
        t0 = time.time()
        rc, tail = run_once(cmd, log_path, env)
        first = False
        elapsed = time.time() - t0

        if rc == 0:
            emit(
                events,
                "DONE",
                attempt=restarts + 1,
                n_jobs=n_jobs,
                attempt_sec=round(elapsed, 1),
                total_sec=round(time.time() - t_start, 1),
            )
            return 0

        is_oom = rc in (-9, 137) or any(m in tail for m in OOM_MARKERS)
        if not is_oom:
            last = [ln.strip() for ln in tail.strip().splitlines()[-5:]]
            emit(events, "CRASH", rc=rc, attempt_sec=round(elapsed, 1), tail=" | ".join(last))
            return rc if rc > 0 else 1

        restarts += 1
        if restarts > args.max_restarts or n_jobs <= args.min_jobs:
            emit(events, "GAVE_UP", rc=rc, restarts=restarts - 1, n_jobs=n_jobs)
            return rc if rc > 0 else 1
        n_jobs = max(args.min_jobs, n_jobs // 2)
        emit(events, "OOM_RETRY", rc=rc, next_n_jobs=n_jobs, restarts=restarts)
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
