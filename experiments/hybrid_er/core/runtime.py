"""
Device selection and structured experiment records.

Every experiment writes a run.json with the commit, populations, seeds, configuration,
per-stage timings, device and peak VRAM, so results never live only in terminal logs.
"""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def select_device(requested: str = "auto") -> str:
    """
    'cpu' -> cpu; an integer string -> that CUDA device; 'auto' -> the CUDA device with the
    most free memory (cpu if CUDA is unavailable).
    """
    import torch

    if requested == "cpu" or not torch.cuda.is_available():
        if requested not in ("cpu", "auto"):
            raise RuntimeError(f"GPU {requested} requested but CUDA is not available.")
        return "cpu"
    if requested != "auto":
        idx = int(requested)
        if idx >= torch.cuda.device_count():
            raise RuntimeError(f"GPU {idx} requested but only {torch.cuda.device_count()} devices exist.")
        return f"cuda:{idx}"

    free = _nvidia_smi_free_mib()
    if free is None:  # fall back to CUDA queries (creates a context per device)
        free = {}
        for i in range(torch.cuda.device_count()):
            try:
                free[i] = torch.cuda.mem_get_info(i)[0] / 2**20
            except RuntimeError as exc:  # a completely full shared GPU cannot even host a context
                logger.warning("GPU %d unavailable: %s", i, exc)
    if not free:
        raise RuntimeError("No usable CUDA device found.")
    for i, f in sorted(free.items()):
        logger.info("GPU %d: %.2f GiB free", i, f / 1024)
    best = max(free, key=free.get)
    logger.info("Selected cuda:%d (%.2f GiB free)", best, free[best] / 1024)
    return f"cuda:{best}"


def _nvidia_smi_free_mib() -> dict[int, float] | None:
    """Free MiB per visible GPU without creating CUDA contexts (None if nvidia-smi is unusable)."""
    import os

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=30,
        ).stdout
    except Exception:
        return None
    phys = {int(a): float(b) for a, b in (line.split(",") for line in out.strip().splitlines())}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return phys
    try:
        order = [int(x) for x in visible.split(",") if x.strip()]
    except ValueError:  # UUID-style masks: let torch decide
        return None
    return {logical: phys[p] for logical, p in enumerate(order) if p in phys}


def device_info(device: str) -> dict[str, Any]:
    info: dict[str, Any] = {"device": device}
    try:
        import torch

        info["torch"] = torch.__version__
        if device.startswith("cuda"):
            idx = int(device.split(":")[1]) if ":" in device else 0
            props = torch.cuda.get_device_properties(idx)
            free_b, total_b = torch.cuda.mem_get_info(idx)
            info.update(
                gpu_name=props.name,
                gpu_total_gib=round(total_b / 2**30, 2),
                gpu_free_gib_at_start=round(free_b / 2**30, 2),
            )
    except Exception as exc:  # informational only
        info["device_info_error"] = repr(exc)
    return info


def peak_vram_gib(device: str) -> float | None:
    try:
        import torch

        if device.startswith("cuda"):
            return round(torch.cuda.max_memory_allocated(device) / 2**30, 3)
    except Exception:
        pass
    return None


def git_info() -> dict[str, Any]:
    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "sha": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_tracked_files": [line[3:] for line in status.splitlines()],
    }


class RunRecorder:
    """Collects configuration, stage timings and results; writes <out_dir>/<run_id>/run.json."""

    def __init__(self, name: str, out_root: Path | str, config: dict[str, Any], device: str = "cpu") -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{name}_{stamp}"
        self.run_dir = Path(out_root) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self._t0 = time.time()
        self.record: dict[str, Any] = {
            "run_id": self.run_id,
            "name": name,
            "started_utc": stamp,
            "git": git_info(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "device": device_info(device),
            "config": config,
            "stages": {},
            "results": {},
        }

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        t = time.time()
        logger.info("[stage] %s ...", name)
        try:
            yield
        finally:
            dt = round(time.time() - t, 2)
            self.record["stages"][name] = {"seconds": dt}
            logger.info("[stage] %s done in %.1fs", name, dt)

    def put(self, key: str, value: Any) -> None:
        self.record["results"][key] = value
        self.save()

    def save(self) -> Path:
        self.record["runtime_seconds"] = round(time.time() - self._t0, 2)
        self.record["peak_vram_gib"] = peak_vram_gib(self.device)
        path = self.run_dir / "run.json"
        path.write_text(json.dumps(self.record, indent=2, default=str), encoding="utf-8")
        return path
