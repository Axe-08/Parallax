# ⚡ Parallax: DGX-1 High-Performance Parallelization Guide & Playbook

> **Target Environment:** NVIDIA DGX-1 Compute Node (`sid-dgx`)  
> **Hardware Topology:** Dual Intel Xeon E5-2698 v4 @ 2.20 GHz (2 Sockets × 20 Physical Cores = 40 Cores / 80 Threads)  
> **System Memory:** 503 GB DDR4 RAM (Dual NUMA Nodes: Node 0 & Node 1)  
> **Objective:** Maximize multi-core throughput and minimize execution latency without deadlocks, memory explosion, or NUMA thread thrashing.

---

## 🧭 Executive Summary: The 5 Golden Invariants

When transitioning single-threaded Python scripts to the DGX-1, follow these five non-negotiable principles:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                   DGX-1 HIGH-PERFORMANCE INVARIANTS                          │
├───────────────────────────────┬──────────────────────────────────────────────┤
│ 1. Zero-Copy IPC via `fork`   │ Share massive read-only lookups via module   │
│                               │ globals + Linux Copy-on-Write (COW).         │
│ 2. NUMA Thread Bounding       │ Workers × OpenMP Threads ≤ 40 Physical Cores │
│                               │ (Never let OpenMP spawn 80 threads per child)│
│ 3. Slice-Based DF Chunking    │ Never use `np.array_split(df)` in NumPy 2.x; │
│                               │ Use index-slice chunking on `.iloc[]`.       │
│ 4. Garbage Collection Hygiene │ Reset global pointers in `finally:` blocks    │
│                               │ and run `gc.collect()` before next phase.    │
│ 5. Detached Execution         │ Run via `nohup` or `tmux` with unbuffered    │
│                               │ output (`PYTHONUNBUFFERED=1`).               │
└────────────────────────────────┴──────────────────────────────────────────────┘
```

---

## 🔬 1. Understanding DGX-1 Hardware & Pitfalls

### Dual-Socket NUMA Topology
The DGX-1 CPU subsystem consists of two physical sockets:
* **NUMA Node 0:** CPUs 0–19 (physical), 40–59 (hyperthreads)
* **NUMA Node 1:** CPUs 20–39 (physical), 60–79 (hyperthreads)
* **Memory:** ~251 GB attached locally to each socket.

### Pitfall #1: The OpenMP / LightGBM NUMA Meltdown
If a library (LightGBM, XGBoost, PyTorch, OpenBLAS, MKL) detects 80 logical cores, its default setting is to spawn **80 OpenMP threads**.
* When 80 threads run across two NUMA sockets, they constantly invalidate CPU cache lines across the inter-socket QPI/UPI bus.
* If you run **5 concurrent worker processes** (e.g. 5 CV folds) and each spawns 80 threads, the OS tries to schedule **400 competing threads**. CPU usage jumps to 100%, but actual progress drops to near zero due to lock contention and memory stall.
* **The Rule:** Always explicitly bound OpenMP threads:
  $$\text{Workers} \times \text{Threads per Worker} \le 40 \quad (\text{Physical Core Count})$$
  * Example for 5 Folds: `workers = 5`, `num_threads = 8` ($5 \times 8 = 40$).
  * Example for 10 Workers: `workers = 10`, `num_threads = 4` ($10 \times 4 = 40$).

### Pitfall #2: IPC Pickling vs. Copy-on-Write (COW)
In Python's `multiprocessing`, passing large objects (like a 1,000,000-entity dictionary or a 500 MB sparse matrix) into `pool.map(worker_fn, data)` via standard IPC requires:
1. **Serializing (pickling)** the entire structure in the parent process.
2. Sending gigabytes over Unix domain pipes.
3. **Deserializing (unpickling)** in each of the 20–40 worker processes.
4. **Memory Multiplier:** 40 workers $\times$ 2 GB = 80 GB RAM wasted just on duplicate IPC payloads, with tens of seconds of CPU stall.
* **The Solution:** Use `multiprocessing.get_context("fork")` and module-level global variables. Linux creates child processes with copy-on-write virtual memory pages. Child workers read from the exact same physical RAM pages as the parent with **zero serialization overhead** and **zero memory duplication**.

---

## 🛠️ 2. Production Code Patterns

### Pattern A: CPU-Bound DataFrame Transformation (Preprocessing / Normalization)
Use this pattern when applying row-by-row cleaning, regex matching, or transliteration across millions of records.

```python
from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd


def _worker_process_chunk(chunk_df: pd.DataFrame) -> pd.DataFrame:
    """Worker function executed inside child process."""
    res = chunk_df.copy()
    # Apply CPU-intensive operations
    res["clean_name"] = res["business_name"].str.lower().str.strip()
    res["token_len"] = res["clean_name"].apply(lambda s: len(s.split()))
    return res


def parallel_transform_df(
    df: pd.DataFrame,
    n_jobs: int = 20,
    min_chunk_threshold: int = 5000,
) -> pd.DataFrame:
    """
    Parallelize DataFrame transformations across n_jobs processes.
    Safe against NumPy 2.x array_split deprecations.
    """
    total_len = len(df)
    if n_jobs <= 1 or total_len < min_chunk_threshold:
        return _worker_process_chunk(df)

    effective_workers = min(n_jobs, total_len)
    chunk_size = int(np.ceil(total_len / effective_workers))
    
    # Safe DataFrame slicing (never use np.array_split on DataFrame in NumPy 2.x)
    chunks = [
        df.iloc[i : i + chunk_size].copy()
        for i in range(0, total_len, chunk_size)
    ]
    chunks = [c for c in chunks if len(c) > 0]

    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx) as executor:
        processed_chunks = list(executor.map(_worker_process_chunk, chunks))

    return pd.concat(processed_chunks, ignore_index=True)
```

---

### Pattern B: High-Throughput Read-Only Lookups (Zero-Copy COW Blocker & Feature Extractor)
Use this pattern when workers must score query items against a massive target lookup dictionary, TF-IDF matrix, or indexed target records.

```python
from __future__ import annotations

import functools
import gc
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Any
import numpy as np
import pandas as pd

# Module-level globals for Zero-Copy Copy-On-Write IPC
_GLOBAL_TARGET_LOOKUP: dict[str, Any] | None = None
_GLOBAL_MODEL_CONFIG: dict[str, Any] | None = None


def _worker_extract_features(query_keys: list[str]) -> list[dict[str, Any]]:
    """Child worker directly reads module globals with ZERO copy overhead."""
    global _GLOBAL_TARGET_LOOKUP, _GLOBAL_MODEL_CONFIG
    assert _GLOBAL_TARGET_LOOKUP is not None

    records: list[dict[str, Any]] = []
    for q_id in query_keys:
        # Direct dictionary read from parent memory
        target_info = _GLOBAL_TARGET_LOOKUP.get(q_id)
        if target_info:
            records.append({
                "query_id": q_id,
                "score": len(target_info),
            })
    return records


def parallel_feature_extraction(
    query_ids: list[str],
    large_target_lookup: dict[str, Any],
    model_config: dict[str, Any],
    n_jobs: int = 20,
) -> pd.DataFrame:
    """
    Extracts features across worker processes utilizing Linux fork zero-copy memory.
    """
    global _GLOBAL_TARGET_LOOKUP, _GLOBAL_MODEL_CONFIG
    
    # 1. Anchor large data in module globals before fork
    _GLOBAL_TARGET_LOOKUP = large_target_lookup
    _GLOBAL_MODEL_CONFIG = model_config

    # 2. Partition query keys into chunks
    chunks = [list(c) for c in np.array_split(query_ids, n_jobs) if len(c) > 0]

    ctx = mp.get_context("fork")
    try:
        with ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx) as executor:
            chunk_results = list(executor.map(_worker_extract_features, chunks))
    finally:
        # 3. CRITICAL: Clean up module globals to avoid memory leaks
        _GLOBAL_TARGET_LOOKUP = None
        _GLOBAL_MODEL_CONFIG = None
        gc.collect()

    flat_records = [item for sublist in chunk_results for item in sublist]
    return pd.DataFrame(flat_records)
```

---

### Pattern C: Concurrent Cross-Validation & OpenMP Thread Capping
Use this pattern when training GBDTs (LightGBM, CatBoost, XGBoost) across multiple cross-validation folds.

```python
from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
import lightgbm as lgb
import pandas as pd


def evaluate_single_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: list[str],
    num_threads: int,
) -> dict[str, float]:
    """Train LightGBM on a single fold with strictly capped OpenMP threads."""
    X_train, y_train = train_df[features], train_df["target"]
    X_val, y_val = val_df[features], val_df["target"]

    # CRITICAL: Bound num_threads to prevent NUMA thread thrashing
    clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.03,
        num_threads=num_threads,  # Capped thread count per worker
        verbosity=-1,
        random_state=42 + fold_idx,
    )
    clf.fit(X_train, y_train)
    val_preds = clf.predict_proba(X_val)[:, 1]
    
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y_val, val_preds))
    return {"fold": fold_idx, "val_auc": auc}


def run_parallel_cv(
    folds: list[int],
    df: pd.DataFrame,
    features: list[str],
    n_workers: int = 5,
) -> list[dict[str, float]]:
    """
    Executes K-fold cross validation concurrently.
    Allocates 40 physical cores across workers.
    """
    # 40 physical cores total on DGX-1
    threads_per_worker = max(1, 40 // n_workers)
    print(f"⚡ Launching {n_workers} fold workers (threads/worker={threads_per_worker})")

    ctx = mp.get_context("fork")
    results = []

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers, mp_context=ctx
    ) as executor:
        futures = {
            executor.submit(
                evaluate_single_fold,
                k,
                df[df["fold"] != k],
                df[df["fold"] == k],
                features,
                threads_per_worker,
            ): k
            for k in folds
        }
        for future in concurrent.futures.as_completed(futures):
            fold_idx = futures[future]
            res = future.result()
            print(f"  ✓ Fold {fold_idx} finished: Val AUC = {res['val_auc']:.4f}")
            results.append(res)

    return results
```

---

## ⚙️ 3. Resource Allocation Matrix for DGX-1

Use this quick-reference table to select worker and thread counts based on pipeline stage:

| Stage / Task | Recommended Workers (`n_jobs`) | Threads per Worker | Memory Pattern | Typical Throughput |
|---|:---:|:---:|---|---|
| **Text Normalization / Regex** | `20` – `30` | `1` | Slice DataFrame Chunks | 150k–200k rows/sec |
| **TF-IDF Blocking / Cosine Sim** | `10` – `20` | `2` | Global Target Matrix (COW) | 1,000 queries/sec against 1M targets |
| **Feature Extraction (Levenshtein, Jaro, Exact)** | `20` – `40` | `1` | Global Lookups (COW) | 35k–50k pairs/sec |
| **5-Fold Cross Validation** | `5` | `8` | Forked S1/Target Cache | 5 models trained concurrently |
| **LightGBM Grid Search** | `10` | `4` | Forked CV Array | 10 configs evaluated simultaneously |

---

## 📊 4. Monitoring & Diagnostic Commands

When running long compute jobs on `sid-dgx`, use these commands to ensure hardware is fully utilized and not stalling:

### 1. View Per-Process CPU and Memory Sorted by Utilization
```bash
ps -u $USER -o pid,%cpu,%mem,time,cmd --sort=-%cpu | head -n 25
```
* **Healthy State:** Each worker process shows ~100% CPU (or ~800% if multi-threaded with 8 threads).
* **Contention State:** CPU usage drops to 5–15% across many processes with high `system` CPU (visible in `htop`). Indicates thread lock contention.

### 2. Check Core-by-Core Saturation
```bash
mpstat -P ALL 2 3
```
* Shows utilization across all 80 logical cores. Ensure work is evenly spread across sockets.

### 3. Check NUMA Node Allocation
```bash
numastat -c python
```
* Confirms memory allocations are balanced between Node 0 and Node 1.

### 4. Running Detached Background Jobs
Always use `PYTHONUNBUFFERED=1` so logs flush in real-time to your log file:
```bash
PYTHONUNBUFFERED=1 nohup /home/23dcs510/miniconda3/envs/py310/bin/python \
    scripts/run_medium_benchmark.py > run_200k.log 2>&1 &

# Monitor in real time:
tail -f run_200k.log
```

### 5. Emergency Process Cleanup
If child worker processes ever get orphaned:
```bash
# Graceful terminate
pkill -u $USER -f python

# Force kill if hung
kill -9 $(pgrep -u $USER -f python)
```

---

## 📋 5. Pre-Flight Checklist Before Running on DGX-1

- [ ] **Start Method:** Context explicitly specified as `multiprocessing.get_context("fork")`.
- [ ] **NUMA Thread Bound:** LightGBM `num_threads` or OpenMP environment variable capped (`workers * threads <= 40`).
- [ ] **No `np.array_split` on DataFrames:** Slices created with `df.iloc[start:end].copy()`.
- [ ] **Zero-Copy Module Globals:** Target lookups stored in module globals before executor launch, set to `None` in `finally:`.
- [ ] **Garbage Collection:** Explicit `gc.collect()` called after large memory release.
- [ ] **Unbuffered Logging:** Script executed with `PYTHONUNBUFFERED=1`.
