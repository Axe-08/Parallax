"""
Country-partitioned dense retrieval over S2/S3 with configurable FAISS backends.

Backends (faiss index_factory strings, inner product on L2-normalized vectors):

    torch     -> exact brute force, chunked fp16 matmul on a CUDA device (CPU-contention-proof)
    flat      -> "Flat"              exact; fine for 5K/research scale
    sq8       -> "SQ8"               4x smaller than fp32, near-exact
    ivf       -> "IVF{nlist},Flat"   approximate, needs training + nprobe
    ivfpq     -> "IVF{nlist},PQ{m}"  approximate + compressed

The choice for full-scale retrieval must be made from measured memory / recall / latency;
`DenseIndex.stats()` reports the serialized size of each partition so this can be measured
rather than assumed.

Country is an open-set field: one partition per distinct target country value. An S1 whose
country has no target partition falls back to searching every partition (never dropped
silently, and counted in `fallback_queries`).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import faiss
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def factory_string(backend: str, n: int, nlist: int | None = None, pq_m: int = 64) -> str:
    if backend == "flat":
        return "Flat"
    if backend == "sq8":
        return "SQ8"
    nlist = nlist or max(1, min(int(4 * np.sqrt(n)), n // 39))
    if backend == "ivf":
        return f"IVF{nlist},Flat"
    if backend == "ivfpq":
        return f"IVF{nlist},PQ{pq_m}"
    raise ValueError(f"Unknown dense index backend {backend!r}")


class _TorchFlat:
    """Exact inner-product search: fp16 corpus on device, chunked matmul + topk."""

    def __init__(self, x: np.ndarray, device: str, chunk: int = 1024) -> None:
        import torch

        self.torch = torch
        self.device = device
        self.chunk = chunk
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device, self.dtype)
        self.ntotal = int(self.x.shape[0])
        self.nbytes = int(self.x.element_size() * self.x.nelement())

    def search(self, q: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        ds, js = [], []
        with torch.inference_mode():
            for i in range(0, len(q), self.chunk):
                qt = torch.from_numpy(np.ascontiguousarray(q[i : i + self.chunk], dtype=np.float32)).to(self.device, self.dtype)
                d, j = torch.topk((qt @ self.x.T).float(), k=k, dim=1)
                ds.append(d.cpu().numpy())
                js.append(j.cpu().numpy())
        return np.concatenate(ds), np.concatenate(js)


@dataclass
class DenseIndex:
    """One FAISS index per (country, source) partition, so per-source top-k is exact."""

    backend: str = "flat"
    nprobe: int = 32
    nlist: int | None = None
    pq_m: int = 64
    device: str = "cpu"  # only used by the "torch" backend
    partitions: dict[tuple[str, str], Any] = field(default_factory=dict)
    part_ids: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    build_seconds: float = 0.0
    fallback_queries: int = 0

    def build(self, emb: np.ndarray, ids: np.ndarray, sources: np.ndarray, countries: np.ndarray) -> "DenseIndex":
        t = time.time()
        ids = np.asarray(ids).astype(object)
        sources = np.asarray(sources).astype(str)
        countries = np.asarray(countries).astype(str)
        for c in sorted(set(countries)):
            for s in sorted(set(sources)):
                mask = (countries == c) & (sources == s)
                if not mask.any():
                    continue
                if self.backend == "torch":
                    self.partitions[(c, s)] = _TorchFlat(emb[mask], self.device)
                    self.part_ids[(c, s)] = ids[mask]
                    continue
                x = np.ascontiguousarray(emb[mask], dtype=np.float32)
                index = faiss.index_factory(
                    x.shape[1], factory_string(self.backend, len(x), self.nlist, self.pq_m), faiss.METRIC_INNER_PRODUCT
                )
                if not index.is_trained:
                    index.train(x)
                index.add(x)
                if hasattr(index, "nprobe"):
                    index.nprobe = self.nprobe
                self.partitions[(c, s)] = index
                self.part_ids[(c, s)] = ids[mask]
        self.build_seconds = time.time() - t
        return self

    @property
    def countries(self) -> set[str]:
        return {c for c, _ in self.partitions}

    def stats(self) -> dict[str, object]:
        sizes = {
            f"{c}/{s}": ix.nbytes if isinstance(ix, _TorchFlat) else int(faiss.serialize_index(ix).nbytes)
            for (c, s), ix in self.partitions.items()
        }
        return {
            "backend": self.backend,
            "partitions": {f"{c}/{s}": int(ix.ntotal) for (c, s), ix in self.partitions.items()},
            "index_bytes": sizes,
            "index_gib_total": round(sum(sizes.values()) / 2**30, 3),
            "build_seconds": round(self.build_seconds, 2),
            "fallback_queries": self.fallback_queries,
        }

    def search(self, q_emb: np.ndarray, q_ids: np.ndarray, q_countries: np.ndarray, k: int) -> pd.DataFrame:
        """
        Returns long-format hits (s1_id, source, cand_id, rank, score) with top-k taken
        separately within each source (rank is 1-based within (s1_id, source)).
        Queries from a country with no target partition search all partitions of that source.
        """
        q_emb = np.ascontiguousarray(q_emb, dtype=np.float32)
        q_ids = np.asarray(q_ids).astype(object)
        q_countries = np.asarray(q_countries).astype(str)
        sources = sorted({s for _, s in self.partitions})
        frames = []
        for c in sorted(set(q_countries)):
            qmask = q_countries == c
            if c not in self.countries:
                self.fallback_queries += int(qmask.sum())
            for s in sources:
                parts = [(c, s)] if (c, s) in self.partitions else [p for p in self.partitions if p[1] == s]
                scores_l, ids_l = [], []
                for p in parts:
                    kk = min(k, self.partitions[p].ntotal)
                    d, i = self.partitions[p].search(q_emb[qmask], kk)
                    valid = i >= 0
                    scores_l.append(np.where(valid, d, -np.inf))
                    ids_l.append(np.where(valid, self.part_ids[p][np.clip(i, 0, None)], None))
                scores = np.concatenate(scores_l, axis=1)
                cids = np.concatenate(ids_l, axis=1)
                width = scores.shape[1]
                df = pd.DataFrame(
                    {
                        "s1_id": np.repeat(q_ids[qmask], width),
                        "source": s,
                        "cand_id": cids.ravel(),
                        "score": scores.ravel().astype(np.float32),
                    }
                )
                frames.append(df[df["cand_id"].notna()])
        hits = pd.concat(frames, ignore_index=True)
        hits = hits.sort_values(["s1_id", "source", "score"], ascending=[True, True, False], kind="stable")
        hits["rank"] = hits.groupby(["s1_id", "source"], sort=False).cumcount() + 1
        hits = hits[hits["rank"] <= k]
        return hits.reset_index(drop=True)[["s1_id", "source", "cand_id", "rank", "score"]]
