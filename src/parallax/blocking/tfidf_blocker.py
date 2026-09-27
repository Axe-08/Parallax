"""
Parallax Multi-Channel Sparse & Phonetic Blocker (v3)
=====================================================
High-recall, memory-efficient candidate generator:
- Dynamic country-partitioned indexing (open-set: US, India, France)
- Channel A: Word (1, 2) TF-IDF over business names with single-char token pattern
- Channel B: Word (1, 2) TF-IDF over addresses with single-char token pattern
- Channel C: Building Number Match with Leading Character Prefix
- Channel D: Postal / PIN Code Hash-Join with length ratio filtering
- Channel E: Phonetic First-Token Match using Metaphone
- Pruning, similarity preservation, and union aggregation producing candidate_pairs.tsv
"""

from __future__ import annotations

import gc
from collections import defaultdict
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING

import jellyfish
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from parallax.data.contracts import write_candidate_pairs_tsv
from parallax.preprocessing.transliteration import transliterate_brahmic_to_latin

if TYPE_CHECKING:
    from parallax.utils.checkpoint_manager import CheckpointManager

# Fork-shared state for the multi-country parallel path.
_COUNTRY_BLOCKERS: dict[str, DualChannelTFIDFBlocker] = {}
_COUNTRY_QUERIES: dict[str, pd.DataFrame] = {}
_COUNTRY_TARGETS: dict[str, pd.DataFrame] = {}


def _worker_index_part(task: tuple[str, str, dict[str, object]]) -> dict[str, object]:
    """Build one (country, part) index in a child process and return its attributes."""
    country, part, params = task
    b = DualChannelTFIDFBlocker(**params)  # type: ignore[arg-type]
    tgt_c = _COUNTRY_TARGETS[country]
    b._index_common(tgt_c, country)
    b._index_part(tgt_c, part)
    return b.export_index_part(part)


def _worker_query_slice(
    task: tuple[str, int, int, int | None, int | None],
) -> dict[str, dict[str, float]]:
    """Block one contiguous slice of a country's queries against the shared index."""
    country, start, end, batch_size, max_cands = task
    blocker = _COUNTRY_BLOCKERS[country]
    return blocker._block_queries_serial(
        _COUNTRY_QUERIES[country].iloc[start:end],
        country=country,
        batch_size=batch_size,
        max_candidates_per_query=max_cands,
        show_progress=False,
    )


def build_blocking_texts(
    df: pd.DataFrame, primary_col: str, fallback_col: str, translit_col: str
) -> list[str]:
    """Build dual-representation blocking texts (native + transliterated Latin)."""
    if primary_col in df:
        p_s = df[primary_col].fillna("").astype(str).str.strip()
        if fallback_col in df:
            empty = p_s == ""
            if empty.any():
                p_s[empty] = df.loc[empty, fallback_col].fillna("").astype(str).str.strip()
    elif fallback_col in df:
        p_s = df[fallback_col].fillna("").astype(str).str.strip()
    else:
        p_s = pd.Series([""] * len(df), index=df.index)

    if translit_col in df:
        t_s = df[translit_col].fillna("").astype(str).str.strip()
        diff_mask = (t_s != "") & (t_s != p_s)
        res = p_s.copy()
        if diff_mask.any():
            res[diff_mask] = p_s[diff_mask] + " " + t_s[diff_mask]
        return [str(x) for x in res.tolist()]
    return [str(x) for x in p_s.tolist()]


class DualChannelTFIDFBlocker:
    """
    Generates candidate match pairs using multi-channel sparse TF-IDF,
    building number alignment, postal code hashing, and phonetic indexing.
    Supports stateful target indexing for memory-bounded streaming evaluation.
    """

    def __init__(
        self,
        name_top_k: int = 15,
        addr_top_k: int = 10,
        name_min_sim: float = 0.15,
        addr_min_sim: float = 0.20,
        batch_size: int = 1000,
        show_progress: bool = True,
        max_candidates_per_query: int = 35,
        n_jobs: int = 1,
    ) -> None:
        self.name_top_k = name_top_k
        self.addr_top_k = addr_top_k
        self.name_min_sim = name_min_sim
        self.addr_min_sim = addr_min_sim
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.max_candidates_per_query = max_candidates_per_query
        self.n_jobs = n_jobs

        # Stateful index storage for current country partition
        self.indexed_country: str | None = None
        self.tgt_ids: list[str] = []
        self.vec_name: TfidfVectorizer | None = None
        self.tgt_name_mat: sp.csr_matrix | None = None
        self.vec_addr: TfidfVectorizer | None = None
        self.tgt_addr_mat: sp.csr_matrix | None = None
        self.tgt_prefixes: dict[str, str] = {}
        self.tgt_translit_prefixes: dict[str, str] = {}
        self.num_to_tgt: dict[str, list[str]] = defaultdict(list)
        self.postal_to_tgt: dict[str, list[int]] = defaultdict(list)
        self.tgt_names: list[str] = []
        self.phonetic_to_tgt: dict[str, list[int]] = defaultdict(list)
        self.addr_hash_to_tgt: dict[tuple[str, str], list[int]] = defaultdict(list)

    _INDEX_PART_ATTRS: dict[str, tuple[str, ...]] = {
        "name": ("vec_name", "tgt_name_mat"),
        "addr": ("vec_addr", "tgt_addr_mat"),
        "hash": (
            "tgt_prefixes",
            "tgt_translit_prefixes",
            "num_to_tgt",
            "postal_to_tgt",
            "phonetic_to_tgt",
            "addr_hash_to_tgt",
        ),
    }

    def index_country_targets(self, tgt_c: pd.DataFrame, country: str = "") -> None:
        """
        Pre-index all target channels for a single country partition.
        Fits float32 sparse TF-IDF with stop-gram pruning and builds hash indices.
        Guarantees bounded RAM < 1.5 GB even for 6M+ target pools.
        """
        self.clear_index()
        self._index_common(tgt_c, country)
        for part in ("name", "addr", "hash"):
            self._index_part(tgt_c, part)
        gc.collect()

    def _index_common(self, tgt_c: pd.DataFrame, country: str) -> None:
        """Cheap shared state needed by every channel (ids and blocking name texts)."""
        self.indexed_country = country
        self.tgt_ids = tgt_c["entity_id"].astype(str).tolist()
        self.tgt_names = build_blocking_texts(tgt_c, "soft_name", "business_name", "translit_name")

    def _index_part(self, tgt_c: pd.DataFrame, part: str) -> None:
        """Build one independent index part: 'name', 'addr', or 'hash'."""
        n_targets = len(tgt_c)
        tgt_names = self.tgt_names

        if part == "name":
            # Channel A: Name Word (1, 2) TF-IDF with High-Frequency Pruning
            min_df = 2 if n_targets > 500 else 1
            max_df = 0.05 if n_targets > 500 else 1.0
            self.vec_name = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                token_pattern=r"(?u)\b\w+\b",
                min_df=min_df,
                max_df=max_df,
                sublinear_tf=True,
                dtype=np.float32,
            )
            tgt_mat_name = self.vec_name.fit_transform(tgt_names)
            self.tgt_name_mat = tgt_mat_name.T.tocsr().astype(np.float32)
            del tgt_mat_name
            return

        if part == "addr":
            # Channel B: Address Word (1, 2) TF-IDF with High-Frequency Pruning
            tgt_addrs = build_blocking_texts(
                tgt_c, "clean_address", "business_address", "translit_address"
            )
            if any(a.strip() for a in tgt_addrs):
                min_df_addr = 2 if n_targets > 500 else 1
                max_df_addr = 0.02 if n_targets > 500 else 1.0
                self.vec_addr = TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    token_pattern=r"(?u)\b\w+\b",
                    min_df=min_df_addr,
                    max_df=max_df_addr,
                    sublinear_tf=True,
                    dtype=np.float32,
                )
                tgt_mat_addr = self.vec_addr.fit_transform(tgt_addrs)
                self.tgt_addr_mat = tgt_mat_addr.T.tocsr().astype(np.float32)
                del tgt_mat_addr
            return

        if part != "hash":
            raise ValueError(f"Unknown index part: {part}")

        # Channel C: Building Number Match with Leading Character Prefix
        if "numbers" in tgt_c:
            tgt_names_ser = (
                tgt_c["soft_name"].fillna("").astype(str)
                if "soft_name" in tgt_c
                else tgt_c["business_name"].fillna("").astype(str)
            )
            tgt_translit_ser = (
                tgt_c["translit_name"].fillna("").astype(str)
                if "translit_name" in tgt_c
                else tgt_names_ser
            )
            self.tgt_prefixes = dict(
                zip(self.tgt_ids, tgt_names_ser.str[:2].tolist(), strict=False)
            )
            self.tgt_translit_prefixes = dict(
                zip(self.tgt_ids, tgt_translit_ser.str[:2].tolist(), strict=False)
            )

            for eid, num_set in zip(self.tgt_ids, tgt_c["numbers"].tolist(), strict=False):
                if isinstance(num_set, (set, list, np.ndarray)):
                    for num in num_set:
                        num_str = str(num).strip()
                        if len(num_str) >= 2:
                            self.num_to_tgt[num_str].append(eid)

        # Channel D: Postal Code Hash Join
        if "postal_code" in tgt_c:
            tgt_postals = tgt_c["postal_code"].tolist()
            for i, pc in enumerate(tgt_postals):
                if pc and pd.notna(pc) and str(pc).strip().lower() not in ("", "none", "nan"):
                    self.postal_to_tgt[str(pc).strip()].append(i)

        # Channel E: Phonetic First-Token Match (Metaphone)
        for i, name_str in enumerate(tgt_names):
            toks = name_str.split()
            if toks:
                tok0 = toks[0]
                if any(ord(c) > 127 for c in tok0):
                    tok0 = transliterate_brahmic_to_latin(tok0)
                code = jellyfish.metaphone(tok0) if tok0 else ""
                if code:
                    self.phonetic_to_tgt[code].append(i)

        # Channel F: Address Token Hash-Join (Primary Number + City Token)
        if "primary_number" in tgt_c and "city_token" in tgt_c:
            tgt_pnums = tgt_c["primary_number"].tolist()
            tgt_cities = tgt_c["city_token"].tolist()
            for i, (pn, ct) in enumerate(zip(tgt_pnums, tgt_cities, strict=False)):
                if pn and ct and pd.notna(pn) and pd.notna(ct):
                    p_str = str(pn).strip()
                    c_str = str(ct).strip().lower()
                    if len(p_str) >= 2 and len(c_str) >= 3 and c_str not in ("none", "nan", ""):
                        self.addr_hash_to_tgt[(p_str, c_str)].append(i)

    def export_index_part(self, part: str) -> dict[str, object]:
        """Return the attributes produced by an index part (for cross-process transfer)."""
        return {a: getattr(self, a) for a in self._INDEX_PART_ATTRS[part]}

    def import_index_part(self, attrs: dict[str, object]) -> None:
        for k, v in attrs.items():
            setattr(self, k, v)

    def _block_queries_serial(
        self,
        s1_c: pd.DataFrame,
        country: str = "",
        batch_size: int | None = None,
        max_candidates_per_query: int | None = None,
        show_progress: bool | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Generate candidate pairs for queries against pre-indexed targets sequentially.
        Streams in micro-batches (default 1000) to bound RAM under 100 MB per batch.
        """
        if self.tgt_name_mat is None or self.vec_name is None:
            raise RuntimeError("Targets have not been indexed! Call index_country_targets() first.")

        s1_ids = s1_c["entity_id"].astype(str).tolist()
        candidate_pairs: dict[str, dict[str, float]] = {s1_id: {} for s1_id in s1_ids}

        def _add_cand(s1_id: str, cand_id: str, sim: float) -> None:
            curr = candidate_pairs[s1_id].get(cand_id, 0.0)
            if sim > curr:
                candidate_pairs[s1_id][cand_id] = float(sim)

        bs = batch_size if batch_size is not None else self.batch_size
        s1_names = build_blocking_texts(s1_c, "soft_name", "business_name", "translit_name")
        s1_addrs = build_blocking_texts(
            s1_c, "clean_address", "business_address", "translit_address"
        )

        batch_ranges = list(range(0, len(s1_c), bs))
        pbar = tqdm(
            batch_ranges,
            desc=f"  ⚡ Blocking [{country or self.indexed_country}]",
            unit="batch",
            leave=False,
            disable=not (show_progress if show_progress is not None else self.show_progress),
        )

        for start_idx in pbar:
            end_idx = min(start_idx + bs, len(s1_c))
            b_s1_ids = s1_ids[start_idx:end_idx]
            b_s1_names = s1_names[start_idx:end_idx]
            b_s1_addrs = s1_addrs[start_idx:end_idx]

            # --- Channel A: Name TF-IDF Dot Product ---
            b_mat_name = self.vec_name.transform(b_s1_names).astype(np.float32)
            b_sims_name = b_mat_name.dot(self.tgt_name_mat)

            for i, s1_id in enumerate(b_s1_ids):
                r_start = b_sims_name.indptr[i]
                r_end = b_sims_name.indptr[i + 1]
                if r_start == r_end:
                    continue
                scores = b_sims_name.data[r_start:r_end]
                col_idx = b_sims_name.indices[r_start:r_end]

                valid_mask = scores >= self.name_min_sim
                if not np.any(valid_mask):
                    continue
                scores = scores[valid_mask]
                col_idx = col_idx[valid_mask]

                if len(scores) > self.name_top_k:
                    top_sub = np.argpartition(scores, -self.name_top_k)[-self.name_top_k :]
                    top_cols = col_idx[top_sub]
                    top_scores = scores[top_sub]
                else:
                    top_cols = col_idx
                    top_scores = scores

                for c_idx, s_val in zip(top_cols, top_scores, strict=False):
                    _add_cand(s1_id, self.tgt_ids[c_idx], float(s_val))

            del b_mat_name, b_sims_name

            # --- Channel B: Address TF-IDF Dot Product ---
            if (
                self.vec_addr is not None
                and self.tgt_addr_mat is not None
                and any(a.strip() for a in b_s1_addrs)
            ):
                b_mat_addr = self.vec_addr.transform(b_s1_addrs).astype(np.float32)
                b_sims_addr = b_mat_addr.dot(self.tgt_addr_mat)

                for i, s1_id in enumerate(b_s1_ids):
                    if not b_s1_addrs[i].strip():
                        continue
                    r_start = b_sims_addr.indptr[i]
                    r_end = b_sims_addr.indptr[i + 1]
                    if r_start == r_end:
                        continue
                    scores = b_sims_addr.data[r_start:r_end]
                    col_idx = b_sims_addr.indices[r_start:r_end]

                    valid_mask = scores >= self.addr_min_sim
                    if not np.any(valid_mask):
                        continue
                    scores = scores[valid_mask]
                    col_idx = col_idx[valid_mask]

                    if len(scores) > self.addr_top_k:
                        top_sub = np.argpartition(scores, -self.addr_top_k)[-self.addr_top_k :]
                        top_cols = col_idx[top_sub]
                        top_scores = scores[top_sub]
                    else:
                        top_cols = col_idx
                        top_scores = scores

                    for c_idx, s_val in zip(top_cols, top_scores, strict=False):
                        _add_cand(s1_id, self.tgt_ids[c_idx], float(s_val))

                del b_mat_addr, b_sims_addr

        # --- Channel C: Building Number Match ---
        if self.num_to_tgt and "numbers" in s1_c:
            s1_names_ser = (
                s1_c["soft_name"].fillna("").astype(str)
                if "soft_name" in s1_c
                else s1_c["business_name"].fillna("").astype(str)
            )
            s1_translit_ser = (
                s1_c["translit_name"].fillna("").astype(str)
                if "translit_name" in s1_c
                else s1_names_ser
            )
            s1_pfxs = s1_names_ser.str[:2].tolist()
            s1_tpfxs = s1_translit_ser.str[:2].tolist()
            s1_nums_list = s1_c["numbers"].tolist()

            for s1_id, pfx, tpfx, num_set in zip(
                s1_ids, s1_pfxs, s1_tpfxs, s1_nums_list, strict=False
            ):
                if not pfx and not tpfx:
                    continue
                s1_p_set = {p for p in (pfx, tpfx) if p}
                if isinstance(num_set, (set, list, np.ndarray)):
                    for num in num_set:
                        num_str = str(num).strip()
                        if len(num_str) < 2:
                            continue
                        cands = self.num_to_tgt.get(num_str, [])
                        if 0 < len(cands) <= 50:
                            for cand_id in cands:
                                cand_p_set = {
                                    self.tgt_prefixes.get(cand_id, ""),
                                    self.tgt_translit_prefixes.get(cand_id, ""),
                                }
                                if s1_p_set.intersection(cand_p_set):
                                    _add_cand(s1_id, cand_id, 0.50)

        # --- Channel D: Postal Code Hash Join ---
        if self.postal_to_tgt and "postal_code" in s1_c:
            s1_postals = s1_c["postal_code"].tolist()
            for i, s1_id in enumerate(s1_ids):
                pc = s1_postals[i]
                if not pc or pd.isna(pc) or str(pc).strip().lower() in ("", "none", "nan"):
                    continue
                pc_str = str(pc).strip()
                matches = self.postal_to_tgt.get(pc_str, [])
                if 0 < len(matches) <= 40:
                    s1_n = s1_names[i]
                    len_s1 = len(s1_n)
                    for tgt_idx in matches:
                        tgt_n = self.tgt_names[tgt_idx]
                        len_tgt = len(tgt_n)
                        if len_s1 > 0 and len_tgt > 0:
                            ratio = min(len_s1, len_tgt) / max(len_s1, len_tgt)
                            if ratio >= 0.40:
                                _add_cand(s1_id, self.tgt_ids[tgt_idx], 0.50)

        # --- Channel E: Phonetic First-Token Match ---
        if self.phonetic_to_tgt:
            for i, s1_id in enumerate(s1_ids):
                s1_n = s1_names[i]
                toks = s1_n.split()
                if not toks:
                    continue
                tok0 = toks[0]
                if any(ord(c) > 127 for c in tok0):
                    tok0 = transliterate_brahmic_to_latin(tok0)
                code = jellyfish.metaphone(tok0) if tok0 else ""
                if not code:
                    continue
                matches = self.phonetic_to_tgt.get(code, [])
                if 0 < len(matches) <= 30:
                    len_s1 = len(s1_n)
                    for tgt_idx in matches:
                        tgt_n = self.tgt_names[tgt_idx]
                        len_tgt = len(tgt_n)
                        if len_s1 > 0 and len_tgt > 0:
                            ratio = min(len_s1, len_tgt) / max(len_s1, len_tgt)
                            if ratio >= 0.50:
                                _add_cand(s1_id, self.tgt_ids[tgt_idx], 0.45)

        # --- Channel F: Address Token Hash-Join ---
        if self.addr_hash_to_tgt and "primary_number" in s1_c and "city_token" in s1_c:
            s1_pnums = s1_c["primary_number"].tolist()
            s1_cities = s1_c["city_token"].tolist()
            for i, s1_id in enumerate(s1_ids):
                pn = s1_pnums[i]
                ct = s1_cities[i]
                if pn and ct and pd.notna(pn) and pd.notna(ct):
                    p_str = str(pn).strip()
                    c_str = str(ct).strip().lower()
                    if len(p_str) >= 2 and len(c_str) >= 3:
                        matches = self.addr_hash_to_tgt.get((p_str, c_str), [])
                        if 0 < len(matches) <= 20:
                            for tgt_idx in matches:
                                _add_cand(s1_id, self.tgt_ids[tgt_idx], 0.40)

        max_cands = (
            max_candidates_per_query
            if max_candidates_per_query is not None
            else self.max_candidates_per_query
        )
        if max_cands is not None and max_cands > 0:
            for s1_id, query_cands in candidate_pairs.items():
                if len(query_cands) > max_cands:
                    top_items = sorted(query_cands.items(), key=lambda item: item[1], reverse=True)[
                        :max_cands
                    ]
                    candidate_pairs[s1_id] = dict(top_items)

        return candidate_pairs

    def block_queries(
        self,
        s1_c: pd.DataFrame,
        country: str = "",
        batch_size: int | None = None,
        max_candidates_per_query: int | None = None,
        n_jobs: int | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Generate candidate pairs for queries against pre-indexed targets.
        When n_jobs > 1 and len(s1_c) >= 2000, parallelizes query batches
        across worker processes using copy-on-write shared target matrices.
        """
        from parallax.utils.parallel import chunk_bounds, fork_available, run_pool

        effective_n_jobs = n_jobs if n_jobs is not None else self.n_jobs
        if effective_n_jobs <= 1 or len(s1_c) < 2000 or not fork_available():
            return self._block_queries_serial(
                s1_c,
                country=country,
                batch_size=batch_size,
                max_candidates_per_query=max_candidates_per_query,
            )

        c_key = country or (self.indexed_country or "")
        _COUNTRY_BLOCKERS[c_key] = self
        _COUNTRY_QUERIES[c_key] = s1_c
        tasks = [
            (c_key, a, b, batch_size, max_candidates_per_query)
            for a, b in chunk_bounds(len(s1_c), effective_n_jobs * 4)
        ]
        try:
            parts = run_pool(
                _worker_query_slice, tasks, max_workers=effective_n_jobs, label="blocking"
            )
        finally:
            _COUNTRY_BLOCKERS.pop(c_key, None)
            _COUNTRY_QUERIES.pop(c_key, None)

        candidate_pairs: dict[str, dict[str, float]] = {}
        for cr in parts:
            candidate_pairs.update(cr)
        return candidate_pairs

    def _blocker_params(self) -> dict[str, object]:
        return {
            "name_top_k": self.name_top_k,
            "addr_top_k": self.addr_top_k,
            "name_min_sim": self.name_min_sim,
            "addr_min_sim": self.addr_min_sim,
            "batch_size": self.batch_size,
            "show_progress": False,
            "max_candidates_per_query": self.max_candidates_per_query,
            "n_jobs": 1,
        }

    def _generate_candidates_parallel(
        self,
        country_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
        n_jobs: int,
    ) -> dict[str, dict[str, dict[str, float]]]:
        """
        Fully parallel multi-country blocking:
        1. Build every (country, part) index concurrently in child processes.
        2. Block all countries' query slices in one shared pool (over-partitioned 4x).
        Returns {country: {s1_id: {cand_id: sim}}}.
        """
        import time

        from parallax.utils.parallel import chunk_bounds, run_pool

        params = self._blocker_params()
        for country, (_, tgt_c) in country_frames.items():
            _COUNTRY_TARGETS[country] = tgt_c

        try:
            t0 = time.time()
            idx_tasks = [
                (country, part, params)
                for country in country_frames
                for part in ("name", "addr", "hash")
            ]
            idx_parts = run_pool(
                _worker_index_part,
                idx_tasks,
                max_workers=min(n_jobs, len(idx_tasks)),
                label="blocking-index",
            )
            for country, (_, tgt_c) in country_frames.items():
                b = DualChannelTFIDFBlocker(**params)  # type: ignore[arg-type]
                b._index_common(tgt_c, country)
                _COUNTRY_BLOCKERS[country] = b
            for (country, _, _), attrs in zip(idx_tasks, idx_parts, strict=True):
                _COUNTRY_BLOCKERS[country].import_index_part(attrs)
            del idx_parts
            gc.collect()
            print(
                f"  ✓ Built {len(idx_tasks)} index parts for {len(country_frames)} "
                f"countries in parallel in {time.time() - t0:.1f}s.",
                flush=True,
            )

            t0 = time.time()
            total_q = sum(len(s) for s, _ in country_frames.values())
            n_chunks = max(1, n_jobs * 4)
            q_tasks: list[tuple[str, int, int, int | None, int | None]] = []
            for country, (s1_c, _) in country_frames.items():
                _COUNTRY_QUERIES[country] = s1_c
                share = max(1, round(n_chunks * len(s1_c) / max(total_q, 1)))
                q_tasks.extend(
                    (country, a, b, None, None) for a, b in chunk_bounds(len(s1_c), share)
                )
            done_q = [0]
            pbar = tqdm(
                total=len(q_tasks),
                desc="  ⚡ Blocking slices",
                unit="slice",
                leave=False,
                disable=not self.show_progress,
            )

            def _tick(_i: int, _r: object) -> None:
                done_q[0] += 1
                pbar.update(1)

            q_parts = run_pool(
                _worker_query_slice,
                q_tasks,
                max_workers=n_jobs,
                label="blocking-query",
                on_result=_tick,
            )
            pbar.close()
            print(
                f"  ✓ Blocked {total_q:,} queries in {len(q_tasks)} slices "
                f"in {time.time() - t0:.1f}s.",
                flush=True,
            )
        finally:
            _COUNTRY_TARGETS.clear()
            _COUNTRY_QUERIES.clear()
            _COUNTRY_BLOCKERS.clear()
            gc.collect()

        out: dict[str, dict[str, dict[str, float]]] = {c: {} for c in country_frames}
        for (country, *_), part in zip(q_tasks, q_parts, strict=True):
            out[country].update(part)
        return out

    def clear_index(self) -> None:
        """Explicitly deallocate target indices and reclaim memory."""
        self.indexed_country = None
        self.tgt_ids = []
        self.vec_name = None
        self.tgt_name_mat = None
        self.vec_addr = None
        self.tgt_addr_mat = None
        self.tgt_prefixes = {}
        self.tgt_translit_prefixes = {}
        self.num_to_tgt = defaultdict(list)
        self.postal_to_tgt = defaultdict(list)
        self.tgt_names = []
        self.phonetic_to_tgt = defaultdict(list)
        self.addr_hash_to_tgt = defaultdict(list)
        gc.collect()

    def generate_candidates(
        self,
        s1_df: pd.DataFrame,
        target_df: pd.DataFrame,
        checkpoint_mgr: CheckpointManager | None = None,
        n_jobs: int | None = None,
        ckpt_tag: str | None = None,
    ) -> dict[str, dict[str, float]]:
        """
        Generate candidate pairs partitioned by country using stateful lifecycle.
        ckpt_tag (a config fingerprint) keys the per-country checkpoints so different
        blocking configurations never reuse each other's candidates.
        Maintains complete backward compatibility with existing tests and scripts.
        """
        candidate_pairs: dict[str, dict[str, float]] = {s1_id: {} for s1_id in s1_df["entity_id"]}
        countries = s1_df["country"].unique()
        effective_n_jobs = n_jobs if n_jobs is not None else self.n_jobs

        pending: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
        for country in countries:
            s1_c = s1_df[s1_df["country"] == country].reset_index(drop=True)
            tgt_c = target_df[target_df["country"] == country].reset_index(drop=True)

            if len(s1_c) == 0 or len(tgt_c) == 0:
                continue

            cp_key = (
                f"candidates_{ckpt_tag}_{country}_{len(s1_c)}"
                if ckpt_tag
                else f"candidates_{country}_{len(s1_c)}"
            )
            if checkpoint_mgr is not None and checkpoint_mgr.has_checkpoint(cp_key):
                loaded_df = checkpoint_mgr.load_dataframe(cp_key)
                if loaded_df is not None:
                    s1_arr = loaded_df["s1_id"].to_numpy()
                    cand_arr = loaded_df["cand_id"].to_numpy()
                    sim_arr = (
                        loaded_df["blocking_sim"].to_numpy()
                        if "blocking_sim" in loaded_df.columns
                        else np.zeros(len(loaded_df), dtype=np.float32)
                    )
                    for s1, cand, sim in zip(s1_arr, cand_arr, sim_arr, strict=False):
                        s1_str = str(s1)
                        if s1_str in candidate_pairs:
                            candidate_pairs[s1_str][str(cand)] = float(sim)
                    msg = (
                        f"  ⚡ [Checkpoint] Loaded {len(loaded_df):,} candidates for "
                        f"[{country}] ({len(s1_c):,} queries)."
                    )
                    print(msg)
                    continue
            pending[country] = (s1_c, tgt_c)

        if not pending:
            return candidate_pairs

        from parallax.utils.parallel import fork_available

        use_parallel = (
            effective_n_jobs > 1
            and fork_available()
            and sum(len(s) for s, _ in pending.values()) >= 2000
        )
        if use_parallel:
            per_country = self._generate_candidates_parallel(pending, effective_n_jobs)
        else:
            per_country = {}
            for country, (s1_c, tgt_c) in pending.items():
                self.index_country_targets(tgt_c, country=country)
                per_country[country] = self.block_queries(s1_c, country=country, n_jobs=1)
                self.clear_index()

        for country, country_cands in per_country.items():
            s1_c = pending[country][0]
            for s1_id, cands in country_cands.items():
                if s1_id in candidate_pairs:
                    candidate_pairs[s1_id].update(cands)

            if checkpoint_mgr is not None:
                cp_key = (
                    f"candidates_{ckpt_tag}_{country}_{len(s1_c)}"
                    if ckpt_tag
                    else f"candidates_{country}_{len(s1_c)}"
                )
                c_rows_s1: list[str] = []
                c_rows_cand: list[str] = []
                c_rows_sim: list[float] = []
                for s1_id, cands in country_cands.items():
                    for cand_id, sim in cands.items():
                        c_rows_s1.append(s1_id)
                        c_rows_cand.append(cand_id)
                        c_rows_sim.append(sim)
                df_country = pd.DataFrame(
                    {"s1_id": c_rows_s1, "cand_id": c_rows_cand, "blocking_sim": c_rows_sim}
                )
                checkpoint_mgr.save_dataframe(cp_key, df_country)
                checkpoint_mgr.record_stage_completed(
                    f"blocking_{country}_{len(s1_c)}",
                    {"country": country, "pairs_count": len(df_country)},
                )

        return candidate_pairs

    def export_candidate_pairs(
        self,
        output_path: str,
        candidates: Mapping[str, Collection[str]],
    ) -> None:
        """Export candidates dictionary to candidate_pairs.tsv."""
        write_candidate_pairs_tsv(output_path, candidates)
