"""
Gap 1A — 5K Controlled Experiment
===================================
Compares:
  BASELINE : A(native K=25) + B(addr K=20) + C       [translit_top_k=0 to disable Channel D]
  GAP_1A   : A(native K=25) + B(addr K=20) + C + D(translit K=5, min_sim=0.30)

Phase 2: End-to-end 5K blocking + classification metrics.
Phase 3: Channel attribution — every new D candidate classified as
         (already_in_ABC | true_match | false_candidate).
Phase 4: Decision-gate summary.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np
from parallax.data.contracts import load_business_records_df, load_ground_truth_dict
from parallax.preprocessing.normalizer import widen_records_df
from parallax.blocking.tfidf_blocker import DualChannelTFIDFBlocker

N_S1 = 5000
DATA_DIR = Path("data/medium_split_200k")


def build_candidates(s1_wide, target_wide, translit_top_k: int, translit_min_sim: float = 0.30):
    blocker = DualChannelTFIDFBlocker(
        name_top_k=25,
        addr_top_k=20,
        translit_top_k=translit_top_k,
        name_min_sim=0.15,
        addr_min_sim=0.20,
        translit_min_sim=translit_min_sim,
        batch_size=2000,
        show_progress=True,
    )
    return blocker.generate_candidates(s1_wide, target_wide)


def blocking_metrics(candidates, gt_dict, n_target):
    total_pairs = sum(len(c) for c in candidates.values())
    n_s1 = len(candidates)
    avg_cands = total_pairs / n_s1 if n_s1 else 0.0

    recovered = 0
    total_true = 0
    blocking_fns = []
    for s1_id, true_set in gt_dict.items():
        if s1_id not in candidates:
            continue
        total_true += len(true_set)
        cand_set = candidates[s1_id]
        hits = true_set & cand_set
        recovered += len(hits)
        missed = true_set - cand_set
        for m in missed:
            blocking_fns.append((s1_id, m))

    pair_completeness = recovered / total_true if total_true else 0.0
    return {
        "total_pairs": total_pairs,
        "avg_cands_per_s1": avg_cands,
        "blocking_recall": pair_completeness,
        "blocking_fns": len(blocking_fns),
        "blocking_fn_list": blocking_fns,
    }


def detect_cross_script_pairs(s1_wide, target_wide, gt_dict):
    """Identify true pairs where S1 is Brahmic/CJK and target is Latin (or vice versa)."""
    s1_scripts = {}
    for _, row in s1_wide.iterrows():
        name = str(row.get("business_name", ""))
        has_non_latin = any(ord(c) > 0x024F for c in name)
        s1_scripts[str(row["entity_id"])] = "non_latin" if has_non_latin else "latin"

    tgt_scripts = {}
    for _, row in target_wide.iterrows():
        name = str(row.get("business_name", ""))
        has_non_latin = any(ord(c) > 0x024F for c in name)
        tgt_scripts[str(row["entity_id"])] = "non_latin" if has_non_latin else "latin"

    cross_script = set()
    for s1_id, true_set in gt_dict.items():
        s1_sc = s1_scripts.get(s1_id, "latin")
        for tgt_id in true_set:
            tgt_sc = tgt_scripts.get(tgt_id, "latin")
            if s1_sc != tgt_sc:
                cross_script.add((s1_id, tgt_id))
    return cross_script


def channel_d_attribution(cands_abc, cands_gap1a, gt_dict, cross_script_pairs):
    """For every pair in D but not ABC, classify it."""
    d_only_true = 0
    d_only_false = 0
    d_redundant = 0

    d_cross_script_new = 0

    for s1_id, cands_full in cands_gap1a.items():
        cands_base = cands_abc.get(s1_id, set())
        new_from_d = cands_full - cands_base
        true_set = gt_dict.get(s1_id, set())
        for cand in new_from_d:
            if cand in true_set:
                d_only_true += 1
                if (s1_id, cand) in cross_script_pairs:
                    d_cross_script_new += 1
            else:
                d_only_false += 1

    # Pairs in ABC already present in gap1a (redundant coverage)
    for s1_id, cands_base in cands_abc.items():
        cands_full = cands_gap1a.get(s1_id, set())
        d_redundant += len(cands_base & cands_full)

    return {
        "d_incremental_true_matches": d_only_true,
        "d_incremental_false_candidates": d_only_false,
        "d_incremental_cross_script_recoveries": d_cross_script_new,
        "d_total_incremental_pairs": d_only_true + d_only_false,
        "d_precision_on_incremental": d_only_true / (d_only_true + d_only_false) if (d_only_true + d_only_false) > 0 else 0.0,
    }


def cross_script_recall(candidates, cross_script_pairs):
    recovered = 0
    for s1_id, cand_id in cross_script_pairs:
        if cand_id in candidates.get(s1_id, set()):
            recovered += 1
    total = len(cross_script_pairs)
    return recovered / total if total else 0.0, total - recovered


def main():
    print("=" * 72)
    print("GAP 1A — 5K CONTROLLED BLOCKING EXPERIMENT")
    print("=" * 72)

    print("\nLoading data...")
    s1_df = load_business_records_df(DATA_DIR / "train_source1.tsv").head(N_S1)
    s2_df = load_business_records_df(DATA_DIR / "train_source2.tsv")
    s3_df = load_business_records_df(DATA_DIR / "train_source3.tsv")
    gt_all = load_ground_truth_dict(DATA_DIR / "train_ground_truth.tsv")

    valid_ids = set(s1_df["entity_id"].astype(str))
    gt_dict = {k: v for k, v in gt_all.items() if k in valid_ids}

    print(f"  S1 entities : {len(s1_df):,}")
    print(f"  S2+S3 target: {len(s2_df) + len(s3_df):,}")
    print(f"  GT entries  : {len(gt_dict):,}")

    print("\nWidening records...")
    s1_wide = widen_records_df(s1_df)
    target_wide = widen_records_df(pd.concat([s2_df, s3_df], ignore_index=True))

    cross_script = detect_cross_script_pairs(s1_wide, target_wide, gt_dict)
    print(f"  Cross-script true pairs: {len(cross_script):,}")

    # ---- BASELINE: A+B+C only (translit_top_k=0 disables Channel D) ----
    print("\n" + "=" * 72)
    print("BASELINE: A(native K=25) + B(addr K=20) + C")
    print("=" * 72)
    cands_baseline = build_candidates(s1_wide, target_wide, translit_top_k=0)
    m_base = blocking_metrics(cands_baseline, gt_dict, len(target_wide))
    cs_recall_base, cs_fns_base = cross_script_recall(cands_baseline, cross_script)

    print(f"\n  Total candidate pairs : {m_base['total_pairs']:,}")
    print(f"  Avg candidates/S1     : {m_base['avg_cands_per_s1']:.2f}")
    print(f"  Blocking recall       : {m_base['blocking_recall'] * 100:.3f}%")
    print(f"  Blocking FNs          : {m_base['blocking_fns']:,}")
    print(f"  Cross-script recall   : {cs_recall_base * 100:.3f}%")
    print(f"  Cross-script FNs      : {cs_fns_base:,}")

    # ---- GAP 1A CLEAN: A+B+C+D ----
    print("\n" + "=" * 72)
    print("GAP 1A CLEAN: A(native K=25) + B(addr K=20) + C + D(translit K=5, min_sim=0.30)")
    print("=" * 72)
    cands_gap1a = build_candidates(s1_wide, target_wide, translit_top_k=5, translit_min_sim=0.30)
    m_gap = blocking_metrics(cands_gap1a, gt_dict, len(target_wide))
    cs_recall_gap, cs_fns_gap = cross_script_recall(cands_gap1a, cross_script)

    print(f"\n  Total candidate pairs : {m_gap['total_pairs']:,}")
    print(f"  Avg candidates/S1     : {m_gap['avg_cands_per_s1']:.2f}")
    print(f"  Blocking recall       : {m_gap['blocking_recall'] * 100:.3f}%")
    print(f"  Blocking FNs          : {m_gap['blocking_fns']:,}")
    print(f"  Cross-script recall   : {cs_recall_gap * 100:.3f}%")
    print(f"  Cross-script FNs      : {cs_fns_gap:,}")

    # ---- PHASE 3: Channel D Attribution ----
    print("\n" + "=" * 72)
    print("PHASE 3: CHANNEL D ATTRIBUTION")
    print("=" * 72)
    attr = channel_d_attribution(cands_baseline, cands_gap1a, gt_dict, cross_script)
    inflation = m_gap['total_pairs'] - m_base['total_pairs']
    inflation_pct = (inflation / m_base['total_pairs']) * 100 if m_base['total_pairs'] else 0.0

    print(f"\n  Net new pairs from D               : {attr['d_total_incremental_pairs']:,}")
    print(f"  Candidate inflation                : +{inflation:,} ({inflation_pct:.2f}%)")
    print(f"  D incremental TRUE matches         : {attr['d_incremental_true_matches']:,}")
    print(f"  D incremental FALSE candidates     : {attr['d_incremental_false_candidates']:,}")
    print(f"  D cross-script new recoveries      : {attr['d_incremental_cross_script_recoveries']:,}")
    print(f"  D precision on incremental pairs   : {attr['d_precision_on_incremental'] * 100:.2f}%")

    fns_recovered = m_base['blocking_fns'] - m_gap['blocking_fns']
    print(f"\n  Baseline blocking FNs              : {m_base['blocking_fns']:,}")
    print(f"  Gap1A blocking FNs                 : {m_gap['blocking_fns']:,}")
    print(f"  FNs recovered by D                 : {fns_recovered:,}")
    cs_fns_recovered = cs_fns_base - cs_fns_gap
    print(f"  Cross-script FNs recovered by D    : {cs_fns_recovered:,}")

    # ---- PHASE 4: Decision Gate ----
    print("\n" + "=" * 72)
    print("PHASE 4: DECISION GATE SUMMARY")
    print("=" * 72)
    print(f"\n  [1] Candidate volume restored?  Baseline={m_base['total_pairs']:,} vs Gap1A={m_gap['total_pairs']:,}  inflation={inflation_pct:.2f}%")
    print(f"  [2] Cross-script FNs recovered: {cs_fns_recovered:,} of {cs_fns_base:,} ({cs_fns_recovered/cs_fns_base*100:.1f}% if cs_fns_base > 0 else 'N/A')")
    print(f"  [3] D precision on new pairs  : {attr['d_precision_on_incremental']*100:.2f}%")
    print(f"\n  Question answered: Does Channel D provide incremental cross-script recall")
    print(f"  when Channel A is native-only?")
    print(f"  -> D recovered {attr['d_incremental_cross_script_recoveries']:,} cross-script true pairs not in A+B+C,")
    print(f"     at {inflation_pct:.2f}% additional candidate cost.")
    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
