"""
Gap 1A - 5K Controlled Experiment v2
=======================================
Compares:
  BASELINE : A(native K=25) + B(addr K=20) + C   [Channel D explicitly disabled]
  GAP_1A   : A(native K=25) + B(addr K=20) + C + D(translit K=5, min_sim=0.30)

Fix from v1: translit_top_k=0 in Python does [-0:] = [entire array] -- unlimited, not disabled.
Correct fix: drop translit_name columns from the dataframes before the baseline run.
The Channel D guard is: `if "translit_name" in s1_c and "translit_name" in tgt_c`
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
from parallax.data.contracts import load_business_records_df, load_ground_truth_dict
from parallax.preprocessing.normalizer import widen_records_df
from parallax.blocking.tfidf_blocker import DualChannelTFIDFBlocker

N_S1 = 5000
DATA_DIR = Path("data/medium_split_200k")


def build_candidates(s1_wide, target_wide, enable_channel_d: bool):
    blocker = DualChannelTFIDFBlocker(
        name_top_k=25,
        addr_top_k=20,
        translit_top_k=5,
        name_min_sim=0.15,
        addr_min_sim=0.20,
        translit_min_sim=0.30,
        batch_size=2000,
        show_progress=True,
    )
    if not enable_channel_d:
        # Drop translit columns so Channel D guard triggers False
        s1_wide = s1_wide.drop(columns=["translit_name"], errors="ignore")
        target_wide = target_wide.drop(columns=["translit_name"], errors="ignore")
    return blocker.generate_candidates(s1_wide, target_wide)


def blocking_metrics(candidates, gt_dict):
    total_pairs = sum(len(c) for c in candidates.values())
    n_s1 = len(candidates)
    recovered, total_true = 0, 0
    blocking_fns = []
    for s1_id, true_set in gt_dict.items():
        if s1_id not in candidates:
            continue
        total_true += len(true_set)
        hits = true_set & candidates[s1_id]
        recovered += len(hits)
        for m in (true_set - candidates[s1_id]):
            blocking_fns.append((s1_id, m))
    return {
        "total_pairs": total_pairs,
        "avg_cands": total_pairs / n_s1 if n_s1 else 0.0,
        "blocking_recall": recovered / total_true if total_true else 0.0,
        "blocking_fns": len(blocking_fns),
        "fn_list": blocking_fns,
    }


def detect_cross_script(s1_wide, target_wide, gt_dict):
    def is_non_latin(name):
        return any(ord(c) > 0x024F for c in str(name))

    s1_sc = {str(r["entity_id"]): "nl" if is_non_latin(r.get("business_name","")) else "l"
             for _, r in s1_wide.iterrows()}
    tgt_sc = {str(r["entity_id"]): "nl" if is_non_latin(r.get("business_name","")) else "l"
              for _, r in target_wide.iterrows()}

    cs = set()
    for s1_id, true_set in gt_dict.items():
        for tgt_id in true_set:
            if s1_sc.get(s1_id, "l") != tgt_sc.get(tgt_id, "l"):
                cs.add((s1_id, tgt_id))
    return cs


def cs_recall(candidates, cross_script):
    hits = sum(1 for s1, tgt in cross_script if tgt in candidates.get(s1, set()))
    total = len(cross_script)
    return hits / total if total else 0.0, total - hits


def channel_d_attribution(cands_base, cands_gap, gt_dict, cross_script):
    d_true, d_false, d_cs_new = 0, 0, 0
    for s1_id, cands_full in cands_gap.items():
        new = cands_full - cands_base.get(s1_id, set())
        true_set = gt_dict.get(s1_id, set())
        for cand in new:
            if cand in true_set:
                d_true += 1
                if (s1_id, cand) in cross_script:
                    d_cs_new += 1
            else:
                d_false += 1
    total_new = d_true + d_false
    return {
        "d_incremental_true": d_true,
        "d_incremental_false": d_false,
        "d_cs_new": d_cs_new,
        "d_total_new": total_new,
        "d_prec": d_true / total_new if total_new else 0.0,
    }


def main():
    SEP = "=" * 72
    print(SEP)
    print("GAP 1A — 5K CONTROLLED BLOCKING EXPERIMENT v2")
    print(SEP)

    print("\nLoading data...")
    s1_df = load_business_records_df(DATA_DIR / "train_source1.tsv").head(N_S1)
    s2_df = load_business_records_df(DATA_DIR / "train_source2.tsv")
    s3_df = load_business_records_df(DATA_DIR / "train_source3.tsv")
    gt_all = load_ground_truth_dict(DATA_DIR / "train_ground_truth.tsv")
    valid_ids = set(s1_df["entity_id"].astype(str))
    gt_dict = {k: v for k, v in gt_all.items() if k in valid_ids}
    print(f"  S1: {len(s1_df):,} | Target: {len(s2_df)+len(s3_df):,} | GT: {len(gt_dict):,}")

    print("\nWidening records...")
    s1_wide = widen_records_df(s1_df)
    target_wide = widen_records_df(pd.concat([s2_df, s3_df], ignore_index=True))

    cross_script = detect_cross_script(s1_wide, target_wide, gt_dict)
    print(f"  Cross-script true pairs: {len(cross_script):,}")
    print(f"  Has translit_name column: {'translit_name' in s1_wide.columns}")

    # ---- BASELINE: A+B+C only (Channel D disabled by dropping column) ----
    print(f"\n{SEP}")
    print("BASELINE: A(native K=25) + B(addr K=20) + C  [Channel D=OFF]")
    print(SEP)
    cands_base = build_candidates(s1_wide, target_wide, enable_channel_d=False)
    m_base = blocking_metrics(cands_base, gt_dict)
    csr_base, csfn_base = cs_recall(cands_base, cross_script)
    print(f"\n  Total candidate pairs : {m_base['total_pairs']:,}")
    print(f"  Avg candidates/S1     : {m_base['avg_cands']:.2f}")
    print(f"  Blocking recall       : {m_base['blocking_recall']*100:.3f}%")
    print(f"  Blocking FNs          : {m_base['blocking_fns']:,}")
    print(f"  Cross-script recall   : {csr_base*100:.3f}%")
    print(f"  Cross-script FNs      : {csfn_base:,}")

    # ---- GAP 1A: A+B+C+D ----
    print(f"\n{SEP}")
    print("GAP 1A: A(native K=25)+B(addr K=20)+C + D(translit K=5, min_sim=0.30) [Channel D=ON]")
    print(SEP)
    cands_gap = build_candidates(s1_wide, target_wide, enable_channel_d=True)
    m_gap = blocking_metrics(cands_gap, gt_dict)
    csr_gap, csfn_gap = cs_recall(cands_gap, cross_script)
    print(f"\n  Total candidate pairs : {m_gap['total_pairs']:,}")
    print(f"  Avg candidates/S1     : {m_gap['avg_cands']:.2f}")
    print(f"  Blocking recall       : {m_gap['blocking_recall']*100:.3f}%")
    print(f"  Blocking FNs          : {m_gap['blocking_fns']:,}")
    print(f"  Cross-script recall   : {csr_gap*100:.3f}%")
    print(f"  Cross-script FNs      : {csfn_gap:,}")

    # ---- PHASE 3: Channel D Attribution ----
    print(f"\n{SEP}")
    print("PHASE 3: CHANNEL D ATTRIBUTION")
    print(SEP)
    attr = channel_d_attribution(cands_base, cands_gap, gt_dict, cross_script)
    inflation = m_gap["total_pairs"] - m_base["total_pairs"]
    inflation_pct = inflation / m_base["total_pairs"] * 100 if m_base["total_pairs"] else 0.0
    fns_recovered = m_base["blocking_fns"] - m_gap["blocking_fns"]
    cs_fns_recovered = csfn_base - csfn_gap

    print(f"\n  Net new pairs from D              : {attr['d_total_new']:,}")
    print(f"  Candidate inflation               : +{inflation:,} ({inflation_pct:+.2f}%)")
    print(f"  D incremental TRUE matches        : {attr['d_incremental_true']:,}")
    print(f"  D incremental FALSE candidates    : {attr['d_incremental_false']:,}")
    print(f"  D cross-script new recoveries     : {attr['d_cs_new']:,}")
    print(f"  D precision on incremental pairs  : {attr['d_prec']*100:.2f}%")
    print(f"\n  Baseline blocking FNs             : {m_base['blocking_fns']:,}")
    print(f"  Gap1A blocking FNs                : {m_gap['blocking_fns']:,}")
    print(f"  FNs recovered by D                : {fns_recovered:,}")
    print(f"  Cross-script FNs recovered by D   : {cs_fns_recovered:,}")

    # ---- PHASE 4: Decision Gate ----
    print(f"\n{SEP}")
    print("PHASE 4: DECISION GATE")
    print(SEP)
    print(f"\n  [1] Volume: Baseline={m_base['total_pairs']:,}  Gap1A={m_gap['total_pairs']:,}  Inflation={inflation_pct:+.2f}%")
    cs_fns_pct = cs_fns_recovered / csfn_base * 100 if csfn_base > 0 else 0.0
    print(f"  [2] Cross-script FN recovery: {cs_fns_recovered:,} of {csfn_base:,} ({cs_fns_pct:.1f}%)")
    print(f"  [3] D precision on new candidates: {attr['d_prec']*100:.2f}%")
    print(f"\n  Experimental question:")
    print(f"  'Does Channel D add incremental cross-script recall when A is native-only?'")
    if attr["d_cs_new"] > 0:
        print(f"  -> YES: D recovered {attr['d_cs_new']:,} cross-script pairs not in A+B+C")
        print(f"     at {inflation_pct:+.2f}% candidate cost ({attr['d_prec']*100:.1f}% precision on new pairs)")
    else:
        print(f"  -> NO: D recovered 0 cross-script pairs not already in A+B+C")
    print(f"\n{SEP}")


if __name__ == "__main__":
    main()
