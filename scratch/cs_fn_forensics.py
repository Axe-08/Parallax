"""
Cross-Script Blocking FN Forensic Analysis
============================================
Identifies the 168 remaining cross-script blocking FNs from the
Baseline (A+B+C, D=OFF) and performs per-pair root-cause analysis.

For each FN pair, computes:
  - Native/soft-name char-3gram TF-IDF cosine similarity
  - Translit char-3gram TF-IDF cosine similarity
  - Address char-3gram TF-IDF cosine similarity
  - Building-number compatibility (shared number)
  - Which channel (if any) already retrieves the pair via a wider search

Root-cause categories:
  A. Bad/inconsistent transliteration   (translit sim < 0.30, native also low)
  B. Insufficient char overlap           (both names short, n-grams sparse)
  C. Short-name generic collision        (name too short < 4 chars)
  D. Phonetic/typo corruption            (edit distance < 3 but char-ngram low)
  E. Address-only relationship           (no usable name signal at all)
  F. Recoverable at higher K             (sim >= 0.30 but rank > 5)
  G. True hard negative / different entities (sanity check)
"""
from __future__ import annotations
import sys, unicodedata
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from parallax.data.contracts import load_business_records_df, load_ground_truth_dict
from parallax.preprocessing.normalizer import widen_records_df
from parallax.blocking.tfidf_blocker import DualChannelTFIDFBlocker

N_S1 = 5000
DATA_DIR = Path("data/medium_split_200k")


def char_ngram_sim(a: str, b: str, n: int = 3) -> float:
    """Cosine similarity between two strings using char n-gram TF-IDF."""
    if not a.strip() or not b.strip():
        return 0.0
    try:
        v = TfidfVectorizer(analyzer="char", ngram_range=(n, n), sublinear_tf=True)
        m = v.fit_transform([a, b])
        return float(cosine_similarity(m[0:1], m[1:2])[0, 0])
    except Exception:
        return 0.0


def levenshtein(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    if not a: return len(b)
    if not b: return len(a)
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        ndp = [i + 1]
        for j, cb in enumerate(b):
            ndp.append(min(dp[j] + (0 if ca == cb else 1), dp[j+1] + 1, ndp[j] + 1))
        dp = ndp
    return dp[-1]


def is_non_latin(text: str) -> bool:
    return any(ord(c) > 0x024F for c in str(text) if not unicodedata.category(c).startswith("Z"))


def classify_fn(s1_row, tgt_row) -> tuple[str, dict]:
    """Return root-cause category and a dict of computed signals."""
    s1_native = str(s1_row.get("soft_name") or s1_row.get("business_name", "")).strip()
    tgt_native = str(tgt_row.get("soft_name") or tgt_row.get("business_name", "")).strip()
    s1_translit = str(s1_row.get("translit_name", "")).strip()
    tgt_translit = str(tgt_row.get("translit_name", "")).strip()
    s1_addr = str(s1_row.get("clean_address") or s1_row.get("business_address", "")).strip()
    tgt_addr = str(tgt_row.get("clean_address") or tgt_row.get("business_address", "")).strip()
    s1_nums = set(str(n) for n in (s1_row.get("numbers") or []))
    tgt_nums = set(str(n) for n in (tgt_row.get("numbers") or []))

    native_sim = char_ngram_sim(s1_native, tgt_native)
    translit_sim = char_ngram_sim(s1_translit, tgt_translit)
    addr_sim = char_ngram_sim(s1_addr, tgt_addr)
    # Cross-mode similarity: s1_native vs tgt_translit and vice versa
    cross1_sim = char_ngram_sim(s1_native, tgt_translit)  # non-latin s1 vs latin tgt
    cross2_sim = char_ngram_sim(s1_translit, tgt_native)  # latin s1_translit vs tgt native
    num_match = bool(s1_nums & tgt_nums)
    name_len = min(len(s1_native.replace(" ", "")), len(tgt_native.replace(" ", "")))
    edit_dist = levenshtein(s1_translit[:30], tgt_translit[:30]) if s1_translit and tgt_translit else 99

    signals = {
        "s1_native": s1_native[:60],
        "tgt_native": tgt_native[:60],
        "s1_translit": s1_translit[:60],
        "tgt_translit": tgt_translit[:60],
        "s1_addr": s1_addr[:60],
        "tgt_addr": tgt_addr[:60],
        "native_sim": round(native_sim, 3),
        "translit_sim": round(translit_sim, 3),
        "addr_sim": round(addr_sim, 3),
        "cross1_sim": round(cross1_sim, 3),
        "cross2_sim": round(cross2_sim, 3),
        "num_match": num_match,
        "name_len_min": name_len,
        "edit_dist": edit_dist,
    }

    best_translit_sim = max(translit_sim, cross1_sim, cross2_sim)

    # Category decision tree (exclusive, priority-ordered)
    # F: pair IS retrievable at higher K (translit_sim >= 0.30 but presumably rank > 5)
    if best_translit_sim >= 0.30:
        cat = "F_recoverable_higher_K"
    # E: address-only — no name signal at all
    elif (not s1_native or not tgt_native) and addr_sim >= 0.20:
        cat = "E_address_only"
    # C: short-name (one name < 4 chars after stripping spaces)
    elif name_len <= 3:
        cat = "C_short_name"
    # D: translit exists but edit dist is small yet ngram sim is low (phonetic/typo)
    elif s1_translit and tgt_translit and edit_dist <= 4 and best_translit_sim < 0.15:
        cat = "D_phonetic_typo"
    # A: transliteration inconsistency — translit exists but similarity very low
    elif s1_translit and tgt_translit and best_translit_sim < 0.15:
        cat = "A_bad_transliteration"
    # B: one or both lacks translit, char overlap low
    elif (not s1_translit or not tgt_translit) and best_translit_sim < 0.15:
        cat = "B_missing_translit"
    else:
        cat = "B_insufficient_char_overlap"

    signals["category"] = cat
    return cat, signals


def main():
    SEP = "=" * 72
    print(SEP)
    print("CROSS-SCRIPT BLOCKING FN FORENSIC ANALYSIS")
    print(SEP)

    print("\nLoading data...")
    s1_df = load_business_records_df(DATA_DIR / "train_source1.tsv").head(N_S1)
    s2_df = load_business_records_df(DATA_DIR / "train_source2.tsv")
    s3_df = load_business_records_df(DATA_DIR / "train_source3.tsv")
    gt_all = load_ground_truth_dict(DATA_DIR / "train_ground_truth.tsv")
    valid_ids = set(s1_df["entity_id"].astype(str))
    gt_dict = {k: v for k, v in gt_all.items() if k in valid_ids}

    target_df = pd.concat([s2_df, s3_df], ignore_index=True)

    print("Widening records...")
    s1_wide = widen_records_df(s1_df)
    target_wide = widen_records_df(target_df)

    # Build lookup dicts
    s1_lookup = {str(r["entity_id"]): r for _, r in s1_wide.iterrows()}
    tgt_lookup = {str(r["entity_id"]): r for _, r in target_wide.iterrows()}

    def is_cs(s1_id, tgt_id):
        s1n = str(s1_lookup.get(s1_id, {}).get("business_name", ""))
        tgtn = str(tgt_lookup.get(tgt_id, {}).get("business_name", ""))
        return is_non_latin(s1n) != is_non_latin(tgtn)

    # Run baseline blocker (D=OFF)
    print("\nRunning Baseline Blocker (A+B+C, D=OFF)...")
    s1_no_translit = s1_wide.drop(columns=["translit_name"], errors="ignore")
    tgt_no_translit = target_wide.drop(columns=["translit_name"], errors="ignore")
    blocker = DualChannelTFIDFBlocker(
        name_top_k=25, addr_top_k=20, translit_top_k=5,
        name_min_sim=0.15, addr_min_sim=0.20, translit_min_sim=0.30,
        batch_size=2000, show_progress=True,
    )
    cands_base = blocker.generate_candidates(s1_no_translit, tgt_no_translit)

    # Collect cross-script FNs
    cs_fns = []
    for s1_id, true_set in gt_dict.items():
        for tgt_id in true_set:
            if tgt_id not in cands_base.get(s1_id, set()):
                if is_cs(s1_id, tgt_id):
                    cs_fns.append((s1_id, tgt_id))

    print(f"\nTotal cross-script blocking FNs found: {len(cs_fns)}")

    # Classify each FN
    print("Classifying each FN pair...")
    categories = Counter()
    all_signals = []
    for s1_id, tgt_id in cs_fns:
        s1_row = s1_lookup.get(s1_id, {})
        tgt_row = tgt_lookup.get(tgt_id, {})
        cat, signals = classify_fn(s1_row, tgt_row)
        signals["s1_id"] = s1_id
        signals["tgt_id"] = tgt_id
        categories[cat] += 1
        all_signals.append(signals)

    # Also check which channel WOULD retrieve each pair at wider K
    # Use wider name search (K=100) to check retrievability
    print("Checking retrievability with wider name search (K=100)...")
    wider_blocker = DualChannelTFIDFBlocker(
        name_top_k=100, addr_top_k=50, translit_top_k=20,
        name_min_sim=0.10, addr_min_sim=0.10, translit_min_sim=0.10,
        batch_size=2000, show_progress=True,
    )
    cands_wide = wider_blocker.generate_candidates(s1_wide, target_wide)

    n_retrievable_wide = 0
    for s1_id, tgt_id in cs_fns:
        if tgt_id in cands_wide.get(s1_id, set()):
            n_retrievable_wide += 1

    # Channel-only attribution: run each channel independently
    # Channel A only (name, D=OFF)
    blocker_a = DualChannelTFIDFBlocker(
        name_top_k=25, addr_top_k=0, translit_top_k=0,
        name_min_sim=0.15, addr_min_sim=1.0, translit_min_sim=1.0,
        batch_size=2000, show_progress=False,
    )
    cands_a = blocker_a.generate_candidates(s1_no_translit, tgt_no_translit)
    
    in_a_only = set()
    in_addr_only = set()
    for s1_id, tgt_id in cs_fns:
        if tgt_id in cands_a.get(s1_id, set()):
            in_a_only.add((s1_id, tgt_id))

    # Check address retrieval by looking at cands_base minus cands_a
    for s1_id, tgt_id in cs_fns:
        if tgt_id not in cands_a.get(s1_id, set()) and tgt_id in cands_base.get(s1_id, set()):
            in_addr_only.add((s1_id, tgt_id))

    # ------- Report -------
    print(f"\n{SEP}")
    print(f"FORENSIC REPORT: {len(cs_fns)} cross-script blocking FNs")
    print(SEP)

    total = len(cs_fns)
    print(f"\n{'Category':<40} {'Count':>6} {'%':>6}")
    print("-" * 56)
    cat_display = {
        "F_recoverable_higher_K":    "F. Recoverable at higher K (sim>=0.30)",
        "A_bad_transliteration":     "A. Bad/inconsistent transliteration",
        "B_missing_translit":        "B. Missing translit_name for one/both",
        "B_insufficient_char_overlap": "B. Insufficient char overlap",
        "C_short_name":              "C. Short name (<= 3 chars)",
        "D_phonetic_typo":           "D. Phonetic/typo corruption",
        "E_address_only":            "E. Address-only relationship",
    }
    for cat, label in cat_display.items():
        n = categories.get(cat, 0)
        print(f"  {label:<40} {n:>5}  {n/total*100:>5.1f}%")
    print("-" * 56)
    uncategorized = total - sum(categories.values())
    if uncategorized:
        print(f"  {'Uncategorized':<40} {uncategorized:>5}  {uncategorized/total*100:>5.1f}%")

    print(f"\n  Retrievable with wider TF-IDF search (K=100, sim=0.10): {n_retrievable_wide} / {total} ({n_retrievable_wide/total*100:.1f}%)")

    # Similarity distribution per category
    df = pd.DataFrame(all_signals)
    print(f"\n{'Category':<30} {'N':>4} {'AvgNative':>10} {'AvgTranslit':>12} {'AvgAddr':>9} {'NumMatch%':>10}")
    print("-" * 78)
    for cat in sorted(df["category"].unique()):
        sub = df[df["category"] == cat]
        n = len(sub)
        print(
            f"  {cat:<30} {n:>4} "
            f"{sub['native_sim'].mean():>9.3f} "
            f"{sub['translit_sim'].mean():>11.3f} "
            f"{sub['addr_sim'].mean():>8.3f} "
            f"{sub['num_match'].mean()*100:>8.1f}%"
        )

    # Sample 5 pairs per category
    print(f"\n{SEP}")
    print("SAMPLE PAIRS PER CATEGORY")
    print(SEP)
    for cat in sorted(df["category"].unique()):
        sub = df[df["category"] == cat].head(3)
        print(f"\n-- {cat} --")
        for _, row in sub.iterrows():
            print(f"  S1 native   : {row['s1_native']}")
            print(f"  TGT native  : {row['tgt_native']}")
            print(f"  S1 translit : {row['s1_translit']}")
            print(f"  TGT translit: {row['tgt_translit']}")
            print(f"  S1 addr     : {row['s1_addr']}")
            print(f"  TGT addr    : {row['tgt_addr']}")
            print(f"  Sims        : native={row['native_sim']} translit={row['translit_sim']} addr={row['addr_sim']} cross={row['cross1_sim']}")
            print(f"  NumMatch    : {row['num_match']}  NameLen: {row['name_len_min']}  EditDist: {row['edit_dist']}")
            print()

    print(f"\n{SEP}")
    print("CONCLUSION — WHAT MECHANISM CAN RETRIEVE THE 168 FNs?")
    print(SEP)
    f_count = categories.get("F_recoverable_higher_K", 0)
    b_count = categories.get("B_missing_translit", 0) + categories.get("B_insufficient_char_overlap", 0)
    a_count = categories.get("A_bad_transliteration", 0)
    c_count = categories.get("C_short_name", 0)
    e_count = categories.get("E_address_only", 0)
    d_count = categories.get("D_phonetic_typo", 0)

    print(f"\n  F ({f_count}): Wider TF-IDF K. Same character n-gram approach, just higher cap.")
    print(f"  A ({a_count}): Requires semantic/phonetic matching. N-gram won't help.")
    print(f"  B ({b_count}): Requires either BM25, prefix matching, or edit-distance blocking.")
    print(f"  C ({c_count}): Short-name problem. Exact-match or phonetic code (Soundex/Metaphone) blocking.")
    print(f"  D ({d_count}): Phonetic blocking (Soundex/Metaphone/DoubleMetaphone).")
    print(f"  E ({e_count}): Address-only pairs. Wider address channel (higher K) or geohash.")
    print(f"\n  Retrievable at K=100 / sim=0.10: {n_retrievable_wide}/{total}")
    print(f"\n{SEP}")


if __name__ == "__main__":
    main()
