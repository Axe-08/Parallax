"""
Phase 1: Address-Rank Audit for Cross-Script FNs
================================================
Identifies the 171 cross-script blocking FNs from the Baseline (A+B+C, D=OFF, K=25/20).
Calculates the exact rank of the true target inside the Address TF-IDF channel.
"""
from __future__ import annotations
import sys
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from parallax.data.contracts import load_business_records_df, load_ground_truth_dict
from parallax.preprocessing.normalizer import widen_records_df
from parallax.blocking.tfidf_blocker import DualChannelTFIDFBlocker
from sklearn.metrics.pairwise import cosine_similarity

N_S1 = 5000
DATA_DIR = Path("data/medium_split_200k")

def is_non_latin(text: str) -> bool:
    return any(ord(c) > 0x024F for c in str(text) if not unicodedata.category(c).startswith("Z"))

def char_ngram_sim(a: str, b: str, n: int = 3) -> float:
    if not a.strip() or not b.strip(): return 0.0
    try:
        v = TfidfVectorizer(analyzer="char", ngram_range=(n, n), sublinear_tf=True)
        m = v.fit_transform([a, b])
        return float(cosine_similarity(m[0:1], m[1:2])[0, 0])
    except Exception:
        return 0.0

def _get_native_addrs(df: pd.DataFrame) -> list[str]:
    if "clean_address" in df:
        p_s = df["clean_address"].fillna("").astype(str).str.strip()
    elif "business_address" in df:
        p_s = df["business_address"].fillna("").astype(str).str.strip()
    else:
        p_s = pd.Series([""] * len(df), index=df.index)
    return [str(x) for x in p_s.tolist()]

def main():
    print("=" * 72)
    print("PHASE 1 — ADDRESS-RANK AUDIT")
    print("=" * 72)

    s1_df = load_business_records_df(DATA_DIR / "train_source1.tsv").head(N_S1)
    target_df = pd.concat([
        load_business_records_df(DATA_DIR / "train_source2.tsv"),
        load_business_records_df(DATA_DIR / "train_source3.tsv")
    ], ignore_index=True)
    gt_all = load_ground_truth_dict(DATA_DIR / "train_ground_truth.tsv")

    valid_ids = set(s1_df["entity_id"].astype(str))
    gt_dict = {k: v for k, v in gt_all.items() if k in valid_ids}

    s1_wide = widen_records_df(s1_df)
    target_wide = widen_records_df(target_df)

    s1_lookup = {str(r["entity_id"]): r for _, r in s1_wide.iterrows()}
    tgt_lookup = {str(r["entity_id"]): r for _, r in target_wide.iterrows()}

    def is_cs(s1_id, tgt_id):
        s1n = str(s1_lookup.get(s1_id, {}).get("business_name", ""))
        tgtn = str(tgt_lookup.get(tgt_id, {}).get("business_name", ""))
        return is_non_latin(s1n) != is_non_latin(tgtn)

    print("\nRunning Baseline Blocker (A+B+C, D=OFF, NameK=25, AddrK=20)...")
    s1_no_translit = s1_wide.drop(columns=["translit_name"], errors="ignore")
    tgt_no_translit = target_wide.drop(columns=["translit_name"], errors="ignore")
    
    blocker = DualChannelTFIDFBlocker(
        name_top_k=25, addr_top_k=20, translit_top_k=0,
        name_min_sim=0.15, addr_min_sim=0.20, translit_min_sim=1.0,
        batch_size=2000, show_progress=False,
    )
    cands_base = blocker.generate_candidates(s1_no_translit, tgt_no_translit)

    cs_fns = []
    for s1_id, true_set in gt_dict.items():
        for tgt_id in true_set:
            if tgt_id not in cands_base.get(s1_id, set()):
                if is_cs(s1_id, tgt_id):
                    cs_fns.append((s1_id, tgt_id))
    print(f"Total cross-script FNs: {len(cs_fns)}")

    print("\nExtracting Address Channels for Ranking...")
    # Get the address strings
    s1_addrs = _get_native_addrs(s1_wide)
    tgt_addrs = _get_native_addrs(target_wide)
    tgt_ids = target_wide["entity_id"].astype(str).tolist()

    # Need to rank them correctly (grouped by country as in blocker)
    # We will compute ranks only for the countries present in FNs
    fn_countries = set()
    fn_by_s1 = defaultdict(list)
    for s1_id, tgt_id in cs_fns:
        c = s1_lookup[s1_id]["country"]
        fn_countries.add(c)
        fn_by_s1[s1_id].append(tgt_id)
        
    ranks = []
    for c in fn_countries:
        c_s1_wide = s1_wide[s1_wide["country"] == c].reset_index(drop=True)
        c_tgt_wide = target_wide[target_wide["country"] == c].reset_index(drop=True)
        c_tgt_ids = c_tgt_wide["entity_id"].astype(str).tolist()
        
        c_s1_addrs = _get_native_addrs(c_s1_wide)
        c_tgt_addrs = _get_native_addrs(c_tgt_wide)
        
        min_df_addr = 2 if len(c_tgt_addrs) > 500 else 1
        max_df_addr = 0.40 if len(c_tgt_addrs) > 500 else 1.0
        vec_addr = TfidfVectorizer(
            analyzer="char", ngram_range=(3, 3),
            min_df=min_df_addr, max_df=max_df_addr, sublinear_tf=True
        )
        try:
            tgt_mat = vec_addr.fit_transform(c_tgt_addrs).T
        except ValueError:
            continue
            
        for i, row in c_s1_wide.iterrows():
            s1_id = str(row["entity_id"])
            if s1_id not in fn_by_s1:
                continue
                
            s1_addr = c_s1_addrs[i]
            if not s1_addr.strip():
                for tgt_id in fn_by_s1[s1_id]:
                    ranks.append({"s1_id": s1_id, "tgt_id": tgt_id, "sim": 0.0, "rank": -1, "cat": "no_addr"})
                continue
                
            try:
                s1_mat = vec_addr.transform([s1_addr])
            except ValueError:
                for tgt_id in fn_by_s1[s1_id]:
                    ranks.append({"s1_id": s1_id, "tgt_id": tgt_id, "sim": 0.0, "rank": -1, "cat": "no_addr"})
                continue
                
            sims = s1_mat.dot(tgt_mat).toarray()[0]
            for tgt_id in fn_by_s1[s1_id]:
                try:
                    tgt_idx = c_tgt_ids.index(tgt_id)
                    sim = float(sims[tgt_idx])
                except ValueError:
                    sim = 0.0
                    
                if sim < 0.20:
                    ranks.append({"s1_id": s1_id, "tgt_id": tgt_id, "sim": sim, "rank": -1, "cat": "sim_too_low"})
                    continue
                    
                # Rank computation: how many candidates have sim strictly greater?
                # (1-indexed rank)
                better_count = np.sum(sims > sim)
                same_count = np.sum(sims == sim)
                # best case rank is better_count + 1
                ranks.append({
                    "s1_id": s1_id, 
                    "tgt_id": tgt_id, 
                    "sim": sim, 
                    "rank": better_count + 1, 
                    "cat": "ranked"
                })

    df = pd.DataFrame(ranks)
    
    cat_counts = df["cat"].value_counts()
    print(f"\nBreakdown by category:")
    for cat, count in cat_counts.items():
        print(f"  {cat}: {count}")
        
    ranked = df[df["cat"] == "ranked"].copy()
    if not ranked.empty:
        r_vals = ranked["rank"]
        print("\nAmong those passing min_sim=0.20:")
        n = len(ranked)
        n20 = sum(r_vals <= 20)
        n50 = sum((r_vals > 20) & (r_vals <= 50))
        n_above = sum(r_vals > 50)
        print(f"  Rank <= 20: {n20} ({n20/n*100:.1f}%)")
        print(f"  Rank 21-50: {n50} ({n50/n*100:.1f}%)")
        print(f"  Rank > 50: {n_above} ({n_above/n*100:.1f}%)")
        print(f"  Median rank: {r_vals.median()}")
        print(f"  P75 rank: {r_vals.quantile(0.75)}")
        print(f"  P90 rank: {r_vals.quantile(0.90)}")
        print(f"  P95 rank: {r_vals.quantile(0.95)}")
        
        # We need to know % of TOTAL FNs (171)
        print("\nAs % of ALL 171 cross-script FNs:")
        print(f"  Rank 21-50: {n50/171*100:.1f}%")
        print(f"  Rank > 50: {n_above/171*100:.1f}%")
    else:
        print("\nNo candidates passed min_sim=0.20.")

if __name__ == "__main__":
    main()
