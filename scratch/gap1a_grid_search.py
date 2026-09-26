import os
import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import gc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from parallax.data.contracts import load_business_records_df, load_ground_truth_dict
from parallax.preprocessing.normalizer import widen_records_df

def main():
    print("Loading datasets...")
    data_dir = "data/medium_split_200k"
    
    # 1. Load targets (1 Million)
    s2_df = load_business_records_df(os.path.join(data_dir, "train_source2.tsv"))
    s3_df = load_business_records_df(os.path.join(data_dir, "train_source3.tsv"))
    target_wide = pd.concat([widen_records_df(s2_df), widen_records_df(s3_df)], ignore_index=True)
    del s2_df, s3_df
    gc.collect()
    
    # 2. Load 5K S1 Diagnostic Subset (Random Seed 42)
    s1_df = load_business_records_df(os.path.join(data_dir, "train_source1.tsv"))
    gt_dict = load_ground_truth_dict(os.path.join(data_dir, "train_ground_truth.tsv"))
    s1_sample = s1_df.sample(n=5000, random_state=42).copy()
    s1_wide = widen_records_df(s1_sample)
    del s1_df
    gc.collect()

    country = "India"
    s1_c = s1_wide[s1_wide["country"] == country].reset_index(drop=True)
    tgt_c = target_wide[target_wide["country"] == country].reset_index(drop=True)
    tgt_ids = tgt_c["entity_id"].tolist()
    
    s1_translit = s1_c["translit_name"].fillna("").astype(str).tolist()
    tgt_translit = tgt_c["translit_name"].fillna("").astype(str).tolist()
    
    print(f"Fitting Target TF-IDF (Target Pool: {len(tgt_c):,})...")
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3,3), min_df=1, sublinear_tf=True)
    tgt_mat = vec.fit_transform(tgt_translit).T
    print(f"Transforming S1 TF-IDF (S1 Pool: {len(s1_c):,})...")
    s1_mat = vec.transform(s1_translit)
    
    print("Computing dense similarities batch...")
    batch_sims = s1_mat.dot(tgt_mat)
    
    # Base B1 configuration for baseline retention reference
    B1_K = 20
    B1_MIN_SIM = 0.15
    b1_recovered_pairs = set()

    # We will simulate the grid
    K_GRID = [3, 5, 10]
    SIM_GRID = [0.20, 0.25, 0.30]
    
    results = []

    # First, get the B1 baseline recoveries (so we can calculate % retained)
    for i, s1_id in enumerate(s1_c["entity_id"]):
        r_start = batch_sims.indptr[i]
        r_end = batch_sims.indptr[i + 1]
        scores = batch_sims.data[r_start:r_end]
        col_idx = batch_sims.indices[r_start:r_end]
        
        if len(scores) > 0:
            sort_idx = np.argsort(-scores)
            scores = scores[sort_idx]
            col_idx = col_idx[sort_idx]
            
            # Apply B1 logic
            mask = scores >= B1_MIN_SIM
            c_scores = scores[mask][:B1_K]
            c_cols = col_idx[mask][:B1_K]
            
            gt_cands = gt_dict.get(s1_id, set())
            for idx in c_cols:
                cand_id = tgt_ids[idx]
                if cand_id in gt_cands:
                    b1_recovered_pairs.add((s1_id, cand_id))

    print(f"\nB1 Baseline (K=20, sim=0.15) recovered {len(b1_recovered_pairs)} true pairs in this sample.")

    # Grid Search
    for k in K_GRID:
        for sim in SIM_GRID:
            total_candidates = 0
            s1_hit_cap = 0
            recovered_pairs = set()
            
            for i, s1_id in enumerate(s1_c["entity_id"]):
                r_start = batch_sims.indptr[i]
                r_end = batch_sims.indptr[i + 1]
                scores = batch_sims.data[r_start:r_end]
                col_idx = batch_sims.indices[r_start:r_end]
                
                if len(scores) > 0:
                    sort_idx = np.argsort(-scores)
                    scores = scores[sort_idx]
                    col_idx = col_idx[sort_idx]
                    
                    mask = scores >= sim
                    valid_scores = scores[mask]
                    valid_cols = col_idx[mask]
                    
                    retrieved_k = min(len(valid_scores), k)
                    total_candidates += retrieved_k
                    if retrieved_k == k and len(valid_scores) >= k:
                        s1_hit_cap += 1
                        
                    c_cols = valid_cols[:retrieved_k]
                    gt_cands = gt_dict.get(s1_id, set())
                    for idx in c_cols:
                        cand_id = tgt_ids[idx]
                        if cand_id in gt_cands:
                            recovered_pairs.add((s1_id, cand_id))
            
            avg_cand = total_candidates / len(s1_c)
            # Extrapolate inflation to 200k scale (assuming similar overlap behavior as B1)
            # B1 added 13189 avg cands, extrapolated to 3.0M (+35%).
            # So extrapolation factor = 3,003,503 / (13.189 * 200000 approx) = wait.
            # B1 had exactly 20 candidates per S1 on average. 20 * 200,000 = 4,000,000 raw.
            # 3,003,503 / 4,000,000 = 0.75 non-overlap rate.
            # Projected inflation = avg_cand * 200,000 * 0.75
            proj_inflation_count = avg_cand * 200_000 * 0.75
            baseline_cands = 8_525_511 # From user prompt
            inflation_pct = (proj_inflation_count / baseline_cands) * 100
            
            b1_retention = len(recovered_pairs) / len(b1_recovered_pairs) * 100 if len(b1_recovered_pairs)>0 else 0
            
            results.append({
                "K": k,
                "Min Sim": sim,
                "Avg Cand": avg_cand,
                "Hit Cap %": (s1_hit_cap / len(s1_c)) * 100,
                "Recovered (Sample)": len(recovered_pairs),
                "B1 Retention %": b1_retention,
                "Projected +Cands": proj_inflation_count,
                "Inflation %": inflation_pct
            })

    print("\n" + "="*80)
    print(f"{'K':<4} | {'MinSim':<7} | {'Avg Cand':<10} | {'Hit Cap %':<10} | {'Proj +Cands':<12} | {'Infl %':<8} | {'Retained B1 Rec %':<15}")
    print("-" * 80)
    for r in results:
        print(f"{r['K']:<4} | {r['Min Sim']:<7.2f} | {r['Avg Cand']:<10.2f} | {r['Hit Cap %']:<10.1f} | {int(r['Projected +Cands']):<12,} | {r['Inflation %']:<8.2f} | {r['B1 Retention %']:<15.2f}")
    print("="*80)

if __name__ == "__main__":
    main()
