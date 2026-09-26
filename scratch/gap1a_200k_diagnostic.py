import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import gc

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from parallax.data.contracts import load_business_records_df, load_ground_truth_dict
from parallax.preprocessing.normalizer import widen_records_df

def main():
    print("Loading 200K datasets...")
    data_dir = "data/medium_split_200k"
    
    # Load targets (1 Million)
    s2_df = load_business_records_df(os.path.join(data_dir, "train_source2.tsv"))
    s3_df = load_business_records_df(os.path.join(data_dir, "train_source3.tsv"))
    target_wide = pd.concat([widen_records_df(s2_df), widen_records_df(s3_df)], ignore_index=True)
    del s2_df, s3_df
    gc.collect()
    
    # Load S1 (sample of 2,000 entities to fit in memory and speed up)
    s1_df = load_business_records_df(os.path.join(data_dir, "train_source1.tsv"))
    gt_dict = load_ground_truth_dict(os.path.join(data_dir, "train_ground_truth.tsv"))
    
    # Let's find some known cross-script BFNs from the 200k set to guarantee they are in the sample
    # Wait, we don't know the exact 1,821 BFNs. We will just use all S1s that have transliterated names.
    # To maximize info, we will take 5,000 random S1s.
    s1_sample = s1_df.sample(n=5000, random_state=42).copy()
    s1_wide = widen_records_df(s1_sample)
    del s1_df
    gc.collect()

    print(f"Target Pool: {len(target_wide):,} | S1 Sample: {len(s1_wide):,}")

    country = "India"
    s1_c = s1_wide[s1_wide["country"] == country].reset_index(drop=True)
    tgt_c = target_wide[target_wide["country"] == country].reset_index(drop=True)
    tgt_ids = tgt_c["entity_id"].tolist()
    
    print(f"\nAnalyzing Country: {country}")
    print(f"Target Pool: {len(tgt_c):,} | S1 Pool: {len(s1_c):,}")
    
    s1_translit = s1_c["translit_name"].fillna("").astype(str).tolist()
    tgt_translit = tgt_c["translit_name"].fillna("").astype(str).tolist()
    
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3,3), min_df=1, sublinear_tf=True)
    print("Fitting Target TF-IDF...")
    tgt_mat = vec.fit_transform(tgt_translit).T
    print("Transforming S1 TF-IDF...")
    s1_mat = vec.transform(s1_translit)
    
    # Evaluate collision rates at different thresholds
    print("\nComputing similarities...")
    batch_sims = s1_mat.dot(tgt_mat)
    
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    stats = {t: [] for t in thresholds}
    
    true_pair_sims = []
    true_pair_ranks = []
    
    for i, s1_id in enumerate(s1_c["entity_id"]):
        r_start = batch_sims.indptr[i]
        r_end = batch_sims.indptr[i + 1]
        scores = batch_sims.data[r_start:r_end]
        col_idx = batch_sims.indices[r_start:r_end]
        
        # Sort scores descending
        if len(scores) > 0:
            sort_idx = np.argsort(-scores)
            scores = scores[sort_idx]
            col_idx = col_idx[sort_idx]
            cand_ids = [tgt_ids[idx] for idx in col_idx]
            
            # Record GT metrics
            gt_cands = gt_dict.get(s1_id, set())
            for rank, (cand, score) in enumerate(zip(cand_ids, scores)):
                if cand in gt_cands:
                    true_pair_sims.append(score)
                    true_pair_ranks.append(rank + 1)
        
        # Count collisions at thresholds
        for t in thresholds:
            stats[t].append(np.sum(scores >= t))
            
    print("\n=== COLLISION STATS (Avg Candidates per S1 at Threshold) ===")
    for t in thresholds:
        counts = stats[t]
        print(f"Threshold >= {t:.2f}:")
        print(f"  Avg candidates passing: {np.mean(counts):.1f}")
        print(f"  P50 passing: {np.median(counts):.1f}")
        print(f"  P90 passing: {np.percentile(counts, 90):.1f}")
        print(f"  % of S1 hitting K=20 limit: {np.mean(np.array(counts) >= 20)*100:.1f}%")
        
    print("\n=== TRUE PAIR RECOVERIES (Sample Size: {}) ===".format(len(true_pair_sims)))
    if len(true_pair_sims) > 0:
        sims = np.array(true_pair_sims)
        ranks = np.array(true_pair_ranks)
        print(f"Similarity - Min: {np.min(sims):.3f} | Median: {np.median(sims):.3f} | P25: {np.percentile(sims, 25):.3f} | P10: {np.percentile(sims, 10):.3f}")
        print(f"Rank - Max: {np.max(ranks)} | Median: {np.median(ranks)} | P90: {np.percentile(ranks, 90)} | P95: {np.percentile(ranks, 95)}")
        print(f"% Recovered if K=5:  {np.mean(ranks <= 5)*100:.1f}%")
        print(f"% Recovered if K=10: {np.mean(ranks <= 10)*100:.1f}%")
        print(f"% Recovered if K=20: {np.mean(ranks <= 20)*100:.1f}%")
        print(f"% Passing Sim>=0.20: {np.mean(sims >= 0.20)*100:.1f}%")
        print(f"% Passing Sim>=0.25: {np.mean(sims >= 0.25)*100:.1f}%")
        print(f"% Passing Sim>=0.30: {np.mean(sims >= 0.30)*100:.1f}%")
        
if __name__ == "__main__":
    main()
