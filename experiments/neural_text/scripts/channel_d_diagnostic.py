import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from pathlib import Path

def has_indic_script(text):
    if not isinstance(text, str):
        return False
    return any(
        (0x0900 <= ord(c) <= 0x097F) or  # Devanagari
        (0x0980 <= ord(c) <= 0x09FF) or  # Bengali
        (0x0A00 <= ord(c) <= 0x0A7F) or  # Gurmukhi
        (0x0A80 <= ord(c) <= 0x0AFF) or  # Gujarati
        (0x0B00 <= ord(c) <= 0x0B7F) or  # Oriya
        (0x0B80 <= ord(c) <= 0x0BFF) or  # Tamil
        (0x0C00 <= ord(c) <= 0x0C7F) or  # Telugu
        (0x0C80 <= ord(c) <= 0x0CFF) or  # Kannada
        (0x0D00 <= ord(c) <= 0x0D7F)     # Malayalam
        for c in text
    )

def main():
    print("Loading data...")
    s1_df = pd.read_csv("data/medium_split_200k/train_source1.tsv", sep="\t", nrows=5000)
    s2_df = pd.read_csv("data/medium_split_200k/train_source2.tsv", sep="\t")
    s3_df = pd.read_csv("data/medium_split_200k/train_source3.tsv", sep="\t")
    tgt_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    # Filter to India
    s1_india = s1_df[s1_df['country'] == 'India'].reset_index(drop=True)
    tgt_india = tgt_df[tgt_df['country'] == 'India'].reset_index(drop=True)
    
    print("Loading transliteration cache...")
    cache_path = Path("experiments/neural_text/caches/indicxlit_translit_cache_5k.parquet")
    cache_df = pd.read_parquet(cache_path)
    translit_map = cache_df.set_index("entity_id")["indicxlit_name"].to_dict()
    
    def get_translit(row):
        eid = str(row['entity_id'])
        if eid in translit_map and pd.notna(translit_map[eid]) and str(translit_map[eid]).strip():
            return str(translit_map[eid]).strip()
        name = str(row.get('soft_name', ''))
        if not name.strip():
            name = str(row.get('business_name', ''))
        return name.strip()

    s1_india['translit_name'] = s1_india.apply(get_translit, axis=1)
    tgt_india['translit_name'] = tgt_india.apply(get_translit, axis=1)

    s1_translit = s1_india["translit_name"].tolist()
    tgt_translit = tgt_india["translit_name"].tolist()
    tgt_ids = tgt_india["entity_id"].astype(str).tolist()
    s1_ids = s1_india["entity_id"].astype(str).tolist()
    s1_idx_map = {s1_id: i for i, s1_id in enumerate(s1_ids)}
    tgt_idx_map = {tgt_id: i for i, tgt_id in enumerate(tgt_ids)}

    print("Loading Ground Truth and Baseline...")
    gt = pd.read_csv("data/medium_split_200k/train_ground_truth.tsv", sep="\t")
    baseline = pd.read_parquet("baseline_artifacts/candidate_pairs_sample.parquet")
    baseline_pairs = set(zip(baseline['s1_id'].astype(str), baseline['cand_id'].astype(str)))
    
    s1_ids_5k = set(s1_ids)
    gt_true_set = set()
    for _, row in gt.iterrows():
        s1 = str(row['source1_entity_id'])
        if s1 in s1_ids_5k:
            matches = str(row['matched_entity_ids']).split(',')
            for m in matches:
                m = m.strip()
                if m:
                    gt_true_set.add((s1, m))
                    
    tgt_info = tgt_df.set_index(tgt_df['entity_id'].astype(str))
    canonical_bfns = set()
    for pair in gt_true_set:
        if pair not in baseline_pairs:
            s1, cand = pair
            if cand in tgt_info.index:
                tgt_row = tgt_info.loc[cand]
                if has_indic_script(str(tgt_row['business_name'])):
                    # Strictly business_name for canonical 171
                    canonical_bfns.add(pair)

    print(f"Identified {len(canonical_bfns)} canonical BFNs.")

    print("Fitting TF-IDF...")
    vec_translit = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=1, sublinear_tf=True)
    tgt_translit_mat = vec_translit.fit_transform(tgt_translit).T
    
    batch_size = 2000
    s1_sims = []
    print("Computing full similarity matrix...")
    for start_idx in tqdm(range(0, len(s1_translit), batch_size)):
        end_idx = min(start_idx + batch_size, len(s1_translit))
        batch_mat = vec_translit.transform(s1_translit[start_idx:end_idx])
        batch_sims = batch_mat.dot(tgt_translit_mat)
        for i in range(batch_sims.shape[0]):
            r_start = batch_sims.indptr[i]
            r_end = batch_sims.indptr[i+1]
            if r_start == r_end:
                s1_sims.append((np.array([]), np.array([])))
            else:
                s1_sims.append((batch_sims.indices[r_start:r_end], batch_sims.data[r_start:r_end]))

    print("\nEvaluating True Targets (Unthresholded)...")
    bfn_sims = []
    bfn_ranks = []
    
    for s1_id, cand_id in canonical_bfns:
        if s1_id not in s1_idx_map or cand_id not in tgt_idx_map:
            continue
            
        i = s1_idx_map[s1_id]
        tgt_i = tgt_idx_map[cand_id]
        
        cols, scores = s1_sims[i]
        
        match_idx = np.where(cols == tgt_i)[0]
        if len(match_idx) > 0:
            sim = scores[match_idx[0]]
        else:
            sim = 0.0
            
        # Compute rank: number of scores strictly greater than this sim, plus 1
        rank = np.sum(scores > sim) + 1
        
        bfn_sims.append(sim)
        bfn_ranks.append(rank)

    bfn_sims = np.array(bfn_sims)
    bfn_ranks = np.array(bfn_ranks)

    print("\n--- SIMILARITY DISTRIBUTION (Canonical BFNs) ---")
    print(f"Min:    {np.min(bfn_sims):.4f}")
    print(f"P25:    {np.percentile(bfn_sims, 25):.4f}")
    print(f"Median: {np.median(bfn_sims):.4f}")
    print(f"P75:    {np.percentile(bfn_sims, 75):.4f}")
    print(f"P95:    {np.percentile(bfn_sims, 95):.4f}")
    print(f"Max:    {np.max(bfn_sims):.4f}")

    print("\n--- RECALL @ K (Canonical BFNs) ---")
    print(f"Recall@1:  {np.mean(bfn_ranks <= 1)*100:.1f}%")
    print(f"Recall@3:  {np.mean(bfn_ranks <= 3)*100:.1f}%")
    print(f"Recall@5:  {np.mean(bfn_ranks <= 5)*100:.1f}%")
    print(f"Recall@10: {np.mean(bfn_ranks <= 10)*100:.1f}%")
    print(f"Recall@20: {np.mean(bfn_ranks <= 20)*100:.1f}%")

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    print("\n--- THRESHOLD ANALYSIS (Using top_k=20 as upper bound) ---")
    # For candidate volume, assume a reasonable top_k like 20 so it doesn't explode infinitely.
    # Actually, the prompt says "candidate volume generated over the complete 5K population at those same thresholds". 
    # Usually we pair min_sim with top_k. If we don't use top_k, candidate volume might be millions.
    # Let's compute exact candidate volume strictly by threshold (no top_k limit) to answer the prompt exactly.
    
    print(f"{'Thresh':<8} | {'True >= Thresh':<14} | {'Total Cands':<12} | {'New Added':<10} | {'171 Recovery %'}")
    print("-" * 75)
    
    for th in thresholds:
        true_ge_th = np.sum(bfn_sims >= th)
        
        # Calculate full volume across all S1s at this threshold (no top_k)
        total_cands = 0
        new_cands = 0
        recovered_171 = 0
        
        for i, s1_id in enumerate(s1_ids):
            cols, scores = s1_sims[i]
            mask = scores >= th
            c_cols = cols[mask]
            
            total_cands += len(c_cols)
            
            # Check overlap with baseline
            for c in c_cols:
                cand_id = tgt_ids[c]
                pair = (s1_id, cand_id)
                if pair not in baseline_pairs:
                    new_cands += 1
                if pair in canonical_bfns:
                    recovered_171 += 1
                    
        recovery_pct = (recovered_171 / max(1, len(canonical_bfns))) * 100
        
        print(f"{th:<8.2f} | {true_ge_th:<14} | {total_cands:<12,d} | {new_cands:<10,d} | {recovery_pct:.1f}%")

if __name__ == "__main__":
    main()
