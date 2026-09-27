import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

# Add local experiment scripts to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from detect_script import has_indic_script, detect_dominant_script

def main():
    print("Loading data...")
    s1_df = pd.read_csv("data/medium_split_200k/train_source1.tsv", sep="\t", nrows=5000)
    s2_df = pd.read_csv("data/medium_split_200k/train_source2.tsv", sep="\t")
    s3_df = pd.read_csv("data/medium_split_200k/train_source3.tsv", sep="\t")
    tgt_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    print("Loading cache...")
    try:
        cache_df = pd.read_parquet("experiments/neural_text/caches/indicxlit_translit_cache_5k.parquet")
        translit_map = cache_df.set_index("entity_id")["indicxlit_name"].to_dict()
    except Exception as e:
        print(f"Failed to load cache: {e}. Ensure the cache exists.")
        return

    def get_translit(row):
        eid = str(row['entity_id'])
        if eid in translit_map and pd.notna(translit_map[eid]) and str(translit_map[eid]).strip():
            return str(translit_map[eid]).strip()
        name = str(row.get('soft_name', ''))
        if not name.strip():
            name = str(row.get('business_name', ''))
        return name.strip()

    print("Mapping transliterations...")
    s1_df['translit_name'] = s1_df.apply(get_translit, axis=1)
    tgt_df['translit_name'] = tgt_df.apply(get_translit, axis=1)

    print("Loading Ground Truth and Baseline...")
    gt = pd.read_csv("data/medium_split_200k/train_ground_truth.tsv", sep="\t")
    gt_true = gt[gt['target'] == 1].copy()
    s1_ids_5k = set(s1_df['entity_id'].astype(str))
    gt_true = gt_true[gt_true['s1_id'].astype(str).isin(s1_ids_5k)]
    gt_true_set = set(zip(gt_true['s1_id'].astype(str), gt_true['cand_id'].astype(str)))

    baseline = pd.read_parquet("baseline_artifacts/candidate_pairs_sample.parquet")
    baseline_pairs = set(zip(baseline['s1_id'].astype(str), baseline['cand_id'].astype(str)))

    tgt_info = tgt_df.set_index(tgt_df['entity_id'].astype(str))
    
    cross_script_bfns = set()
    for pair in gt_true_set:
        if pair not in baseline_pairs:
            if pair[1] in tgt_info.index:
                tgt_row = tgt_info.loc[pair[1]]
                if has_indic_script(str(tgt_row['business_name'])) or has_indic_script(str(tgt_row['business_address'])):
                    cross_script_bfns.add(pair)
    
    print(f"Identified {len(cross_script_bfns)} audited Cross-script BFNs (expected ~171).")

    # Limit to India for the blocking
    s1_india = s1_df[s1_df['country'] == 'India'].reset_index(drop=True)
    tgt_india = tgt_df[tgt_df['country'] == 'India'].reset_index(drop=True)

    s1_translit = s1_india["translit_name"].tolist()
    tgt_translit = tgt_india["translit_name"].tolist()
    tgt_ids = tgt_india["entity_id"].astype(str).tolist()
    s1_ids = s1_india["entity_id"].astype(str).tolist()

    print("Fitting TF-IDF on target translit names...")
    vec_translit = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=1, sublinear_tf=True)
    tgt_translit_mat = vec_translit.fit_transform(tgt_translit).T

    batch_size = 2000
    s1_sims = []
    
    print("Computing similarity matrix...")
    for start_idx in tqdm(range(0, len(s1_translit), batch_size), desc="TF-IDF batches"):
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

    top_ks = [3, 5, 10]
    min_sims = [0.25, 0.30, 0.35, 0.40]

    print("\n========================================")
    print("      CHANNEL D SENSITIVITY ANALYSIS")
    print("========================================")
    
    selected_channel_d_pairs = set()
    selected_scores_map = {}

    for tk in top_ks:
        for ms in min_sims:
            channel_d_pairs = set()
            scores_map = {}
            
            for i, s1_id in enumerate(s1_ids):
                cols, scores = s1_sims[i]
                mask = scores >= ms
                if not np.any(mask): continue
                c_scores = scores[mask]
                c_cols = cols[mask]
                
                if len(c_scores) > tk:
                    top_idx = np.argpartition(c_scores, -tk)[-tk:]
                    c_cols = c_cols[top_idx]
                    c_scores = c_scores[top_idx]
                
                for c, sc in zip(c_cols, c_scores):
                    c_id = tgt_ids[c]
                    channel_d_pairs.add((s1_id, c_id))
                    scores_map[(s1_id, c_id)] = sc
                    
            union_pairs = baseline_pairs | channel_d_pairs
            
            cands_added = len(channel_d_pairs - baseline_pairs)
            total_after_union = len(union_pairs)
            cands_per_s1 = total_after_union / len(s1_ids_5k)
            
            recovered_bfns = cross_script_bfns.intersection(channel_d_pairs)
            recovery_pct = (len(recovered_bfns) / max(1, len(cross_script_bfns))) * 100
            
            tps_in_union = sum(1 for p in union_pairs if p in gt_true_set)
            tp_recovery_rate = tps_in_union / max(1, len(gt_true_set)) * 100
            
            tps_added = sum(1 for p in (channel_d_pairs - baseline_pairs) if p in gt_true_set)
            distractors = cands_added - tps_added
            
            lost_baseline = len(baseline_pairs - union_pairs)
            assert lost_baseline == 0, "Baseline candidates must never be removed!"
            
            unrecovered_bfns = len(cross_script_bfns) - len(recovered_bfns)

            if tk == 5 and ms == 0.30:
                selected_channel_d_pairs = channel_d_pairs
                selected_scores_map = scores_map

            print(f"[top_k={tk:2d} | min_sim={ms:.2f}]")
            print(f"  Channel-D Cands Added: {cands_added:,} (Distractors: {distractors:,} | TPs: {tps_added:,})")
            print(f"  Total Union Cands    : {total_after_union:,} ({cands_per_s1:.2f} cands/S1)")
            print(f"  Cross-Script BFNs    : {len(recovered_bfns)} recovered ({recovery_pct:.1f}%) | {unrecovered_bfns} unrecovered")
            print(f"  True-Pair Recovery   : {tp_recovery_rate:.2f}% (Total TPs: {tps_in_union:,})")
            print(f"  Baseline Lost        : {lost_baseline}")
            print("-" * 50)

    print("\n========================================")
    print(" 171 AUDITED CROSS-SCRIPT BFN TABLE")
    print(" Configuration: top_k=5, min_sim=0.30")
    print("========================================")
    
    s1_info = s1_df.set_index(s1_df['entity_id'].astype(str))

    records = []
    for pair in cross_script_bfns:
        s1_id, cand_id = pair
        baseline_present = pair in baseline_pairs
        channel_d_present = pair in selected_channel_d_pairs
        recovered_after_union = baseline_present or channel_d_present
        sim = selected_scores_map.get(pair, 0.0)
        
        tgt_row = tgt_info.loc[cand_id]
        tgt_script = detect_dominant_script(str(tgt_row['business_name']))
        transliteration = tgt_row['translit_name']
        
        records.append({
            "s1_id": s1_id,
            "true_cand_id": cand_id,
            "baseline_present": baseline_present,
            "channel_d_present": channel_d_present,
            "recovered_after_union": recovered_after_union,
            "similarity": f"{sim:.4f}",
            "target_script": tgt_script,
            "transliteration": transliteration
        })
    
    # Sort for consistent output
    records = sorted(records, key=lambda x: (not x['recovered_after_union'], x['s1_id']))
    
    df_report = pd.DataFrame(records)
    print(df_report.to_markdown(index=False))
    
    df_report.to_csv("scratch/channel_d_bfn_report.csv", index=False)
    print("\nSaved detailed table to scratch/channel_d_bfn_report.csv")

if __name__ == "__main__":
    main()
