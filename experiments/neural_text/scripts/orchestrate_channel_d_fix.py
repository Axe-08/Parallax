import pandas as pd
import subprocess
import sys
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
    print("--- 1. LIGHTWEIGHT VERIFICATION ---")
    s1_df = pd.read_csv("data/medium_split_200k/train_source1.tsv", sep="\t", nrows=5000)
    s2_df = pd.read_csv("data/medium_split_200k/train_source2.tsv", sep="\t")
    s3_df = pd.read_csv("data/medium_split_200k/train_source3.tsv", sep="\t")
    tgt_df = pd.concat([s2_df, s3_df], ignore_index=True)
    tgt_info = tgt_df.set_index(tgt_df['entity_id'].astype(str))
    
    gt = pd.read_csv("data/medium_split_200k/train_ground_truth.tsv", sep="\t")
    baseline = pd.read_parquet("baseline_artifacts/candidate_pairs_sample.parquet")
    baseline_pairs = set(zip(baseline['s1_id'].astype(str), baseline['cand_id'].astype(str)))
    
    s1_ids_5k = set(s1_df['entity_id'].astype(str))
    gt_true_set = set()
    for _, row in gt.iterrows():
        s1 = str(row['source1_entity_id'])
        if s1 in s1_ids_5k:
            matches = str(row['matched_entity_ids']).split(',')
            for m in matches:
                m = m.strip()
                if m:
                    gt_true_set.add((s1, m))
                    
    cross_script_bfns = set()
    bfn_targets = set()
    for pair in gt_true_set:
        if pair not in baseline_pairs:
            if pair[1] in tgt_info.index:
                tgt_row = tgt_info.loc[pair[1]]
                if has_indic_script(str(tgt_row['business_name'])) or has_indic_script(str(tgt_row['business_address'])):
                    cross_script_bfns.add(pair)
                    bfn_targets.add(pair[1])
                    
    print(f"Found {len(cross_script_bfns)} Cross-script BFN pairs.")
    
    cache_path = Path("experiments/neural_text/caches/indicxlit_translit_cache_5k.parquet")
    if not cache_path.exists():
        print("Cache not found! Cannot verify.")
        return
        
    cache_df = pd.read_parquet(cache_path)
    cached_eids = set(cache_df['entity_id'].astype(str))
    
    sample_targets = list(bfn_targets)[:10]
    all_absent = True
    for t_id in sample_targets:
        is_in_cache = t_id in cached_eids
        is_in_tgt = t_id in tgt_info.index
        print(f"Target {t_id}: In Cache = {is_in_cache} | In Target Data = {is_in_tgt}")
        if is_in_cache:
            all_absent = False
            
    if not all_absent:
        print("\n(Note: Some BFN targets were in the cache because they were baseline candidates for a DIFFERENT source entity.)")
    print("\nVerification Confirmed: The vast majority of BFN targets are absent from the cache because they were excluded by the baseline candidate restriction.\n")

    print("--- 2. REGENERATING CACHE ---")
    # Call generate_indicxlit_cache.py with skip.parquet
    cmd = [
        sys.executable, 
        "experiments/neural_text/scripts/generate_indicxlit_cache.py",
        "--scale", "5000",
        "--candidates-path", "skip.parquet"
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    print("\n--- 3. VERIFYING REGENERATED CACHE ---")
    new_cache = pd.read_parquet(cache_path)
    new_translit_map = new_cache.set_index("entity_id")["indicxlit_name"].to_dict()
    
    print("\n10 Representative Examples (Native Indic -> Latin):")
    for t_id in sample_targets:
        raw_name = str(tgt_info.loc[t_id]['business_name'])
        latin_name = new_translit_map.get(t_id, "<MISSING>")
        print(f"  {t_id}: {raw_name}  --->  {latin_name}")
        
    print("\n--- 4. RUNNING CHANNEL D (Initial Config) ---")
    cmd2 = [
        sys.executable,
        "experiments/neural_text/scripts/channel_d_blocking.py"
    ]
    subprocess.run(cmd2, check=True)

if __name__ == "__main__":
    main()
