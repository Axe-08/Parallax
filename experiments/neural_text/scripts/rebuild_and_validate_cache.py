import pandas as pd
import subprocess
import sys
import time
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
    print("--- 1. REBUILDING INDICXLIT CACHE FROM SCRATCH ---")
    # Regenerate completely, ignoring baseline candidates
    cmd_build = [
        sys.executable,
        "experiments/neural_text/scripts/generate_indicxlit_cache.py",
        "--scale", "5000",
        "--candidates-path", "skip.parquet",
        "--no-resume"  # CRITICAL: Force rebuild from scratch
    ]
    print(f"Executing: {' '.join(cmd_build)}")
    t0 = time.time()
    subprocess.run(cmd_build, check=True)
    print(f"Cache rebuild complete in {time.time() - t0:.1f}s.")

    print("\n--- 2. LOADING DATA FOR VALIDATION ---")
    cache_path = Path("experiments/neural_text/caches/indicxlit_translit_cache_5k.parquet")
    cache_df = pd.read_parquet(cache_path)
    
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
                    
    canonical_171_targets = set()
    heuristic_21_targets = set()
    
    for pair in gt_true_set:
        if pair not in baseline_pairs:
            if pair[1] in tgt_info.index:
                tgt_row = tgt_info.loc[pair[1]]
                name_indic = has_indic_script(str(tgt_row['business_name']))
                addr_indic = has_indic_script(str(tgt_row['business_address']))
                
                # Assuming the strict 171 was based on business_name having indic script
                # while the remaining 21 were from address-only or soft-name.
                # If name_indic is True, we count it as canonical.
                if name_indic:
                    canonical_171_targets.add(pair[1])
                elif addr_indic:
                    heuristic_21_targets.add(pair[1])

    print(f"Identified {len(canonical_171_targets)} canonical targets (expected ~171) and {len(heuristic_21_targets)} heuristic-only targets.")
    
    print("\n--- 3. CACHE VALIDATION ---")
    total_cache_rows = len(cache_df)
    duplicates = len(cache_df) - len(cache_df['entity_id'].unique())
    failed_translits = cache_df['indicxlit_name'].isna() | (cache_df['indicxlit_name'].str.strip() == "")
    
    print(f"Total target rows in cache: {total_cache_rows:,}")
    print(f"Duplicates: {duplicates}")
    print(f"Empty/failed transliterations: {failed_translits.sum():,}")
    
    cache_eids = set(cache_df['entity_id'].astype(str))
    canonical_covered = sum(1 for t in canonical_171_targets if t in cache_eids)
    heuristic_covered = sum(1 for t in heuristic_21_targets if t in cache_eids)
    
    print(f"Canonical 171 Coverage: {canonical_covered}/{len(canonical_171_targets)} ({canonical_covered/max(1, len(canonical_171_targets))*100:.1f}%)")
    print(f"Heuristic 21 Coverage: {heuristic_covered}/{len(heuristic_21_targets)} ({heuristic_covered/max(1, len(heuristic_21_targets))*100:.1f}%)")
    
    print("\n10 Representative Examples (Canonical BFN Targets):")
    translit_map = cache_df.set_index("entity_id")["indicxlit_name"].to_dict()
    for t_id in list(canonical_171_targets)[:10]:
        raw_name = str(tgt_info.loc[t_id]['business_name'])
        t_name = translit_map.get(t_id, "<MISSING>")
        print(f"  {t_id}: {raw_name}  --->  {t_name}")
        
    print("\n--- 4. EXECUTING CHANNEL D ---")
    # Execute Channel D script directly. It uses its own logic (which uses the 192 total, 
    # but the logs will show exactly what it recovers).
    cmd_d = [
        sys.executable,
        "experiments/neural_text/scripts/channel_d_blocking.py"
    ]
    subprocess.run(cmd_d, check=True)

if __name__ == "__main__":
    main()
