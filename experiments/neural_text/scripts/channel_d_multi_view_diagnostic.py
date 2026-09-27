import pandas as pd
import numpy as np
import re
import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from rapidfuzz import fuzz, process
import sys
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

def strip_vowels(text):
    if not isinstance(text, str): return ""
    return re.sub(r'[aeiouAEIOU]', '', text)

def consonant_skeleton(text):
    # Remove vowels, and deduplicate consecutive identical consonants
    if not isinstance(text, str): return ""
    no_vowels = re.sub(r'[aeiouAEIOU\s]', '', text).lower()
    if not no_vowels: return ""
    return re.sub(r'(.)\1+', r'\1', no_vowels)

def main():
    print("Loading data...")
    s1_df = pd.read_csv("data/medium_split_200k/train_source1.tsv", sep="\t", nrows=5000)
    s2_df = pd.read_csv("data/medium_split_200k/train_source2.tsv", sep="\t")
    s3_df = pd.read_csv("data/medium_split_200k/train_source3.tsv", sep="\t")
    tgt_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    # Restrict targets to India
    tgt_india = tgt_df[tgt_df['country'] == 'India'].reset_index(drop=True)
    tgt_info = tgt_india.set_index(tgt_india['entity_id'].astype(str))
    
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

    s1_df['translit_name'] = s1_df.apply(get_translit, axis=1)
    tgt_india['translit_name'] = tgt_india.apply(get_translit, axis=1)

    print("Loading Ground Truth and Identifying 171 Canonical BFNs...")
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
                    
    canonical_bfns = []
    bfn_s1_ids = set()
    for pair in gt_true_set:
        if pair not in baseline_pairs:
            s1, cand = pair
            if cand in tgt_info.index:
                tgt_row = tgt_info.loc[cand]
                if has_indic_script(str(tgt_row['business_name'])):
                    canonical_bfns.append(pair)
                    bfn_s1_ids.add(s1)

    print(f"Identified {len(canonical_bfns)} Canonical BFNs involving {len(bfn_s1_ids)} unique S1 queries.")

    # We only need to compute distances between the S1s in the BFNs and ALL targets.
    s1_bfn_df = s1_df[s1_df['entity_id'].astype(str).isin(bfn_s1_ids)].reset_index(drop=True)
    s1_names = s1_bfn_df['translit_name'].fillna("").str.lower().tolist()
    s1_ids = s1_bfn_df['entity_id'].astype(str).tolist()
    s1_idx_map = {eid: i for i, eid in enumerate(s1_ids)}
    
    tgt_names = tgt_india['translit_name'].fillna("").str.lower().tolist()
    tgt_ids = tgt_india['entity_id'].astype(str).tolist()
    tgt_idx_map = {eid: i for i, eid in enumerate(tgt_ids)}

    print(f"S1 Query Pool: {len(s1_names)} | Target Pool: {len(tgt_names)}")

    def compute_tfidf_similarities(vec, src_texts, tgt_texts):
        # Fit on targets
        tgt_mat = vec.fit_transform(tgt_texts).T
        src_mat = vec.transform(src_texts)
        sim_mat = src_mat.dot(tgt_mat).toarray()
        return sim_mat

    def compute_jaccard_similarities(src_texts, tgt_texts):
        vec = CountVectorizer(analyzer="word", binary=True, min_df=1)
        tgt_mat = vec.fit_transform(tgt_texts)
        src_mat = vec.transform(src_texts)
        
        # Jaccard = intersection / (len1 + len2 - intersection)
        intersection = src_mat.dot(tgt_mat.T).toarray()
        src_sum = np.array(src_mat.sum(axis=1))
        tgt_sum = np.array(tgt_mat.sum(axis=1)).T
        
        union = src_sum + tgt_sum - intersection
        # Avoid div by zero
        union[union == 0] = 1
        return intersection / union

    # --- Precompute variations ---
    s1_vowel_strip = [strip_vowels(t) for t in s1_names]
    tgt_vowel_strip = [strip_vowels(t) for t in tgt_names]
    
    s1_cons = [consonant_skeleton(t) for t in s1_names]
    tgt_cons = [consonant_skeleton(t) for t in tgt_names]

    print("\nComputing Dense Similarity Matrices (S1_BFN x All_Targets)...")
    
    matrices = {}
    
    # TF-IDF Char N-Grams
    matrices["TFIDF_Char_2gram"] = compute_tfidf_similarities(
        TfidfVectorizer(analyzer="char", ngram_range=(2, 2), sublinear_tf=True), s1_names, tgt_names)
        
    matrices["TFIDF_Char_3gram"] = compute_tfidf_similarities(
        TfidfVectorizer(analyzer="char", ngram_range=(3, 3), sublinear_tf=True), s1_names, tgt_names)
        
    matrices["TFIDF_Char_4gram"] = compute_tfidf_similarities(
        TfidfVectorizer(analyzer="char", ngram_range=(4, 4), sublinear_tf=True), s1_names, tgt_names)

    # Token TF-IDF & Jaccard
    matrices["TFIDF_Token"] = compute_tfidf_similarities(
        TfidfVectorizer(analyzer="word", sublinear_tf=True), s1_names, tgt_names)
        
    matrices["Token_Jaccard"] = compute_jaccard_similarities(s1_names, tgt_names)
    
    # Vowel Stripped & Consonant
    matrices["Vowel_Stripped_3gram"] = compute_tfidf_similarities(
        TfidfVectorizer(analyzer="char", ngram_range=(3, 3), sublinear_tf=True), s1_vowel_strip, tgt_vowel_strip)
        
    matrices["Consonant_Skeleton_3gram"] = compute_tfidf_similarities(
        TfidfVectorizer(analyzer="char", ngram_range=(3, 3), sublinear_tf=True), s1_cons, tgt_cons)

    print("Computing RapidFuzz Similarities (Jaro-Winkler & Token Ratio)...")
    # Initialize empty matrices
    jw_mat = np.zeros((len(s1_names), len(tgt_names)), dtype=np.float32)
    ratio_mat = np.zeros((len(s1_names), len(tgt_names)), dtype=np.float32)
    
    for i, s1_text in enumerate(tqdm(s1_names, desc="Fuzzy Scoring")):
        # rapidfuzz cdist returns array of shape (1, len(tgt_names))
        # normalized between 0 and 100, we divide by 100 for 0-1 scale
        jw_mat[i] = process.cdist([s1_text], tgt_names, scorer=fuzz.jaro_winkler)[0] / 100.0
        ratio_mat[i] = process.cdist([s1_text], tgt_names, scorer=fuzz.token_sort_ratio)[0] / 100.0
        
    matrices["Jaro_Winkler"] = jw_mat
    matrices["Token_Sort_Ratio"] = ratio_mat
    
    # Ensemble: Max of Vowel_Stripped, JaroWinkler, and TFIDF_3gram
    matrices["Ensemble_Max_Similarity"] = np.maximum.reduce([
        matrices["TFIDF_Char_3gram"],
        matrices["Vowel_Stripped_3gram"],
        matrices["Jaro_Winkler"],
        matrices["Token_Sort_Ratio"]
    ])

    print("\n========================================")
    print("      MULTI-VIEW RECALL DIAGNOSTIC")
    print("========================================")
    
    print(f"{'Representation':<25} | {'R@1':<6} | {'R@5':<6} | {'R@10':<6} | {'R@20':<6} | {'R@100':<6}")
    print("-" * 65)

    for name, sim_mat in matrices.items():
        ranks = []
        for s1_id, cand_id in canonical_bfns:
            if s1_id not in s1_idx_map or cand_id not in tgt_idx_map:
                continue
            i = s1_idx_map[s1_id]
            tgt_i = tgt_idx_map[cand_id]
            
            sim = sim_mat[i, tgt_i]
            # Rank = number of targets with strictly greater score + 1
            rank = np.sum(sim_mat[i] > sim) + 1
            ranks.append(rank)
            
        ranks = np.array(ranks)
        r1 = np.mean(ranks <= 1) * 100
        r5 = np.mean(ranks <= 5) * 100
        r10 = np.mean(ranks <= 10) * 100
        r20 = np.mean(ranks <= 20) * 100
        r100 = np.mean(ranks <= 100) * 100
        
        print(f"{name:<25} | {r1:<5.1f}% | {r5:<5.1f}% | {r10:<5.1f}% | {r20:<5.1f}% | {r100:<5.1f}%")

if __name__ == "__main__":
    main()
