import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from rapidfuzz import fuzz, process
import sys
from tqdm import tqdm

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
    if not isinstance(text, str): return ""
    no_vowels = re.sub(r'[aeiouAEIOU\s]', '', text).lower()
    if not no_vowels: return ""
    return re.sub(r'(.)\1+', r'\1', no_vowels)

def get_rank_stats(scores, sim):
    strictly_greater = np.sum(scores > sim)
    equal = np.sum(scores == sim)
    best_rank = strictly_greater + 1
    worst_rank = strictly_greater + equal
    return best_rank, worst_rank, equal

def main():
    print("Loading data...")
    s1_df = pd.read_csv("data/medium_split_200k/train_source1.tsv", sep="\t", nrows=5000)
    s2_df = pd.read_csv("data/medium_split_200k/train_source2.tsv", sep="\t")
    s3_df = pd.read_csv("data/medium_split_200k/train_source3.tsv", sep="\t")
    tgt_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    tgt_india = tgt_df[tgt_df['country'] == 'India'].reset_index(drop=True)
    tgt_info = tgt_india.set_index(tgt_india['entity_id'].astype(str))
    
    print("Loading transliteration cache...")
    cache_df = pd.read_parquet("experiments/neural_text/caches/indicxlit_translit_cache_5k.parquet")
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
    
    # Pre-calculate boolean mask for Indic targets
    tgt_is_indic = tgt_india.apply(lambda r: has_indic_script(str(r.get('business_name', ''))), axis=1).values

    print("Identifying Canonical 171 BFNs...")
    gt = pd.read_csv("data/medium_split_200k/train_ground_truth.tsv", sep="\t")
    baseline = pd.read_parquet("baseline_artifacts/candidate_pairs_sample.parquet")
    baseline_pairs = set(zip(baseline['s1_id'].astype(str), baseline['cand_id'].astype(str)))
    
    s1_ids_5k = set(s1_df['entity_id'].astype(str))
    gt_true_set = set()
    for _, row in gt.iterrows():
        s1 = str(row['source1_entity_id'])
        if s1 in s1_ids_5k:
            for m in str(row['matched_entity_ids']).split(','):
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

    print(f"Isolated {len(canonical_bfns)} canonical BFNs.")

    s1_bfn_df = s1_df[s1_df['entity_id'].astype(str).isin(bfn_s1_ids)].reset_index(drop=True)
    s1_names = s1_bfn_df['translit_name'].fillna("").str.lower().tolist()
    s1_ids = s1_bfn_df['entity_id'].astype(str).tolist()
    
    tgt_names = tgt_india['translit_name'].fillna("").str.lower().tolist()
    tgt_ids = tgt_india['entity_id'].astype(str).tolist()
    
    s1_idx_map = {eid: i for i, eid in enumerate(s1_ids)}
    tgt_idx_map = {eid: i for i, eid in enumerate(tgt_ids)}

    s1_vowel = [strip_vowels(t) for t in s1_names]
    tgt_vowel = [strip_vowels(t) for t in tgt_names]
    s1_cons = [consonant_skeleton(t) for t in s1_names]
    tgt_cons = [consonant_skeleton(t) for t in tgt_names]

    reps = [
        ("TFIDF_Char_2gram", TfidfVectorizer(analyzer="char", ngram_range=(2, 2), sublinear_tf=True), s1_names, tgt_names),
        ("TFIDF_Char_3gram", TfidfVectorizer(analyzer="char", ngram_range=(3, 3), sublinear_tf=True), s1_names, tgt_names),
        ("TFIDF_Char_4gram", TfidfVectorizer(analyzer="char", ngram_range=(4, 4), sublinear_tf=True), s1_names, tgt_names),
        ("TFIDF_Token", TfidfVectorizer(analyzer="word", sublinear_tf=True), s1_names, tgt_names),
        ("Vowel_Stripped_3gram", TfidfVectorizer(analyzer="char", ngram_range=(3, 3), sublinear_tf=True), s1_vowel, tgt_vowel),
        ("Consonant_Skel_3gram", TfidfVectorizer(analyzer="char", ngram_range=(3, 3), sublinear_tf=True), s1_cons, tgt_cons),
    ]

    print("\nEvaluating Representations (Streaming Memory-Efficient)...")
    
    results_all = {name: [] for name, _, _, _ in reps}
    results_indic = {name: [] for name, _, _, _ in reps}
    
    # Extra reps for fuzzy
    results_all["Jaro_Winkler"] = []
    results_indic["Jaro_Winkler"] = []
    results_all["Token_Sort_Ratio"] = []
    results_indic["Token_Sort_Ratio"] = []
    results_all["Ensemble_Max"] = []
    results_indic["Ensemble_Max"] = []

    # Prepare vectors
    vec_models = {}
    for name, vec, s1_src, tgt_src in reps:
        tgt_mat = vec.fit_transform(tgt_src)
        s1_mat = vec.transform(s1_src)
        vec_models[name] = (s1_mat, tgt_mat.T)
        
    for i, s1_id in enumerate(tqdm(s1_ids, desc="Querying S1 BFNs")):
        # Target scores for this specific S1 query across all representations
        scores_dict = {}
        for name, (s1_mat, tgt_mat_T) in vec_models.items():
            scores_dict[name] = s1_mat[i].dot(tgt_mat_T).toarray()[0]
            
        jw_scores = process.cdist([s1_names[i]], tgt_names, scorer=fuzz.jaro_winkler)[0] / 100.0
        ts_scores = process.cdist([s1_names[i]], tgt_names, scorer=fuzz.token_sort_ratio)[0] / 100.0
        
        scores_dict["Jaro_Winkler"] = jw_scores
        scores_dict["Token_Sort_Ratio"] = ts_scores
        
        scores_dict["Ensemble_Max"] = np.maximum.reduce([
            scores_dict["TFIDF_Char_3gram"],
            scores_dict["Vowel_Stripped_3gram"],
            scores_dict["Jaro_Winkler"],
            scores_dict["Token_Sort_Ratio"]
        ])
        
        for cand_id in [c for s, c in canonical_bfns if s == s1_id]:
            tgt_i = tgt_idx_map[cand_id]
            for rep_name, scores in scores_dict.items():
                sim = scores[tgt_i]
                
                # Universe 1: ALL Indian targets
                best_all, worst_all, eq_all = get_rank_stats(scores, sim)
                results_all[rep_name].append((sim, best_all, worst_all, eq_all))
                
                # Universe 2: Indic-script-only targets
                indic_scores = scores[tgt_is_indic]
                best_ind, worst_ind, eq_ind = get_rank_stats(indic_scores, sim)
                results_indic[rep_name].append((sim, best_ind, worst_ind, eq_ind))

    def print_report(results, universe_name):
        print(f"\n=========================================================================")
        print(f" {universe_name} UNIVERSE DIAGNOSTIC")
        print(f"=========================================================================")
        header = f"{'Representation':<22} | {'Med Sim':<7} | {'Z-Score':<7} | {'Med Rank':<9} | {'R@1':<6} | {'R@5':<6} | {'R@10':<6} | {'R@20':<6} | {'R@100':<6}"
        print(header)
        print("-" * len(header))
        
        for name in results_all.keys():
            res = results[name]
            sims = np.array([x[0] for x in res])
            # For recall, strictly we use worst_rank to be robust against ties.
            worst_ranks = np.array([x[2] for x in res])
            best_ranks = np.array([x[1] for x in res])
            
            med_sim = np.median(sims)
            z_score = np.sum(sims == 0.0)
            
            # Using worst rank for conservative recall
            med_worst_rank = np.median(worst_ranks)
            
            r1 = np.mean(worst_ranks <= 1) * 100
            r5 = np.mean(worst_ranks <= 5) * 100
            r10 = np.mean(worst_ranks <= 10) * 100
            r20 = np.mean(worst_ranks <= 20) * 100
            r100 = np.mean(worst_ranks <= 100) * 100
            
            print(f"{name:<22} | {med_sim:<7.3f} | {z_score:<7d} | {med_worst_rank:<9.0f} | {r1:<5.1f}% | {r5:<5.1f}% | {r10:<5.1f}% | {r20:<5.1f}% | {r100:<5.1f}%")

    print_report(results_all, "FULL TARGET")
    print_report(results_indic, "INDIC-SCRIPT GATED")

if __name__ == "__main__":
    main()
