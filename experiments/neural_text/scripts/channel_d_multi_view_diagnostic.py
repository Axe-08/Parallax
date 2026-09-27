import re
import time
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm

def has_indic_script(text):
    if not isinstance(text, str):
        return False
    return any(
        (0x0900 <= ord(c) <= 0x097F) or (0x0980 <= ord(c) <= 0x09FF) or 
        (0x0A00 <= ord(c) <= 0x0A7F) or (0x0A80 <= ord(c) <= 0x0AFF) or 
        (0x0B00 <= ord(c) <= 0x0B7F) or (0x0B80 <= ord(c) <= 0x0BFF) or 
        (0x0C00 <= ord(c) <= 0x0C7F) or (0x0C80 <= ord(c) <= 0x0CFF) or 
        (0x0D00 <= ord(c) <= 0x0D7F) for c in text
    )

def strip_vowels(text):
    if not isinstance(text, str): return ""
    return re.sub(r"[aeiouAEIOU]", "", text)

def consonant_skeleton(text):
    if not isinstance(text, str): return ""
    no_vowels = re.sub(r"[aeiouAEIOU\s]", "", text).lower()
    if not no_vowels: return ""
    return re.sub(r"(.)\1+", r"\1", no_vowels)

def get_rank_stats(scores, sim):
    strictly_greater = np.sum(scores > sim)
    equal = np.sum(scores == sim)
    best_rank = strictly_greater + 1
    worst_rank = strictly_greater + equal
    return best_rank, worst_rank, equal

def main():
    print("Loading data...")
    # Load authoritative 5K population
    feat_df = pd.read_parquet("baseline_artifacts/features_sample.parquet")
    eval_s1_ids = set(feat_df['s1_id'].astype(str).unique())
    cands_df = pd.read_parquet("baseline_artifacts/candidate_pairs_sample.parquet")
    e0_cand_pairs = set(zip(cands_df['s1_id'].astype(str), cands_df['cand_id'].astype(str)))

    s1_df = pd.read_csv("data/medium_split_200k/train_source1.tsv", sep="\t")
    s1_df['entity_id'] = s1_df['entity_id'].astype(str)
    s1_df = s1_df[s1_df['entity_id'].isin(eval_s1_ids)].reset_index(drop=True)

    s2_df = pd.read_csv("data/medium_split_200k/train_source2.tsv", sep="\t")
    s3_df = pd.read_csv("data/medium_split_200k/train_source3.tsv", sep="\t")
    tgt_df = pd.concat([s2_df, s3_df], ignore_index=True)
    tgt_df['entity_id'] = tgt_df['entity_id'].astype(str)
    
    tgt_india = tgt_df[tgt_df["country"] == "India"].reset_index(drop=True)
    tgt_info = tgt_india.set_index("entity_id")

    print("Loading transliteration cache...")
    cache_df = pd.read_parquet("experiments/neural_text/caches/indicxlit_translit_cache_5k.parquet")
    cache_df['entity_id'] = cache_df['entity_id'].astype(str)
    translit_map = cache_df.set_index("entity_id")["indicxlit_name"].to_dict()

    def get_translit(row):
        eid = str(row["entity_id"])
        if eid in translit_map and pd.notna(translit_map[eid]) and str(translit_map[eid]).strip():
            return str(translit_map[eid]).strip()
        name = str(row.get("soft_name", ""))
        if not name.strip():
            name = str(row.get("business_name", ""))
        return name.strip()

    s1_df["translit_name"] = s1_df.apply(get_translit, axis=1)
    tgt_india["translit_name"] = tgt_india.apply(get_translit, axis=1)

    tgt_is_indic = tgt_india.apply(lambda r: has_indic_script(str(r.get("business_name", ""))), axis=1).values

    print("Identifying Canonical BFNs for the 5K Evaluation Population...")
    gt = pd.read_csv("data/medium_split_200k/train_ground_truth.tsv", sep="\t")
    gt['source1_entity_id'] = gt['source1_entity_id'].astype(str)
    
    gt_true_set = set()
    for _, row in gt.iterrows():
        s1 = str(row["source1_entity_id"])
        if s1 in eval_s1_ids:
            matches_str = str(row["matched_entity_ids"])
            if pd.notna(row["matched_entity_ids"]) and matches_str.strip():
                for m in matches_str.split(","):
                    m = m.strip()
                    if m:
                        gt_true_set.add((s1, m))

    canonical_bfns = []
    for pair in gt_true_set:
        if pair not in e0_cand_pairs:
            s1, cand = pair
            if cand in tgt_info.index:
                canonical_bfns.append(pair)
                
    bfn_s1_ids = set([s for s, c in canonical_bfns])
    print(f"Isolated {len(canonical_bfns)} canonical BFNs out of {len(bfn_s1_ids)} S1s.")

    s1_bfn_df = s1_df[s1_df["entity_id"].astype(str).isin(bfn_s1_ids)].reset_index(drop=True)
    s1_names = s1_bfn_df["translit_name"].fillna("").str.lower().tolist()
    s1_ids = s1_bfn_df["entity_id"].astype(str).tolist()

    tgt_names = tgt_india["translit_name"].fillna("").str.lower().tolist()
    tgt_ids = tgt_india["entity_id"].astype(str).tolist()

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
        ("Token_Jaccard", CountVectorizer(analyzer="word", binary=True), s1_names, tgt_names),
        ("Vowel_Stripped_3gram", TfidfVectorizer(analyzer="char", ngram_range=(3, 3), sublinear_tf=True), s1_vowel, tgt_vowel),
        ("Consonant_Skel_3gram", TfidfVectorizer(analyzer="char", ngram_range=(3, 3), sublinear_tf=True), s1_cons, tgt_cons),
    ]

    print("\nEvaluating Representations (Streaming Memory-Efficient)...")

    # To calculate candidate volumes across the 396 BFNs we track exact sets per K.
    # We will compute results for FULL TARGET and INDIC TARGET.
    # results format: universe -> rep -> { 'rank_stats': [], 'cand_sets': { K: set() } }
    
    universes = {
        "FULL TARGET": np.ones(len(tgt_names), dtype=bool),
        "INDIC-SCRIPT GATED": tgt_is_indic
    }
    
    rep_names = [r[0] for r in reps] + ["Jaro_Winkler", "Token_Sort_Ratio", "Ensemble_Max"]
    
    master_results = {}
    for u_name in universes:
        master_results[u_name] = {}
        for r_name in rep_names:
            master_results[u_name][r_name] = {
                'rank_stats': [], 
                'cands_by_k': {k: set() for k in [1, 5, 10, 20, 50, 100]}
            }

    vec_models = {}
    for name, vec, s1_src, tgt_src in reps:
        tgt_mat = vec.fit_transform(tgt_src)
        s1_mat = vec.transform(s1_src)
        if name == "Token_Jaccard":
            tgt_sum = np.array(tgt_mat.sum(axis=1)).flatten()
            vec_models[name] = (s1_mat, tgt_mat.T, tgt_sum)
        else:
            vec_models[name] = (s1_mat, tgt_mat.T, None)

    for i, s1_id in enumerate(tqdm(s1_ids, desc="Querying S1 BFNs")):
        scores_dict = {}
        for name, (s1_mat, tgt_mat_T, tgt_sum) in vec_models.items():
            intersection = s1_mat[i].dot(tgt_mat_T).toarray()[0]
            if name == "Token_Jaccard":
                s1_sum = s1_mat[i].sum()
                union = s1_sum + tgt_sum - intersection
                union[union == 0] = 1
                scores_dict[name] = intersection / union
            else:
                scores_dict[name] = intersection

        jw_scores = process.cdist([s1_names[i]], tgt_names, scorer=JaroWinkler.normalized_similarity)[0]
        ts_scores = process.cdist([s1_names[i]], tgt_names, scorer=fuzz.token_sort_ratio)[0] / 100.0

        scores_dict["Jaro_Winkler"] = jw_scores
        scores_dict["Token_Sort_Ratio"] = ts_scores
        scores_dict["Ensemble_Max"] = np.maximum.reduce([
            scores_dict["TFIDF_Char_3gram"],
            scores_dict["Vowel_Stripped_3gram"],
            scores_dict["Jaro_Winkler"],
            scores_dict["Token_Sort_Ratio"],
        ])
        
        target_cands_for_s1 = [cand for (s, cand) in canonical_bfns if s == s1_id]

        for u_name, u_mask in universes.items():
            for rep_name, scores in scores_dict.items():
                u_scores = scores.copy()
                u_scores[~u_mask] = -1.0 # Mask out non-universe items
                
                # Record rank stats for BFN recovery
                for cand_id in target_cands_for_s1:
                    tgt_i = tgt_idx_map[cand_id]
                    if not u_mask[tgt_i]:
                        master_results[u_name][rep_name]['rank_stats'].append((0.0, 999999, 999999, 0))
                        continue
                        
                    sim = u_scores[tgt_i]
                    best_r, worst_r, eq_r = get_rank_stats(u_scores, sim)
                    master_results[u_name][rep_name]['rank_stats'].append((sim, best_r, worst_r, eq_r))
                
                # Sort for Top-K candidate extraction (only nonzero scores)
                # We use argpartition for speed, then sort the top part
                valid_idx = np.where(u_scores > 0)[0]
                if len(valid_idx) == 0:
                    continue
                    
                valid_scores = u_scores[valid_idx]
                sort_idx = np.argsort(valid_scores)[::-1]
                top_idx_sorted = valid_idx[sort_idx]
                
                for k in [1, 5, 10, 20, 50, 100]:
                    k_idx = top_idx_sorted[:k]
                    for idx in k_idx:
                        master_results[u_name][rep_name]['cands_by_k'][k].add((s1_id, tgt_ids[idx]))

    # Analyze composition of BFNs
    def get_composition(bfn_pairs):
        cross = 0
        domain_handle = 0
        other = 0
        for (s, c) in bfn_pairs:
            s_indic = has_indic_script(str(s1_df[s1_df['entity_id'] == s]['business_name'].values[0]))
            c_indic = has_indic_script(str(tgt_info.loc[c]['business_name']))
            s_row = s1_df[s1_df['entity_id'] == s]
            s_dom = False
            if 'email' in s_row.columns and len(s_row) > 0:
                s_dom = s_dom or ("@" in str(s_row['email'].values[0]))
            if 'website' in s_row.columns and len(s_row) > 0:
                s_dom = s_dom or ("." in str(s_row['website'].values[0]))
            
            if s_indic != c_indic:
                cross += 1
            elif s_dom:
                domain_handle += 1
            else:
                other += 1
        return cross, domain_handle, other

    def print_report(u_name):
        print(f"\n=========================================================================================================")
        print(f" {u_name} UNIVERSE DIAGNOSTIC (For {len(canonical_bfns)} Canonical BFNs in {len(bfn_s1_ids)} S1s)")
        print(f"=========================================================================================================")
        
        # 1. R@K Summary
        header = f"{'Representation':<22} | {'Med Sim':<7} | {'Zero-Count':<10} | {'Med Best':<8} | {'Med Worst':<9} | {'R@1 (B/W)':<11} | {'R@5 (B/W)':<11} | {'R@10 (B/W)':<12} | {'R@20 (B/W)':<12} | {'R@100 (B/W)':<12}"
        print(header)
        print("-" * len(header))
        
        for name in rep_names:
            res = master_results[u_name][name]['rank_stats']
            sims = np.array([x[0] for x in res])
            best_ranks = np.array([x[1] for x in res])
            worst_ranks = np.array([x[2] for x in res])

            med_sim = np.median(sims)
            zero_count = np.sum(sims == 0.0)
            med_best = np.median(best_ranks)
            med_worst = np.median(worst_ranks)

            r1_b = np.mean(best_ranks <= 1) * 100; r1_w = np.mean(worst_ranks <= 1) * 100
            r5_b = np.mean(best_ranks <= 5) * 100; r5_w = np.mean(worst_ranks <= 5) * 100
            r10_b = np.mean(best_ranks <= 10) * 100; r10_w = np.mean(worst_ranks <= 10) * 100
            r20_b = np.mean(best_ranks <= 20) * 100; r20_w = np.mean(worst_ranks <= 20) * 100
            r100_b = np.mean(best_ranks <= 100) * 100; r100_w = np.mean(worst_ranks <= 100) * 100

            print(f"{name:<22} | {med_sim:<7.3f} | {zero_count:<10d} | {med_best:<8.0f} | {med_worst:<9.0f} | {r1_b:>4.1f}/{r1_w:<4.1f} | {r5_b:>4.1f}/{r5_w:<4.1f} | {r10_b:>4.1f}/{r10_w:<4.1f} | {r20_b:>4.1f}/{r20_w:<4.1f} | {r100_b:>4.1f}/{r100_w:<4.1f}")
            
        # 2. Candidate Volume & Tradeoff (Only analyzing the strongest overall representation for simplicity or all?)
        # Let's print Candidate Volume for ALL representations, but maybe just for K=10 and K=50 to avoid spam, or print full table for the best.
        # Actually, let's print full table for Ensemble_Max and TFIDF_Char_3gram
        print("\n--- CANDIDATE VOLUME TRADEOFF FOR TOP REPRESENTATIONS ---")
        
        # Calculate E0 baseline volume restricted to these BFN S1s
        e0_cands_for_bfn_s1s = set([(s, c) for (s, c) in e0_cand_pairs if s in bfn_s1_ids])
        e0_volume = len(e0_cands_for_bfn_s1s)
        
        for name in ["TFIDF_Char_3gram", "Ensemble_Max"]:
            print(f"\n>> {name} <<")
            v_header = f"{'K':<4} | {'Recov BFNs':<10} | {'Recov %':<7} | {'New Cands Added':<16} | {'New/S1':<6} | {'Total Union Vol':<15} | {'Mult vs E0':<10} | {'Composition (Cross/Dom/Oth)':<25}"
            print(v_header)
            print("-" * len(v_header))
            
            for k in [1, 5, 10, 20, 50, 100]:
                cands_at_k = master_results[u_name][name]['cands_by_k'][k]
                
                recovered_bfns = cands_at_k.intersection(set(canonical_bfns))
                recov_pct = (len(recovered_bfns) / len(canonical_bfns)) * 100 if canonical_bfns else 0
                
                new_cands = cands_at_k - e0_cands_for_bfn_s1s
                new_vol = len(new_cands)
                new_per_s1 = new_vol / len(bfn_s1_ids) if bfn_s1_ids else 0
                
                total_union = len(e0_cands_for_bfn_s1s.union(cands_at_k))
                mult = total_union / e0_volume if e0_volume else 0
                
                cross, dom, oth = get_composition(recovered_bfns)
                comp_str = f"{cross}/{dom}/{oth}"
                
                print(f"{k:<4} | {len(recovered_bfns):<10d} | {recov_pct:>6.1f}% | {new_vol:<16d} | {new_per_s1:>6.1f} | {total_union:<15d} | {mult:>9.2f}x | {comp_str:<25}")


    print_report("FULL TARGET")
    print_report("INDIC-SCRIPT GATED")

if __name__ == "__main__":
    main()
