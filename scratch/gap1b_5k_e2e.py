"""
Phase 2: Gap 1B 5K Controlled Experiment (End-to-End)
======================================================
BASELINE: name_top_k=25, addr_top_k=20, C
EXPERIMENT: name_top_k=25, addr_top_k=50, C
(Channel D transliteration is OFF)
"""
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from parallax.data.contracts import load_business_records_df, load_ground_truth_dict
from parallax.preprocessing.normalizer import widen_records_df
from parallax.blocking.tfidf_blocker import DualChannelTFIDFBlocker
from parallax.features.extractor import FEATURE_COLUMNS, PairwiseFeatureExtractor
from parallax.postprocessing.singleton_gate import SingletonGatedPredictor
from parallax.metrics.evaluator import evaluate_resolution_predictions

N_S1 = 5000
DATA_DIR = Path("data/medium_split_200k")

def is_non_latin(text: str) -> bool:
    return any(ord(c) > 0x024F for c in str(text) if not unicodedata.category(c).startswith("Z"))

def _get_native_addrs(df: pd.DataFrame) -> list[str]:
    if "clean_address" in df:
        p_s = df["clean_address"].fillna("").astype(str).str.strip()
    elif "business_address" in df:
        p_s = df["business_address"].fillna("").astype(str).str.strip()
    else:
        p_s = pd.Series([""] * len(df), index=df.index)
    return [str(x) for x in p_s.tolist()]

def count_failures(predictions, val_gt, candidates):
    fp_cnt = 0
    sing_cnt = 0
    bfn_cnt = 0
    cfn_cnt = 0
    for s1_id, true_set in val_gt.items():
        cand_set = candidates.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())
        if len(true_set) == 0:
            sing_cnt += len(pred_set)
        else:
            fp_cnt += len(pred_set - true_set)
            fn_set = true_set - pred_set
            for fn in fn_set:
                if fn in cand_set:
                    cfn_cnt += 1
                else:
                    bfn_cnt += 1
    return {
        "FALSE_MERGE_POSITIVE": fp_cnt,
        "SINGLETON_VIOLATION": sing_cnt,
        "CLASSIFICATION_FALSE_NEGATIVE": cfn_cnt,
        "BLOCKING_FALSE_NEGATIVE": bfn_cnt,
    }

def main():
    print("=" * 80)
    print("GAP 1B — 5K CONTROLLED EXPERIMENT (ADDR K=50)")
    print("=" * 80)

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

    # Compute cross-script ground truth
    cs_true_pairs = set()
    for s1_id, true_set in gt_dict.items():
        s1n = str(s1_lookup.get(s1_id, {}).get("business_name", ""))
        for tgt_id in true_set:
            tgtn = str(tgt_lookup.get(tgt_id, {}).get("business_name", ""))
            if is_non_latin(s1n) != is_non_latin(tgtn):
                cs_true_pairs.add((s1_id, tgt_id))
    print(f"Total True Pairs: {sum(len(v) for v in gt_dict.values())}")
    print(f"Total CS True Pairs: {len(cs_true_pairs)}")

    s1_no_translit = s1_wide.drop(columns=["translit_name"], errors="ignore")
    tgt_no_translit = target_wide.drop(columns=["translit_name"], errors="ignore")

    print("\n[BLOCKING] Baseline (Name K=25, Addr K=20)...")
    blocker_base = DualChannelTFIDFBlocker(
        name_top_k=25, addr_top_k=20, translit_top_k=0,
        name_min_sim=0.15, addr_min_sim=0.20, translit_min_sim=1.0,
        batch_size=2000, show_progress=False
    )
    cands_base = blocker_base.generate_candidates(s1_no_translit, tgt_no_translit)

    print("[BLOCKING] Experiment Gap 1B (Name K=25, Addr K=50)...")
    blocker_exp = DualChannelTFIDFBlocker(
        name_top_k=25, addr_top_k=50, translit_top_k=0,
        name_min_sim=0.15, addr_min_sim=0.20, translit_min_sim=1.0,
        batch_size=2000, show_progress=False
    )
    cands_exp = blocker_exp.generate_candidates(s1_no_translit, tgt_no_translit)

    def blocking_metrics(candidates):
        total_pairs = sum(len(c) for c in candidates.values())
        avg_cands = total_pairs / N_S1 if N_S1 else 0
        total_true = sum(len(v) for v in gt_dict.values())
        recovered_true = 0
        cs_recovered = 0
        fns = set()
        cs_fns = set()
        
        for s1_id, true_set in gt_dict.items():
            cand_set = candidates.get(s1_id, set())
            recovered_true += len(true_set & cand_set)
            missed = true_set - cand_set
            for tgt_id in missed:
                fns.add((s1_id, tgt_id))
                if (s1_id, tgt_id) in cs_true_pairs:
                    cs_fns.add((s1_id, tgt_id))
            for tgt_id in (true_set & cand_set):
                if (s1_id, tgt_id) in cs_true_pairs:
                    cs_recovered += 1

        return {
            "total_pairs": total_pairs,
            "avg_cands": avg_cands,
            "recall": recovered_true / total_true if total_true else 0.0,
            "cs_recall": cs_recovered / len(cs_true_pairs) if cs_true_pairs else 0.0,
            "blocking_fns": len(fns),
            "cs_fns": len(cs_fns),
            "fn_set": fns,
            "cs_fn_set": cs_fns
        }

    met_base = blocking_metrics(cands_base)
    met_exp = blocking_metrics(cands_exp)

    inflation = (met_exp["total_pairs"] - met_base["total_pairs"]) / met_base["total_pairs"] * 100
    
    print("\n--- BLOCKING RESULTS ---")
    print(f"Baseline pairs: {met_base['total_pairs']:,} ({met_base['avg_cands']:.2f} per S1)")
    print(f"Exp pairs:      {met_exp['total_pairs']:,} ({met_exp['avg_cands']:.2f} per S1)")
    print(f"Inflation:      {inflation:+.2f}%")
    print(f"Recall:         {met_base['recall']*100:.2f}% -> {met_exp['recall']*100:.2f}%")
    print(f"CS Recall:      {met_base['cs_recall']*100:.2f}% -> {met_exp['cs_recall']*100:.2f}%")
    print(f"Total FNs:      {met_base['blocking_fns']} -> {met_exp['blocking_fns']}")
    print(f"CS FNs:         {met_base['cs_fns']} -> {met_exp['cs_fns']}")

    newly_recovered = met_base["cs_fn_set"] - met_exp["cs_fn_set"]
    print(f"\n[CHANNEL ATTRIBUTION] Newly recovered CS true pairs: {len(newly_recovered)}")
    
    # Check address similarity for newly recovered
    if len(newly_recovered) > 0:
        for s1_id, tgt_id in newly_recovered:
            s1_row = s1_lookup.get(s1_id, {})
            tgt_row = tgt_lookup.get(tgt_id, {})
            s1_addr = s1_row.get("clean_address", "") or s1_row.get("business_address", "")
            tgt_addr = tgt_row.get("clean_address", "") or tgt_row.get("business_address", "")
            s1_nums = set(str(n) for n in (s1_row.get("numbers") or []))
            tgt_nums = set(str(n) for n in (tgt_row.get("numbers") or []))
            num_match = bool(s1_nums & tgt_nums)
            
            try:
                v = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), sublinear_tf=True)
                m = v.fit_transform([str(s1_addr), str(tgt_addr)])
                sim = float(cosine_similarity(m[0:1], m[1:2])[0, 0])
            except:
                sim = 0.0
            print(f"  - {s1_id} <-> {tgt_id}: AddrSim={sim:.3f} | NumMatch={num_match}")

    print("\n[END-TO-END] Extracting 28 Features for Baseline...")
    extractor = PairwiseFeatureExtractor()
    feats_28 = FEATURE_COLUMNS[:28]
    pairs_base = extractor.extract_features_df(cands_base, s1_no_translit, tgt_no_translit, ground_truth=gt_dict, show_progress=False)

    print("[END-TO-END] Extracting 28 Features for Experiment...")
    pairs_exp = extractor.extract_features_df(cands_exp, s1_no_translit, tgt_no_translit, ground_truth=gt_dict, show_progress=False)

    # 80/20 train/val split
    unique_s1 = list(s1_df["entity_id"].unique())
    np.random.seed(42)
    np.random.shuffle(unique_s1)
    split_idx = int(0.8 * len(unique_s1))
    train_s1 = set(unique_s1[:split_idx])
    val_s1 = set(unique_s1[split_idx:])
    val_gt = {k: v for k, v in gt_dict.items() if k in val_s1}

    def train_and_eval(pairs_df, candidates):
        train_pairs = pairs_df[pairs_df["s1_id"].isin(train_s1)].copy()
        val_pairs = pairs_df[pairs_df["s1_id"].isin(val_s1)].copy()
        
        x_train = train_pairs[feats_28]
        y_train = train_pairs["target"].astype(int)
        train_data = lgb.Dataset(x_train, label=y_train)

        x_val = val_pairs[feats_28]
        y_val = val_pairs["target"].astype(int)
        val_data = lgb.Dataset(x_val, label=y_val, reference=train_data)

        params = {
            "objective": "binary", "metric": "binary_logloss",
            "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
            "verbose": -1, "seed": 42
        }
        model = lgb.train(params, train_data, num_boost_round=150, valid_sets=[train_data, val_data])

        probs = model.predict(x_val)
        val_eval_df = val_pairs[["s1_id", "cand_id"]].copy()
        val_eval_df["prob"] = probs

        predictor = SingletonGatedPredictor(decision_threshold=0.75)
        best_tau, best_score, best_report, best_preds = 0.75, -1.0, None, None
        
        for tau in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
            preds = predictor.filter_predictions(val_eval_df, list(val_s1), threshold=tau)
            rep = evaluate_resolution_predictions(val_gt, preds)
            if rep.macro_f05 > best_score:
                best_score = rep.macro_f05
                best_tau = tau
                best_report = rep
                best_preds = preds
                
        fails = count_failures(best_preds, val_gt, candidates)
        prec = best_report.total_correct_pairs / best_report.total_predicted_pairs if best_report.total_predicted_pairs else 0
        rec = best_report.total_correct_pairs / best_report.total_true_pairs if best_report.total_true_pairs else 0
        return best_report, fails, prec, rec

    print("\nTraining & Evaluating Baseline...")
    rep_b, fail_b, prec_b, rec_b = train_and_eval(pairs_base, cands_base)
    
    print("Training & Evaluating Experiment...")
    rep_e, fail_e, prec_e, rec_e = train_and_eval(pairs_exp, cands_exp)

    print("\n" + "="*80)
    print("END-TO-END RESULTS (VALIDATION SET)")
    print("="*80)
    print(f"{'Metric':<30} | {'Baseline (K=20)':<15} | {'Exp (K=50)':<15} | {'Delta':<10}")
    print("-" * 78)
    metrics = [
        ("Macro F0.5", rep_b.macro_f05, rep_e.macro_f05, f"{rep_e.macro_f05 - rep_b.macro_f05:+.4f}"),
        ("Precision", prec_b, prec_e, f"{prec_e - prec_b:+.4f}"),
        ("Recall", rec_b, rec_e, f"{rec_e - rec_b:+.4f}"),
        ("Singleton Accuracy", rep_b.singleton_score, rep_e.singleton_score, f"{rep_e.singleton_score - rep_b.singleton_score:+.4f}"),
        ("Predicted Pairs", rep_b.total_predicted_pairs, rep_e.total_predicted_pairs, f"{rep_e.total_predicted_pairs - rep_b.total_predicted_pairs:+d}"),
        ("True Positives", rep_b.total_correct_pairs, rep_e.total_correct_pairs, f"{rep_e.total_correct_pairs - rep_b.total_correct_pairs:+d}"),
        ("Class. False Negatives", fail_b['CLASSIFICATION_FALSE_NEGATIVE'], fail_e['CLASSIFICATION_FALSE_NEGATIVE'], f"{fail_e['CLASSIFICATION_FALSE_NEGATIVE'] - fail_b['CLASSIFICATION_FALSE_NEGATIVE']:+d}"),
        ("Blocking False Negatives", fail_b['BLOCKING_FALSE_NEGATIVE'], fail_e['BLOCKING_FALSE_NEGATIVE'], f"{fail_e['BLOCKING_FALSE_NEGATIVE'] - fail_b['BLOCKING_FALSE_NEGATIVE']:+d}"),
        ("False Merges", fail_b['FALSE_MERGE_POSITIVE'], fail_e['FALSE_MERGE_POSITIVE'], f"{fail_e['FALSE_MERGE_POSITIVE'] - fail_b['FALSE_MERGE_POSITIVE']:+d}"),
        ("Singleton Violations", fail_b['SINGLETON_VIOLATION'], fail_e['SINGLETON_VIOLATION'], f"{fail_e['SINGLETON_VIOLATION'] - fail_b['SINGLETON_VIOLATION']:+d}")
    ]
    
    for m, b_v, e_v, d in metrics:
        if isinstance(b_v, float):
            print(f"{m:<30} | {b_v:<15.4f} | {e_v:<15.4f} | {d:<10}")
        else:
            print(f"{m:<30} | {str(b_v):<15} | {str(e_v):<15} | {d:<10}")

if __name__ == "__main__":
    main()
