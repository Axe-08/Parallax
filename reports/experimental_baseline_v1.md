# Parallax Experimental Baseline v1

## 1. Baseline Purpose
This document establishes the frozen, safe, and reproducible "Experimental Baseline v1" for the Parallax Entity Resolution system. It serves as the common starting point for three independent teammate tracks:
- **Model-Family Track**: Comparing LightGBM vs XGBoost vs CatBoost.
- **Retrieval Track**: Testing new candidate generation logic.
- **Semantic Track**: Developing semantic/embedding-based features.

By freezing the data, candidate pool, and exact feature matrix, we guarantee that future modeling experiments compare *only* the models themselves, without silent contamination from retrieval or feature drift.

## 2. Git Information
- **Branch**: `feature/v2-blocking-and-features`
- **Commit**: `94604f3debccb13602c6125569c16d48c3ce66ae` (plus baseline freezing modifications)
- **Tag**: `parallax-experimental-baseline-v1`

## 3. Dataset & Version
- **Dataset**: 5K Golden Split (`data/medium_split_200k/`)
- **Records**: Exact `train_source1.tsv` (Head 5,000 rows) against `train_source2.tsv` and `train_source3.tsv`.

## 4. Blocker Configuration
The `DualChannelTFIDFBlocker` is frozen with the following parameters:
- **Channel A (Name)**: `name_top_k=25`, `name_min_sim=0.15` (Native soft names only).
- **Channel B (Address)**: `addr_top_k=20`, `addr_min_sim=0.20` (Native clean addresses only).
- **Channel C (Building Number)**: Enabled with existing building-number / 2-char prefix logic.
- **Channel D (Translit)**: DISABLED (`translit_top_k=0`).

## 5. Candidate Counts
- **Total Candidate Pairs Generated**: [TBD]
- **Average Candidates per S1**: [TBD]
- **Cached Location**: `baseline_artifacts/candidate_pairs_sample.parquet`

## 6. Feature Manifest
Exactly 28 pairwise features are extracted and frozen.
**Cached Location**: `baseline_artifacts/features_sample.parquet`

**Lexical / Base:**
1. `raw_name_ratio`
2. `soft_name_ratio`
3. `token_sort_ratio`
4. `token_set_ratio`
5. `partial_ratio`
6. `addr_token_set_ratio`
7. `addr_ratio`
8. `num_match_score`
9. `is_s1_addr_null`
10. `is_cand_addr_null`
11. `both_addr_present`
12. `len_diff_name`
13. `len_ratio_name`

**Group 1: Building & Street Number:**
14. `primary_num_match`
15. `primary_num_conflict`
16. `primary_num_missing`
17. `num_jaccard`
18. `num_conflict_count`

**Group 2: Postal / PIN Code:**
19. `postal_match`
20. `postal_conflict`
21. `postal_missing`

**Group 3: Prefix & Token Overlap:**
22. `jaro_winkler_soft`
23. `jaro_winkler_raw`
24. `token_jaccard_name`
25. `token_overlap_name`
26. `first_token_match`

**Group 4: Canonical Address Similarity:**
27. `canon_addr_ratio`
28. `token_jaccard_addr`

## 7. Model Configuration
- **Matcher**: LightGBM (Baseline implementation)
- **Parameters**: Found via threshold sweep on Fold 0.
- **Winner Config**: [TBD]
- **Optimal Tau**: [TBD]

## 8. Evaluation Protocol
- **Validation Scheme**: 5-Fold Cross Validation.
- **Metrics**: Macro F0.5 (primary), Singleton Accuracy, Blocking FNs, Classification FNs, False Merges.
- **Gate**: SingletonGatedPredictor is applied identically for all evaluations.

## 9. Baseline Metrics (5-Fold CV on 5K Split)
- **Macro F0.5**: [TBD]
- **Precision**: [TBD]
- **Recall**: [TBD]
- **True Positives**: [TBD]
- **Classification FNs**: [TBD]
- **Blocking FNs**: [TBD]
- **False Merges**: [TBD]
- **Singleton Accuracy**: [TBD]
- **Singleton Violations**: [TBD]
- **Runtime**: [TBD]

## 10. Frozen Assumptions
- The candidate set generation is purely deterministic for the given data split.
- Features are exact copies (no jitter, no downstream transformations inside the model class).
- Train/Val fold assignments (`cv_folds_source1.tsv`) remain completely fixed.

## 11. Explicitly Rejected Components
The following hypotheses have been tested, rejected, and **must not be used**:
1. **Transliteration TF-IDF (Channel D)**: Yields only +2% recall at huge precision costs (100:1 false positive ratio).
2. **Address K=50 (Gap 1B)**: Adding 30 ranks to the address channel recovers ~36 true pairs but inflates candidates by 70%, actively harming LightGBM precision and downstream classification recall.

## 12. Rules for Future Experiments
- **Change ONE major dimension at a time.**
- If you are testing a new model (e.g. XGBoost), you MUST read from `baseline_artifacts/features_sample.parquet` and you MUST NOT re-run `extractor.py` or `DualChannelTFIDFBlocker`.
- If you are testing retrieval, you MUST use the frozen LightGBM parameters and evaluation code so you can directly compare your candidate metrics against this baseline document.
- Caches are isolated. Output your experiment artifacts to distinct directories (e.g., `output_xgboost`, `output_retrieval_v2`).
