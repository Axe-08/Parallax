# Parallax: Entity Resolution System Architecture Brief (v1.0)

This document provides a comprehensive technical overview of the current entity resolution architecture implemented in the Parallax repository. It is intended for teammates exploring alternative approaches, components, or complementary model pipelines.

---

## 1. Problem Framing & Core Invariants

The objective is multi-source business entity resolution across three disparate tables:
- **Source 1 ($S_1$):** Query entity table containing business records (`entity_id`, `business_name`, `business_address`, `country`).
- **Source 2 ($S_2$) & Source 3 ($S_3$):** Target candidate tables containing potential matching business entities.
- **Ground Truth:** Maps each $S_1$ entity ID to a comma-separated list of matching entity IDs in $S_2$ and $S_3$. Singletons (entities with 0 matches across $S_2$ and $S_3$) map to an empty string.

### The Competition Evaluation Metric
The evaluation metric is **Macro $F_{0.5}$** across all $N$ Source 1 entities:

$$F_{0.5}(s_1) = \begin{cases} 
1.0 & \text{if } |\text{true}| = 0 \text{ and } |\text{pred}| = 0 \quad (\text{Correct Singleton}) \\
0.0 & \text{if } |\text{true}| = 0 \text{ and } |\text{pred}| > 0 \quad (\text{Singleton Violation}) \\
0.0 & \text{if } |\text{true}| > 0 \text{ and } |\text{pred}| = 0 \quad (\text{Total False Negative}) \\
\frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}} & \text{otherwise}
\end{cases}$$

$$\text{Macro } F_{0.5} = \frac{1}{N} \sum_{i=1}^N F_{0.5}(s_1^{(i)})$$

**Key Metric Behavior:**
1. **Precision is weighted $2\times$ as heavily as Recall** ($\beta = 0.5$). False merge positives severely penalize the entity score.
2. **True Singletons are all-or-nothing:** Correctly identifying a singleton awards $1.0$. A single spurious distractor drops the score instantly to $0.0$.

---

## 2. Dataset Scale & The Medium Benchmark Split (200k)

To iterate without waiting for full 2.1M raw records, a representative **200,000 $S_1$ record medium benchmark split** was generated from raw S3 train files and cached locally at `data/medium_split_200k/` (also on S3 at `s3://amazon-ml-challange-2026-parallax/splits/medium_split_200k/`):

| File | Row Count | Characteristics / Details |
| :--- | :--- | :--- |
| `train_source1.tsv` | **200,000** | Stratified: US (119,958; 60%) and India (80,042; 40%) |
| `train_source2.tsv` | **500,000** | 334,213 true positive matches + 165,787 negative distractors |
| `train_source3.tsv` | **500,000** | 357,980 true positive matches + 142,020 negative distractors |
| `train_ground_truth.tsv` | **200,000** | 11,170 true singletons (5.58%) + 188,830 non-singletons |
| `cv_folds_source1.tsv` | **200,000** | 5 balanced folds (40k entities/fold), stratified by `(country, is_singleton)` |

The total candidate pool across $S_2 + S_3$ is **1,000,000 entities** (692k true matches + 308k negative distractors).

---

## 3. End-to-End Pipeline Architecture

The current pipeline consists of six sequential stages:

```
[Raw TSVs] 
     │
     ▼
[Stage 1: Preprocessing & Unicode Widening]
     │ (NFKC, Indic-Latin transliteration, number extraction)
     ▼
[Stage 2: Country-Partitioned Dual-Channel TF-IDF Blocker]
     │ (Sparse Character 3-Grams on Names & Addresses + Building Numbers)
     ▼
[Candidate Pairs (~5.5M pairs, ~28 cands / query)]
     │
     ▼
[Stage 3: RapidFuzz Pairwise Feature Extraction]
     │ (13 lexical, structural, and alignment features)
     ▼
[Feature Matrix (Snappy Parquet)]
     │
     ▼
[Stage 4: LightGBM Asymmetric Classifier & Threshold Calibrator]
     │ (Histogram gradient boosting + tau optimization for Macro F0.5)
     ▼
[Stage 5: Singleton Gating & Post-Processing]
     │ (Confidence gating: max prob < tau -> singleton empty set)
     ▼
[Stage 6: 5-Fold Cross-Validation & Failure Diagnostics]
     │ (OOF metrics, JSONL error logs, executive markdown scorecard)
     ▼
[matching_results.tsv] & [diagnostics_medium_200k.md]
```

---

## 4. In-Depth Component Specifications

### Stage 1: Data Contracts, Normalization & Entity Widening
- **Module:** `src/parallax/preprocessing/normalizer.py` & `src/parallax/preprocessing/transliteration.py`
- **Transforms:**
  1. **Unicode NFKC Canonical Normalization:** Strips unprintable control characters, normalizes Unicode diacritics, and strips excess whitespace.
  2. **Script Detection & Latin Transliteration:** Uses deterministic phonetic mappings for Indic scripts (Hindi/Devanagari, Tamil) into phonetic Latin character representations. This allows cross-lingual names (e.g., Hindi script vs English spelling) to match in the same vector space.
  3. **Building & Unit Number Extraction:** Regex extraction of numerical address components (`\b\d+[-/]?\w*\b`) into clean sets (e.g. `{"12", "4B"}`).
  4. **Null Address Indicators:** Identifies missing, empty, or degenerate addresses (`is_addr_null = 1`).
  5. **Soft Representation:** Generates `soft_name` (lowercased, alphanumeric tokens only) alongside `translit_name`.

### Stage 2: Dual-Channel Sparse TF-IDF Blocker
- **Module:** `src/parallax/blocking/tfidf_blocker.py`
- **Strategy:**
  1. **Country Partitioning:** Blocks US entities strictly against US candidates, and India entities strictly against India candidates.
  2. **Channel A (Name Channel):** Character 3-Gram TF-IDF vectorizer over `soft_name + translit_name`. Generates sparse CSR matrix dot-products (`s1_batch.dot(tgt_matrix)`). Prunes pairs with similarity below `name_min_sim = 0.15` and retains top $K = 35$ candidates per $S_1$.
  3. **Channel B (Address Channel):** Character 3-Gram TF-IDF vectorizer over cleaned addresses. Prunes pairs below `addr_min_sim = 0.20` and retains top $K = 25$ candidates per $S_1$.
  4. **Channel C (Building Number Channel):** Inverted index over building numbers ($\ge 2$ digits) intersected with 2-character soft name prefixes, bounded to numbers with $\le 50$ entities to prevent distractor explosion.
- **Output:** Set union of Channel A, B, and C candidates.
- **Performance Characteristics:**
  - Blocking Recall (Pair Completeness): **~95.8%**
  - Reduction Ratio: **99.994%** (reduces 200B potential pairs to ~5.5M candidate pairs, ~28 candidates per query).

### Stage 3: Pairwise Feature Engineering Layer
- **Module:** `src/parallax/features/extractor.py`
- **Feature Set (13 Dense Features):**
  1. `raw_name_ratio`: Full Levenshtein similarity on raw string.
  2. `soft_name_ratio`: Levenshtein ratio on lowercased alphanumeric string.
  3. `token_sort_ratio`: Levenshtein ratio after sorting tokens (handles inverted word order, e.g. "Acme Hardware" vs "Hardware Acme").
  4. `token_set_ratio`: Set intersection & difference ratio (handles added legal suffixes or department names).
  5. `partial_ratio`: Highest matching substring ratio (handles truncated titles).
  6. `addr_token_set_ratio`: Token set similarity on address strings.
  7. `addr_ratio`: Levenshtein ratio on cleaned addresses.
  8. `num_match_score`: Jaccard similarity of extracted building/unit numbers.
  9. `is_s1_addr_null`: Binary flag (Source 1 address missing).
  10. `is_cand_addr_null`: Binary flag (Candidate address missing).
  11. `both_addr_present`: Binary flag (both addresses present and comparable).
  12. `len_diff_name`: Absolute length difference in characters.
  13. `len_ratio_name`: Length ratio ($\min / \max$).
- **Implementation:** Vectorized record lookup tuples + RapidFuzz C++ bindings; outputs memory-mapped/Parquet files.

### Stage 4: LightGBM Asymmetric Classification & Threshold Calibration
- **Module:** `src/parallax/models/matcher.py`
- **Model Structure:** LightGBM Binary Classifier (`objective="binary"`, `metric="binary_logloss"`).
- **Default Hyperparameters:**
  - `learning_rate`: 0.05
  - `num_leaves`: 31 (deep variant: 63)
  - `max_depth`: 6 (deep variant: 8)
  - `n_estimators`: 150
- **Threshold Calibration:**
  - Predictions are continuous probabilities $P(s_1, c) \in [0, 1]$.
  - The decision threshold $\tau$ is not fixed at 0.50. It is systematically optimized over a grid $\tau \in [0.60, 0.95]$ directly maximizing Macro $F_{0.5}$ on out-of-fold validation data.
  - Typical optimal $\tau$ ranges between **0.78 and 0.88** to enforce the high precision demanded by $F_{0.5}$.

### Stage 5: Singleton Gating & Post-Processing Engine
- **Module:** `src/parallax/postprocessing/singleton_gate.py`
- **Rule:** For each query entity $s_1$:
  - If $\max_{c} P(s_1, c) < \tau$, predicted match set is $\emptyset$ (entity classified as singleton).
  - If one or more candidates have $P(s_1, c) \ge \tau$, output matches are joined with commas: `"cand_1,cand_2"`.
- This ensures that uncertain predictions are suppressed to preserve the perfect 1.0 singleton accuracy.

### Stage 6: Failure Diagnostics & Error Triaging System
- **Module:** `src/parallax/diagnostics/failure_logger.py`
- Evaluates full Out-Of-Fold predictions against ground truth and logs every single discrepancy into structured JSONL (`reports/failures_medium_200k.jsonl`) and an executive markdown report (`reports/diagnostics_medium_200k.md`).
- **The 4 Categorized Error Modes:**
  1. `FALSE_MERGE_POSITIVE`: A distractor candidate was predicted above threshold ($P \ge \tau$). Penalized 2x under Macro $F_{0.5}$.
  2. `SINGLETON_VIOLATION`: A true singleton (0 matches in reality) was assigned one or more candidates, dropping entity score from 1.0 to 0.0.
  3. `BLOCKING_FALSE_NEGATIVE`: True match was never retrieved by the blocker (defines the recall ceiling).
  4. `CLASSIFICATION_FALSE_NEGATIVE`: True match was retrieved by the blocker, but the LightGBM probability fell below $\tau$.

---

## 5. Potential Exploration Avenues for Teammates

For teammates looking to explore alternative or complementary directions:

1. **Alternative Blocking Mechanisms:**
   - Dense bi-encoder embeddings (e.g. `bge-m3`, `MiniLM`) + FAISS / HNSW indexing to capture semantic synonyms that Character 3-Grams miss.
   - Learned blocking / DeepMatcher / contrastive sparse representations.
2. **Graph & Relational Post-Processing:**
   - The current pipeline treats pairs independently. Applying graph clustering (connected components, Louvain, or transitive closure validation) across $S_1 \leftrightarrow S_2 \leftrightarrow S_3$ could resolve conflicting multi-match predictions.
3. **Advanced Classifiers & Ensembles:**
   - CatBoost or XGBoost pairwise classifiers to ensemble with LightGBM.
   - Cross-encoder rerankers applied only to top-5 candidates per query.
4. **Enhanced Address Parsing:**
   - Using `libpostal` or token-based address parsing to separate Street, City, State, and Postal Code into distinct comparative channels.
