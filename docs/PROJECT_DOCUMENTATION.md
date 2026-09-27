# Parallax: Comprehensive Project Documentation

> **Project:** Parallax — Amazon ML Challenge 2026 Entity Resolution Engine  
> **Branch:** `feature/v3a-hyper-resolution`  
> **Last Updated:** September 27, 2026  
> **Codebase:** 42 source modules (6,905 LoC) · 20 test files (1,796 LoC) · 65 passing unit tests

---

## Table of Contents

1. [Problem Statement & Evaluation Metric](#1-problem-statement--evaluation-metric)
2. [Team & Compute Architecture](#2-team--compute-architecture)
3. [Repository Structure](#3-repository-structure)
4. [End-to-End Pipeline Architecture](#4-end-to-end-pipeline-architecture)
5. [Component Deep Dives](#5-component-deep-dives)
6. [Feature Engineering: The 28-Feature Vector](#6-feature-engineering-the-28-feature-vector)
7. [Engineering Timeline & Git History](#7-engineering-timeline--git-history)
8. [DGX-1 Parallelization Engineering](#8-dgx-1-parallelization-engineering)
9. [Benchmark Results: All DGX Runs](#9-benchmark-results-all-dgx-runs)
10. [Failure Analysis & Known Gaps](#10-failure-analysis--known-gaps)
11. [Quality Gate & Testing](#11-quality-gate--testing)
12. [Data Assets & Storage](#12-data-assets--storage)
13. [Future Work & Open Exploration Avenues](#13-future-work--open-exploration-avenues)

---

## 1. Problem Statement & Evaluation Metric

### Problem Framing

The objective is **multi-source business entity resolution** across three disparate tables:

- **Source 1 ($S_1$):** Query entity table — 200,000 business records with `entity_id`, `business_name`, `business_address`, `country`.
- **Source 2 ($S_2$) & Source 3 ($S_3$):** Target candidate tables — 500,000 records each containing potential matching business entities.
- **Ground Truth:** Maps each $S_1$ entity ID to a comma-separated list of matching entity IDs in $S_2$/$S_3$. **Singletons** (entities with 0 matches) map to an empty string.

### The Competition Metric: Macro $F_{0.5}$

The evaluation metric is **Macro $F_{0.5}$** across all $N$ Source 1 entities:

$$F_{0.5}(s_1) = \begin{cases} 
1.0 & \text{if } |\text{true}| = 0 \text{ and } |\text{pred}| = 0 \quad (\text{Correct Singleton}) \\
0.0 & \text{if } |\text{true}| = 0 \text{ and } |\text{pred}| > 0 \quad (\text{Singleton Violation}) \\
0.0 & \text{if } |\text{true}| > 0 \text{ and } |\text{pred}| = 0 \quad (\text{Total False Negative}) \\
\frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R} & \text{otherwise}
\end{cases}$$

$$\text{Macro } F_{0.5} = \frac{1}{N} \sum_{i=1}^N F_{0.5}(s_1^{(i)})$$

**Key metric behavior:**
1. **Precision is weighted 2× as heavily as Recall** ($\beta = 0.5$). Every false merge positive is penalized twice as hard.
2. **True Singletons are all-or-nothing:** Correctly identifying a singleton awards $1.0$. A single spurious prediction drops it to $0.0$.

---

## 2. Team & Compute Architecture

### Team Roles

| Teammate | Focus Area | Key Deliverables |
|---|---|---|
| **Teammate A (Akshit — Infra & Tabular/Rules)** | Data pipeline, S3 sync, blocking, features, LightGBM, parallelization | Full V1–V3a engine, benchmark suite, DGX parallelization |
| **Teammate B (NLP / Text Specialist)** | Text representation, semantic models | DeBERTa-v3 / cross-encoder scoring |
| **Teammate C (Vision / OCR Specialist)** | Product images, OCR, address parsing | Visual features, address decomposition |

### Compute Tiers

```
┌────────────────────────────────────────────────────────────────────────┐
│                        COMPUTE ARCHITECTURE                            │
├────────────────────────────────┬───────────────────────────────────────┤
│ Heavy CPU Data Prep & S3       │ AWS EC2 (c6i.xlarge Spot) [Credits]   │
│ Day 1 GPU Prototyping          │ Kaggle (90h Free T4 / 2x T4 GPUs)     │
│ Asynchronous Full Downloads    │ AWS EC2 Background Detached tmux      │
│ Day 2 Heavy Training & CV     │ College DGX-1 (80 CPU, 503 GB RAM)    │
│ Final Ensembling & Submission  │ Local + Kaggle / AWS G4dn             │
└────────────────────────────────┴───────────────────────────────────────┘
```

### DGX-1 Hardware Topology (`sid-dgx`)

- **CPU:** Dual Intel Xeon E5-2698 v4 @ 2.20 GHz (2 Sockets × 20 Physical Cores = 40 Cores / 80 Threads)
- **RAM:** 503 GB DDR4 (Dual NUMA: Node 0 CPUs 0–19,40–59; Node 1 CPUs 20–39,60–79)
- **Python:** 3.10 via Miniconda (`/home/23dcs510/miniconda3/envs/py310`)

---

## 3. Repository Structure

```
Parallax/
├── .agents/rules/            # AGY SDK & agent behavioral invariants
├── data/
│   ├── raw/                  # Original competition TSVs
│   ├── processed/            # Cleaned parquet data & embeddings
│   ├── golden_split/         # 5k Golden Benchmark (Day-1 smoke test)
│   └── medium_split_200k/    # 200k Medium Benchmark (representative scale)
├── docs/
│   └── DGX_PARALLELIZATION_GUIDE.md  # DGX-1 parallelization playbook
├── notebooks/                # EDA & Kaggle prototypes
├── output/
│   ├── checkpoints/          # Per-fold metrics, scored pairs, manifest
│   ├── backup/               # Archived earlier run checkpoints
│   └── *.parquet             # Candidate pairs, features, results
├── reports/
│   ├── architecture_brief_v1.md        # Full system architecture doc
│   ├── diagnostics_medium_200k.md      # Latest failure diagnostics
│   ├── diagnostics_sample_5000.md      # Golden split diagnostics
│   ├── feature_gap_analysis.md         # 28-feature gap analysis
│   ├── failures_medium_200k.jsonl      # Structured error logs (28 MB)
│   ├── team_status_report_v1.md        # Team sync report
│   └── benchmark_execution.log         # Full DGX execution log
├── scripts/
│   ├── run_medium_benchmark.py         # CLI wrapper for 200k benchmark
│   ├── run_full_submission.py          # Full-scale submission pipeline
│   └── kaggle_resource_test.py         # Kaggle notebook runner
├── src/parallax/                       # Core library (42 modules, 6,905 LoC)
│   ├── blocking/
│   │   └── tfidf_blocker.py            # Dual-Channel TF-IDF Blocker (617 lines)
│   ├── config.py                       # Pydantic v2 configuration schemas
│   ├── core.py                         # Health checks & entry points
│   ├── data/
│   │   ├── contracts.py                # Data loading contracts
│   │   ├── downloader.py               # Async image downloader
│   │   └── splitter.py                 # Stratified K-Fold & Golden Split generator
│   ├── diagnostics/
│   │   ├── execution_logger.py         # Pipeline stage telemetry
│   │   └── failure_logger.py           # Error categorization & JSONL tracing
│   ├── experiments/
│   │   └── runner.py                   # Main benchmark orchestrator (974 lines)
│   ├── features/
│   │   ├── extractor.py                # Pairwise Feature Extractor (899 lines)
│   │   └── meta_features.py            # Entity-level aggregation features
│   ├── metrics/
│   │   └── evaluator.py                # Macro F0.5, singleton accuracy, diagnostics
│   ├── models/
│   │   ├── base.py                     # Abstract BaseModel interface
│   │   ├── matcher.py                  # LightGBM binary classifier
│   │   ├── rules.py                    # Deterministic regex rule engine
│   │   └── ensemble.py                 # Priority fallback & voting
│   ├── observability/
│   │   └── tracer.py                   # Structured trace spans
│   ├── pipeline.py                     # End-to-end orchestration
│   ├── postprocessing/
│   │   └── singleton_gate.py           # Confidence-based singleton gating
│   ├── preprocessing/
│   │   ├── normalizer.py               # Unicode NFKC, soft names, widening
│   │   └── transliteration.py          # Brahmic → Latin phonetic mapper
│   ├── s3_utils.py                     # S3 bucket operations
│   ├── submission/
│   │   ├── formatter.py                # Competition submission formatting
│   │   ├── validator.py                # Submission validation
│   │   └── test_inference.py           # Inference smoke test
│   ├── training/
│   │   └── full_trainer.py             # Full-scale training pipeline
│   └── utils/
│       ├── checkpoint_manager.py       # Resumable checkpoint system
│       └── s3_sync.py                  # S3 sync CLI
├── tests/                              # 20 test files, 1,796 LoC, 65 tests
├── Makefile                            # Quality gate: ruff + mypy + pytest
├── pyproject.toml                      # uv project config
└── GEMINI.md                           # AI Constitution
```

### Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `lightgbm` | ≥ 4.7.0 | Gradient-boosted decision tree classifier |
| `rapidfuzz` | ≥ 3.14.5 | C++ Levenshtein, Jaro-Winkler, token set/sort ratios |
| `jellyfish` | ≥ 1.2.1 | Jaro-Winkler similarity (alternative) |
| `scikit-learn` | ≥ 1.7.2 | TF-IDF vectorizer, cosine similarity, metrics |
| `pandas` | ≥ 2.3.3 | DataFrame operations, parquet I/O |
| `numpy` | ≥ 2.2.6 | Array operations, sparse matrix math |
| `pydantic` | ≥ 2.7.0 | Configuration schemas, data validation |
| `pyarrow` | ≥ 25.0.1 | Parquet serialization |
| `boto3` | ≥ 1.43.101 | AWS S3 operations |
| `aiohttp` | ≥ 3.14.3 | Async image downloading |
| `tqdm` | ≥ 4.70.1 | Progress bars |

---

## 4. End-to-End Pipeline Architecture

```
[Raw TSVs: S1 (200k), S2 (500k), S3 (500k)]
         │
         ▼
[Stage 1: Preprocessing & Unicode Widening]
    ├── Unicode NFKC canonical normalization
    ├── Brahmic-to-Latin phonetic transliteration (7 Indic scripts)
    ├── Building/unit number extraction (regex)
    ├── Soft name generation (lowered, alphanumeric tokens)
    └── Null address indicators
         │
         ▼
[Stage 2: Country-Partitioned Dual-Channel TF-IDF Blocker]
    ├── Channel A: Character 3-gram TF-IDF on soft_name + translit_name
    ├── Channel B: Character 3-gram TF-IDF on clean addresses
    ├── Channel C: Building number inverted index + prefix matching
    └── Output: Union of all channels (max_candidates=35 per query)
         │
         ▼
[Candidate Pairs (~6.98M pairs, ~35 candidates/query)]
         │
         ▼
[Stage 3: RapidFuzz Pairwise Feature Extraction]
    └── 28 dense features per pair (lexical, structural, phonetic)
         │
         ▼
[Feature Matrix (Snappy Parquet)]
         │
         ▼
[Stage 4: LightGBM Hyperparameter Grid Sweep]
    ├── Binary classification (objective="binary")
    ├── Grid over: num_leaves, max_depth, learning_rate, n_estimators
    └── Threshold τ calibrated on validation Macro F0.5
         │
         ▼
[Stage 5: 5-Fold Stratified Cross Validation]
    ├── 5 concurrent fold workers (ProcessPoolExecutor, fork COW)
    ├── Each worker: LightGBM train → predict → τ-optimize
    └── Output: Per-fold metrics, OOF predictions
         │
         ▼
[Stage 6: Singleton Gating & Post-Processing]
    ├── If max_prob(s1) < τ → singleton (empty prediction set)
    └── If max_prob(s1) ≥ τ → emit all candidates above τ
         │
         ▼
[Stage 7: Failure Diagnostics & Error Triaging]
    ├── Categorize: Blocking FN, Classification FN, False Merge, Singleton Violation
    ├── JSONL error log (reports/failures_medium_200k.jsonl)
    └── Executive scorecard (reports/diagnostics_medium_200k.md)
         │
         ▼
[matching_results.tsv] & [diagnostics report]
```

---

## 5. Component Deep Dives

### 5.1 Preprocessing & Transliteration

**Module:** `src/parallax/preprocessing/normalizer.py`

The normalizer transforms raw business records into enriched "wide" DataFrames with derived columns for downstream matching:

| Output Column | Derivation |
|---|---|
| `soft_name` | Lowered, domain-stripped, alphanumeric-only business name |
| `token_sorted_name` | Alphabetically sorted tokens of `soft_name` |
| `translit_name` | Brahmic → Latin phonetic transliteration of `business_name` |
| `clean_address` | Canonicalized address (abbreviation expansion, whitespace collapse) |
| `translit_address` | Brahmic → Latin transliteration of `business_address` |
| `numbers` | Regex-extracted set of building/unit numbers |
| `primary_number` | First extracted number (building number heuristic) |
| `is_addr_null` | Binary: address is null or empty |
| `city_token` | Extracted city/locality token from address |
| `postal_code` | Extracted ZIP/PIN code |

**Parallelization:** `widen_records_df(df, n_jobs=20)` uses `ProcessPoolExecutor` with `fork` context and index-based DataFrame chunking (safe against NumPy 2.x `array_split` deprecation).

**Module:** `src/parallax/preprocessing/transliteration.py`

Zero-dependency, deterministic phonetic transliteration covering **7 Brahmic scripts**:
- Devanagari (Hindi, Marathi, Sanskrit)
- Bengali (Bangla)
- Gujarati
- Gurmukhi (Punjabi)
- Kannada
- Telugu
- Tamil
- Malayalam

Uses Unicode codepoint offset mapping (`cp % 0x80`) against a 70-entry `_BRAHMIC_OFFSET_MAP` to produce Latin phonemes. O(n) single-pass, no external API calls.

### 5.2 Dual-Channel TF-IDF Blocker

**Module:** `src/parallax/blocking/tfidf_blocker.py` (617 lines)

**Class:** `DualChannelTFIDFBlocker`

The blocker reduces the $200\text{k} \times 1\text{M} = 200\text{B}$ potential pairwise comparison space to ~6.98M candidate pairs (a **99.994% reduction ratio**).

**Three-to-Six candidate channels (evolved from V1 → V3a):**

| Channel | Input | Vectorizer / Index | top_k | min_sim |
|---|---|---|---|---|
| **A (Name TF-IDF)** | `soft_name + " " + translit_name` | Word (1,2)-gram TF-IDF (`sublinear_tf`, `max_df=0.05`) | 15 | 0.15 |
| **B (Address TF-IDF)** | `clean_address + translit_address` | Word (1,2)-gram TF-IDF (`max_df=0.02`) | 10 | 0.20 |
| **C (Building Number)** | Extracted numbers + 2-char name prefix | Inverted index (exact match) | — | — |
| **D (Postal Hash-Join)** | 5-digit US ZIP / 6-digit Indian PIN | Hash join + name length ratio ≥ 0.40 | — | — |
| **E (Phonetic First-Token)** | `jellyfish.metaphone` on first soft name token | Inverted index + length ratio ≥ 0.50 | — | — |
| **F (Address Token)** | `(primary_number, city_token)` composite key | Hash join (bucket ≤ 20) | — | — |

**Union & Truncation:** Candidates from all channels are merged per S1 entity, taking the max similarity score across channels. If exceeding `max_candidates_per_query=35`, truncated to top-K by score.

**Blocking performance:**
- **Pair Completeness (Recall):** ~95.8%
- **Reduction Ratio:** 99.994%

**Parallelization:** Country-partitioned processing. Within each country, query entities are chunked across `n_jobs` workers using `ProcessPoolExecutor(mp_context=fork)`. The fitted TF-IDF target matrices and inverted indices are stored in module-level globals (`_CURRENT_BLOCKER`) for zero-copy COW sharing across forked child processes.

### 5.3 Pairwise Feature Extractor

**Module:** `src/parallax/features/extractor.py` (899 lines)

**Class:** `PairwiseFeatureExtractor`

For each (S1, candidate) pair, computes 28 dense similarity features using RapidFuzz C++ bindings and Jellyfish. Features are computed from prebuilt record lookup dictionaries (`s1_dict`, `target_dict`).

**Parallelization:** Candidate pair keys are partitioned across `n_jobs` workers. The lookup dictionaries are stored in module globals (`_CURRENT_S1_DICT`, `_CURRENT_TARGET_DICT`, `_CURRENT_EXTRACTOR`) for fork COW IPC. Each worker processes its chunk independently and returns a partial DataFrame. Results are concatenated with `pd.concat(chunk_dfs, ignore_index=True)`.

**Throughput:** ~35,000–50,000 pairs/second on DGX-1 with 20 workers.

### 5.4 LightGBM Classifier & Two-Pass Resolution

**Module:** `src/parallax/models/matcher.py`

Binary LightGBM classifier with asymmetric threshold calibration and a **Two-Pass resolution strategy**:

**Hyperparameter Grid (6 configs, swept on Fold 0):**

| Config | `learning_rate` | `num_leaves` | `max_depth` | `n_estimators` |
|---|:---:|:---:|:---:|:---:|
| Config-Fast | 0.08 | 31 | 6 | 100 |
| Config-Balanced | 0.05 | 31 | 6 | 150 |
| Config-Deep | 0.05 | 63 | 8 | 150 |
| Config-Dense | 0.04 | 63 | 8 | 250 |
| **Config-XDeep** ★ | **0.04** | **127** | **10** | **250** |
| Config-Conservative | 0.03 | 31 | 6 | 180 |

All configs enforce `num_threads = 8` (NUMA-bounded).

**Two-Pass Training Strategy:**

1. **Pass 1:** Train LightGBM on 49 raw pairwise features → generate match probabilities for all pairs
2. **Meta-Feature Engineering:** Compute 5 entity-level distribution features from Pass 1 probabilities
3. **Pass 2:** Train LightGBM on 54 features (49 raw + 5 meta) → final probabilities
4. **Threshold Calibration:** Sweep $\tau \in [0.50, 0.54, 0.58, \ldots, 0.90]$ maximizing Macro $F_{0.5}$

### 5.5 Singleton Gating

**Module:** `src/parallax/postprocessing/singleton_gate.py`

For each query entity $s_1$:
- If $\max_c P(s_1, c) < \tau$: predicted match set is $\emptyset$ (entity classified as singleton, scored 1.0)
- If one or more candidates have $P(s_1, c) \geq \tau$: output all such matches comma-separated

### 5.6 Failure Diagnostics System

**Module:** `src/parallax/diagnostics/failure_logger.py`

Evaluates full out-of-fold predictions against ground truth and categorizes every error:

| Error Category | Description | Metric Impact |
|---|---|---|
| `BLOCKING_FALSE_NEGATIVE` | True match never retrieved by blocker | Defines recall ceiling; no model fix possible |
| `CLASSIFICATION_FALSE_NEGATIVE` | True match retrieved but model score < τ | Reduces recall |
| `FALSE_MERGE_POSITIVE` | Distractor predicted above τ | **Penalized 2× under $F_{0.5}$** |
| `SINGLETON_VIOLATION` | True singleton assigned ≥1 candidate | Entity score drops from 1.0 → 0.0 |

Outputs:
- **Structured JSONL:** `reports/failures_medium_200k.jsonl` (28 MB, every individual error)
- **Executive Markdown:** `reports/diagnostics_medium_200k.md` (scorecard + sample case studies)

### 5.7 Checkpoint & Resumability System

**Module:** `src/parallax/utils/checkpoint_manager.py`

Saves intermediate pipeline state (widened DataFrames, candidate pairs, features, per-fold metrics) as Snappy-compressed Parquet files. Allows resuming pipeline from any checkpoint after crashes or interruptions. Manifest stored in `output/checkpoints/manifest.json`.

---

## 6. Feature Engineering: The 49+5 Feature Architecture

The feature set evolved from 13 features (V1) → 28 features (V1.2) → **49 raw features + 5 entity meta-features = 54 total** (V3a Two-Pass).

### 6.1 Pass 1: 49 Raw Pairwise Features

| # | Feature | Type | Information Signal |
|:--|:--------|:-----|:-------------------|
| | **A. Sequence-Level Name Similarity (5)** | | |
| 1 | `raw_name_ratio` | float | Levenshtein ratio on raw business name |
| 2 | `soft_name_ratio` | float | Levenshtein ratio on normalized name |
| 3 | `token_sort_ratio` | float | Sorted-token Levenshtein (order-invariant) |
| 4 | `token_set_ratio` | float | Set-intersection Levenshtein (subset-tolerant) |
| 5 | `partial_ratio` | float | Best-substring Levenshtein (length-invariant) |
| | **B. Prefix-Weighted Name Similarity (2)** | | |
| 6 | `jaro_winkler_soft` | float | JW on soft name (prefix bonus) |
| 7 | `jaro_winkler_raw` | float | JW on raw name |
| | **C. Token-Set Name Analysis (3)** | | |
| 8 | `token_jaccard_name` | float | Jaccard of name word sets |
| 9 | `token_overlap_name` | float | Overlap coefficient |
| 10 | `first_token_match` | binary | Exact match on first word |
| | **D. Name Length Geometry (2)** | | |
| 11 | `len_diff_name` | int | Absolute character length difference |
| 12 | `len_ratio_name` | float | min(len) / max(len) |
| | **E. Address Sequence Similarity (4)** | | |
| 13 | `addr_ratio` | float | Levenshtein ratio on clean address |
| 14 | `addr_token_set_ratio` | float | Token-set Levenshtein on address |
| 15 | `canon_addr_ratio` | float | Levenshtein on canonicalized address |
| 16 | `token_jaccard_addr` | float | Jaccard of address word sets |
| | **F. Structural Number Discrimination (5)** | | |
| 17 | `num_match_score` | ternary | Number overlap: 1.0 / 0.0 / 0.5 (missing) |
| 18 | `primary_num_match` | binary | Primary building number exact match |
| 19 | `primary_num_conflict` | binary | Both have different primary numbers |
| 20 | `num_jaccard` | float | Jaccard of all extracted numbers |
| 21 | `num_conflict_count` | int | Count of non-shared numbers |
| | **G. Postal Code Alignment (3)** | | |
| 22 | `postal_match` | binary | ZIP/PIN exact match |
| 23 | `postal_conflict` | binary | Both have different postal codes |
| 24 | `postal_missing` | binary | At least one postal code absent |
| | **H. Null / Presence Indicators (3)** | | |
| 25 | `is_s1_addr_null` | binary | S1 address is null |
| 26 | `is_cand_addr_null` | binary | Candidate address is null |
| 27 | `both_addr_present` | binary | Neither address is null |
| | **I. Primary Number Presence (1)** | | |
| 28 | `primary_num_missing` | binary | At least one has no primary number |
| | **J. Address Structural Details (2)** | | |
| 29 | `primary_num_distance` | int | Numerical distance between primary numbers |
| 30 | `city_match` | binary | City token match (exact or ≥80 similarity) |
| 31 | `city_conflict` | binary | Both present but conflicting city tokens |
| | **K. Interaction Features (4)** | | |
| 32 | `name_ratio_null_cand_addr` | float | `soft_name_ratio × is_cand_addr_null` |
| 33 | `name_ratio_null_either_addr` | float | `soft_name_ratio × (1 - both_addr_present)` |
| 34 | `high_conf_name_no_addr` | binary | `soft_name_ratio ≥ 0.85` and no address |
| 35 | `token_set_null_cand_addr` | float | `token_set_ratio × is_cand_addr_null` |
| | **L. Cross-Script & Transliteration (3)** | | |
| 36 | `translit_boost_name` | float | Delta: max(0, translit_score - raw_score) |
| 37 | `is_cross_script` | binary | Candidate name contains non-Latin Unicode |
| 38 | `translit_name_ratio` | float | Levenshtein on both transliterated names |
| | **M. Name Decomposition (3)** | | |
| 39 | `name_core_ratio` | float | Levenshtein on core names (suffix/prefix stripped) |
| 40 | `suffix_match` | ternary | Legal suffix match: 1.0 / 0.0 / 0.5 (missing) |
| 41 | `name_word_count_diff` | int | Absolute difference in token counts |
| | **N. Character-Level Patterns (3)** | | |
| 42 | `name_edit_distance` | int | Raw Levenshtein edit distance count |
| 43 | `common_prefix_len` | int | Longest common character prefix length |
| 44 | `containment_ratio_name` | binary | One name is substring of the other (min len ≥ 3) |
| | **O. Phonetic & Address Advanced (3)** | | |
| 45 | `addr_token_overlap` | float | Overlap coefficient of address tokens |
| 46 | `phonetic_name_match` | binary | Metaphone code match on first word |
| 47 | `country_match` | binary | Country fields match exactly |
| | **P. Contextual Signals (2)** | | |
| 48 | `s1_candidate_count` | int | Total candidate pool size for this S1 |
| 49 | `blocking_sim_score` | float | Preserved similarity score from blocker |

### 6.2 Pass 2: Entity Meta-Features (5 Additional)

After Pass 1 LightGBM generates match probabilities, 5 **entity-level distribution features** are computed per S1 query:

| # | Feature | Derivation |
|:--|:--------|:-----------|
| 50 | `s1_max_score` | $\max_{c \in \text{Cands}} P(\text{match})$ |
| 51 | `s1_mean_score` | $\mu_c P(\text{match})$ across all candidates |
| 52 | `s1_std_score` | $\sigma_c P(\text{match})$ — spread of probabilities |
| 53 | `score_rank_pct` | Percentile rank of candidate within S1's pool |
| 54 | `score_gap_to_best` | Margin between top candidate and current |

These features allow Pass 2 LightGBM to reason about **entity context** — distinguishing singletons (many mediocre candidates) from true matches (one dominant candidate).

---

## 7. Engineering Timeline & Git History

The project evolved through 22 commits across two branches:

### Phase 1: Bootstrap & Infrastructure (`main`)

| Date | Commit | Description |
|---|---|---|
| Sep 25 | `06d45dc` | Initialize Parallax engine — `uv` project scaffold, Pydantic config, S3 sync |
| Sep 25 | `6a9a841` | Add `.env.example` and auto-loading in config |
| Sep 25 | `5546f6b` | Update compute allocation for rejected AWS GPU; add Kaggle bridge |
| Sep 25 | `ca283f8` | Configure exact S3 bucket name `amazon-ml-challange-2026-parallax` |

### Phase 2: V1 Entity Resolution (`feature/v1-entity-resolution`)

| Date | Commit | Description |
|---|---|---|
| Sep 25 | `7c2c347` | **V1 Pipeline:** Implement blocking, feature extraction, LightGBM, failure diagnostics |
| Sep 25 | `7481515` | **Brahmic Transliteration:** Add deterministic 7-script phonetic mapper → **Macro F0.5 = 0.9857** on 5k |
| Sep 25 | `5197d07` | **200k Runner:** Add benchmark runner with live progress, failure logging, checkpoints |

### Phase 3: V3a Hyper-Resolution & Parallelization (`feature/v3a-hyper-resolution`)

| Date | Commit | Description |
|---|---|---|
| Sep 25 | `693b5d4` | Experiment A: Expand from 13 → 28 features + Kaggle runner |
| Sep 25 | `7d15bed` | Fix Kaggle runner diagnostic method calls |
| Sep 25 | `8d7f4bf` | **V3a Engine:** Stateful low-memory blocker, hyper-resolution candidate expansion |
| Sep 25 | `f9975f0` | Fix orchestrator: sys.path bootstrap, real-time observability telemetry |
| Sep 25 | `d188da0` | **21× blocking speedup:** Word-level address vectorizer + calibrated 15k training sample |
| Sep 25 | `83c5442` | Batch size → 2000, precompute target lookup, short-circuit model reuse |
| Sep 25 | `3d71e6e` | Eliminate redundant RapidFuzz calls, cap candidates to top-15, per-chunk telemetry |
| Sep 26 | `1da41b9` | **52× dot product speedup:** Word (1,2) single-char token pattern on TF-IDF (+2.5–4.6% recall) |
| Sep 26 | `6e57682` | **Parallel CV:** 5 concurrent fold workers via ProcessPoolExecutor (40 OMP threads total) |
| Sep 26 | `12bb6f0` | Update 200k benchmark failure diagnostics and 5-fold CV scorecard |
| Sep 26 | `f815fca` | **End-to-end parallelization:** Multi-process Steps 2, 3, and 4 |
| Sep 26 | `f3ca4a3` | **NUMA fix:** Cap LightGBM OpenMP threads to 8 to prevent cross-socket contention |
| Sep 26 | `5f9e6e6` | Update 200k empirical diagnostic scorecard — **Macro F0.5 = 0.9543** |
| Sep 27 | `8aa89ae` | **DGX Parallelization Guide:** Add `docs/DGX_PARALLELIZATION_GUIDE.md` playbook |

---

## 8. DGX-1 Parallelization Engineering

### The NUMA Thread Contention Problem

When LightGBM/OpenMP detects 80 logical cores, it spawns 80 threads by default. With 5 concurrent CV fold workers, the OS schedules **400 competing threads** across two NUMA sockets, causing cache-line bouncing and near-zero actual throughput despite 100% reported CPU usage.

**Solution:** Enforce the invariant:

$$\text{Workers} \times \text{Threads per Worker} \leq 40 \quad (\text{Physical Core Count})$$

### Three Parallelization Patterns Implemented

| Pattern | Used In | Mechanism |
|---|---|---|
| **A. DataFrame Slice Chunking** | `normalizer.widen_records_df()` | `df.iloc[i:i+chunk_size].copy()` → `ProcessPoolExecutor(fork)` |
| **B. Zero-Copy COW IPC** | `tfidf_blocker`, `extractor` | Store large lookups in module globals before fork; children read parent memory pages via Linux COW |
| **C. OpenMP Thread Capping** | `runner.evaluate_single_fold()` | `num_threads = max(1, 40 // n_workers)` per LightGBM instance |

### Speedup Results (200k Scale)

| Pipeline Stage | Serial | Parallelized | Strategy | Speedup |
|---|:---:|:---:|---|:---:|
| Preprocessing & Widening | 154.5s | **59.7s** | 20-worker slice chunking | **2.6×** |
| TF-IDF Blocking | 1,323.1s (~22 min) | **219.9s** (~3.6 min) | 20-worker zero-copy fork | **6.0×** |
| Feature Extraction | 409.4s (~6.8 min) | **213.0s** (~3.5 min) | 20-worker COW lookup | **1.9×** |
| 5-Fold Cross Validation | 1,323.4s (~22 min) | **760.7s** (~12.6 min) | 5 workers × 8 threads | **1.74×** |
| **Total Pipeline** | **~72.0 min** | **~21.3 min** | **End-to-end** | **3.38×** |

---

## 9. Benchmark Results: All DGX Runs

### 9.1 Master Scorecard

| Run | Dataset | Architecture | Macro $F_{0.5}$ | Precision | Recall | Singleton Acc. | τ | Time |
|---|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Run 1** | 5k / 25k | 28 Features, Serial | `0.9657 ± 0.0035` | 98.66% | 93.22% | 97.64% | 0.82–0.94 | ~3.5 min |
| **Run 2** | 5k / 25k | + Transliteration | `0.9857 ± 0.0021` | 99.12% | 96.40% | 98.80% | 0.88 | ~4.0 min |
| **Run 3** | 200k / 1M | Baseline (max_cand=15) | `0.9635 ± 0.0007` | 98.77% | 92.21% | 95.29% | 0.66–0.74 | ~72.0 min |
| **Run 4** | 200k / 1M | Uncapped OpenMP | *Aborted* | — | — | — | — | NUMA stall |
| **Run 5** | 200k / 1M | **v3a (max_cand=35, parallel)** | **`0.9543 ± 0.0007`** | **98.96%** | 90.55% | **98.10%** | **0.86** | **21.3 min** |

### 9.2 Latest 200k Run: Per-Fold Breakdown (v3a Hyper-Resolution)

| Fold | Train Pairs | Val Pairs | τ | Macro $F_{0.5}$ | Singleton Acc. | Non-Sing $F_{0.5}$ | Precision | Recall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 5,585,784 | 1,397,092 | 0.86 | **0.9549** | 97.45% | 0.9537 | 98.95% | 90.64% |
| 1 | 5,586,520 | 1,396,356 | 0.86 | **0.9538** | 98.08% | 0.9522 | 98.91% | 90.58% |
| 2 | 5,586,530 | 1,396,346 | 0.86 | **0.9537** | 98.08% | 0.9520 | 98.98% | 90.49% |
| 3 | 5,586,527 | 1,396,349 | 0.86 | **0.9538** | 98.39% | 0.9520 | 98.96% | 90.48% |
| 4 | 5,586,143 | 1,396,733 | 0.86 | **0.9555** | 98.52% | 0.9537 | 99.02% | 90.56% |
| **Mean ± Std** | 5,586,301 | 1,396,575 | 0.86 | **0.9543 ± 0.0007** | **98.10 ± 0.37%** | 0.9527 | **98.96 ± 0.04%** | **90.55 ± 0.06%** |

### 9.3 Baseline 200k Run: Per-Fold Breakdown (Run 3)

| Fold | Train Pairs | Val Pairs | τ | Macro $F_{0.5}$ | Singleton Acc. | Non-Sing $F_{0.5}$ | Precision | Recall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 9,209,230 | 2,302,282 | 0.74 | 0.9640 | 95.75% | 0.9644 | 98.99% | 91.65% |
| 1 | 9,208,471 | 2,303,041 | 0.66 | 0.9635 | 94.85% | 0.9644 | 98.56% | 92.57% |
| 2 | 9,209,470 | 2,302,042 | 0.66 | 0.9633 | 95.26% | 0.9639 | 98.61% | 92.37% |
| 3 | 9,209,328 | 2,302,184 | 0.70 | 0.9623 | 95.12% | 0.9629 | 98.77% | 91.86% |
| 4 | 9,209,549 | 2,301,963 | 0.66 | 0.9642 | 95.43% | 0.9648 | 98.62% | 92.62% |
| **Mean ± Std** | 9,209,210 | 2,302,302 | 0.68 | **0.9635 ± 0.0007** | **95.29 ± 0.31%** | 0.9641 | **98.77 ± 0.16%** | **92.21 ± 0.40%** |

### 9.4 Golden Split 5k Run: Per-Fold Breakdown

| Fold | Train Pairs | Val Pairs | τ | Macro $F_{0.5}$ | Singleton Acc. | Precision | Recall |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 235,532 | 61,124 | 0.90 | 0.9708 | 98.48% | 98.91% | 93.30% |
| 1 | 238,582 | 58,074 | 0.90 | 0.9623 | 94.34% | 98.88% | 92.25% |
| 2 | 235,164 | 61,492 | 0.90 | 0.9645 | 98.04% | 98.39% | 93.19% |
| 3 | 241,398 | 55,258 | 0.94 | 0.9690 | 98.39% | 98.78% | 93.36% |
| 4 | 235,948 | 60,708 | 0.82 | 0.9623 | 98.44% | 98.32% | 94.01% |
| **Mean ± Std** | 237,325 | 59,331 | 0.89 | **0.9657 ± 0.0035** | **97.64 ± 1.66%** | **98.66 ± 0.27%** | **93.22 ± 0.57%** |

### 9.5 Failure Taxonomy Comparison (200k)

| Error Category | Baseline (Run 3) | v3a (Run 5) | Delta |
|---|:---:|:---:|---|
| **Blocking False Negatives** | 73,593 | **42,536** | **−42.2%** (31,057 matches recovered) |
| **Classification False Negatives** | 16,812 | 22,863 | +36% (stricter τ=0.86) |
| **False Merge Positives** | 8,941 | **6,310** | **−29.4%** |
| **Singleton Violations** | 526 | **248** | **−52.9%** |
| **Singleton Accuracy** | 95.29% | **98.10%** | **+2.81%** |

---

## 10. Failure Analysis & Known Gaps

### 10.1 Top 7 Feature Gaps Identified

Based on systematic failure case analysis in `reports/feature_gap_analysis.md`:

| Gap # | Category | Description | Impact |
|---|---|---|---|
| **1** | Script-Crossing Signal | No feature captures _how much_ transliteration helped | 33% of classification FNs |
| **2** | Entity-Level Aggregation | Model sees pairs independently, no candidate distribution context | Singleton violations |
| **3** | Phonetic Similarity | No Soundex, Metaphone, or NYSIIS encoding | Garbled/typo names |
| **4** | Country/Metadata | Country column unused by classifier | Missing priors |
| **5** | Name-Part Decomposition | No separation of core name vs. legal suffix (LLC, Pvt Ltd) | 12% of classification FNs |
| **6** | Character-Level Patterns | No edit distance, LCS, containment, or n-gram overlap features | 8% of classification FNs |
| **7** | Address Decomposition | No city, state, or locality extraction/matching | Same-address false merges |

### 10.2 Sample Failure Case Studies

**Blocking False Negative:**
- `Bryan Square LLC` (850 Gorman Road, Gatesville, TX) ↔ `bryansquare.com` (Gatesville, Texas, Gorman Rd)
- Root cause: Domain-format name completely different in character space

**False Merge Positive:**
- `Dandekar Vidyalaya` (Delhi, Block B, Dilshad Garden) ↔ `Dandekar Vidyalaya Industries` (दिल्ली, BLOCK B DILSHAD GARDEN)
- Model score: 0.956 — Near-identical name + transliterated address match, but distinct entities

**Singleton Violation:**
- `Gulf Voya` (6 Edgewater Road, Agawam, MA) ↔ `The Gulf Voya Inc` (19 Edgewater Road, Agawam, MA)
- Model score: 0.882 — Same street, similar name, but different building number → should be singleton

---

## 11. Quality Gate & Testing

### Quality Gate (`make gate`)

```
make gate → ruff check . && mypy src && pytest -q
```

| Tool | Purpose | Status |
|---|---|---|
| `ruff check .` | Linting (E, F, I, UP, B, SIM rules) | ✅ All checks passed |
| `mypy src` | Strict type checking (42 source files) | ✅ No issues found |
| `pytest -q` | 65 unit tests across 20 test files | ✅ 65 passed, 4 warnings |

### Test Coverage by Component

| Test File | Component Tested | Tests |
|---|---|---|
| `test_er_normalizer.py` | Text normalization, soft names, address cleaning | ~8 |
| `test_er_transliteration.py` | Brahmic script detection & transliteration | 3 |
| `test_er_contracts.py` | Data loading contracts | ~4 |
| `test_er_metrics.py` | Macro F0.5 computation, singleton accuracy | ~6 |
| `test_feature_extractor_v2.py` | 28-feature extraction correctness | ~8 |
| `test_parallel_pipeline.py` | Multi-process equivalence (n_jobs=1 vs n_jobs=2) | 3 |
| `test_full_pipeline_smoke.py` | End-to-end integration smoke test | 1 |
| `test_checkpoint_manager.py` | Checkpoint save/load/resume | ~4 |
| `test_rules.py` | Regex rule engine | ~5 |
| `test_config.py` | Pydantic configuration schemas | ~3 |
| `test_validator.py` / `test_er_validator.py` | Submission format validation | ~6 |
| Others | Splitter, downloader, metrics, observability | ~14 |

---

## 12. Data Assets & Storage

### S3 Central Storage

**Bucket:** `s3://amazon-ml-challange-2026-parallax`

| S3 Path | Contents |
|---|---|
| `splits/medium_split_200k/` | 200k benchmark: S1, S2, S3 TSVs, ground truth, CV folds |
| `golden_split/` | 5k Golden Benchmark for rapid iteration |
| `raw/` | Original competition CSV files |

### 200k Medium Benchmark Composition

| File | Rows | Details |
|---|---:|---|
| `train_source1.tsv` | 200,000 | US: 119,958 (60%) · India: 80,042 (40%) |
| `train_source2.tsv` | 500,000 | 334,213 true matches + 165,787 distractors |
| `train_source3.tsv` | 500,000 | 357,980 true matches + 142,020 distractors |
| `train_ground_truth.tsv` | 200,000 | 11,170 singletons (5.58%) + 188,830 non-singletons |
| `cv_folds_source1.tsv` | 200,000 | 5 balanced folds (40k/fold), stratified by (country, is_singleton) |

### Local Output Artifacts

| Path | Size | Contents |
|---|---|---|
| `output/candidate_pairs_medium_200k.parquet` | ~100 MB | 6.98M candidate pairs |
| `output/features_medium_200k.parquet` | ~320 MB | 6.98M × 28 feature matrix |
| `output/matching_results_medium_200k.tsv` | ~11 MB | Final predictions |
| `output/checkpoints/fold_*_metrics.json` | ~1 KB each | Per-fold metrics |
| `reports/failures_medium_200k.jsonl` | ~28 MB | Every single error case |

---

## 13. Future Work & Open Exploration Avenues

### High-Impact Feature Additions (from Gap Analysis)

1. **Transliteration boost features:** `translit_score - raw_score` as explicit signal for cross-script pairs
2. **Entity-level aggregation:** Candidate count, score rank, max/mean/std of scores per S1
3. **Phonetic encoding:** Soundex or Metaphone features for garbled name matching
4. **Name decomposition:** Strip legal suffixes (LLC, Pvt Ltd, Inc) and compare core names separately

### Alternative Blocking Mechanisms

- **Dense bi-encoder embeddings** (e.g. `bge-m3`, `MiniLM`) + FAISS/HNSW for semantic blocking
- **Learned blocking** / DeepMatcher for capturing synonyms that character n-grams miss

### Advanced Classification

- **CatBoost / XGBoost ensemble** with LightGBM for model diversity
- **Cross-encoder reranker** applied only to top-5 candidates per query (high precision, high cost)

### Graph-Based Post-Processing

- **Transitive closure validation** across S1 ↔ S2 ↔ S3 to resolve conflicting multi-match predictions
- **Connected component clustering** (Louvain, label propagation) for entity group coherence

### Address Intelligence

- **`libpostal`** or token-based address parsing for structured city/state/postal comparison
- **Geocoding features** (if external APIs available) for distance-based discrimination
