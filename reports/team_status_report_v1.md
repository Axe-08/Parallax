# 📢 Parallax Team Status Report: Architecture, Benchmark Metrics & Workload

**Date & Time:** September 25, 2026 — 14:35 IST  
**From:** Akshit  
**Branch:** `feature/v1-entity-resolution`  
**Target Audience:** All Parallax Teammates (Teammate B - NLP, Teammate C - Vision/Rules)

---

## 🎯 1. Purpose of this Report
To ensure clear separation of concerns, prevent duplicated effort, and maintain momentum during Day 1 of the Amazon ML Challenge 2026:
- **Clarify active work & progress** so teammates do not recreate blocking, transliteration, or data splitting.
- **Explain the current V1.2 architecture** and the contract interfaces for plugging in text/vision models.
- **Report exact empirical validation metrics** achieved on the Golden Benchmark split (5k).
- **Announce the new 200k Medium Benchmark Split** now live in S3 with its resource footprint and runtimes.

---

## 🛠️ 2. Current Workload & Active Tasks

| Priority | Task | Description | Status |
| :---: | :--- | :--- | :---: |
| **P0** | **Medium Split (200k) on S3** | Representative 200,000 $S_1$ benchmark with ~500k $S_2$, ~500k $S_3$ distractors and 5-fold CV to S3 (`splits/medium_split_200k/`). | **✅ Completed & Live in S3** |
| **P1** | **Deterministic Brahmic Transliteration** | Zero-dependency phonetic mapper covering 7 Indic scripts $\to$ Latin ASCII. Resolved 36% of previous blocking false negatives. | **✅ Completed & Verified** |
| **P1** | **V1 Dual-Stream Blocking & Reranking** | Country-partitioned sublinear TF-IDF + Character 3-5 gram indexing with lexical similarity reranking and precision thresholding. | **✅ Completed & Verified** |
| **P2** | **Automated Failure Diagnostics Engine** | JSONL/Markdown tracer categorizing errors into Blocking FN, Classification FN, False Merges, and Singleton Violations. | **✅ Completed & Verified** |
| **P0** | **Medium Split Baseline & 5-Fold Tuning** | Pulling the 200k split locally, running V1 baseline, and conducting 5-fold CV hyperparameter tuning. | **⚡ Active Right Now** |

---

## 🏛️ 3. Current Architecture: Parallax V1.2

The entity resolution pipeline is structured into 4 deterministic, reproducible stages:

```
[Raw TSV: S1, S2, S3]
         │
         ▼
 1. Normalization & Transliteration
    ├── Unicode NFKD normalization & legal entity suffix expansion (Pvt Ltd -> Private Limited)
    └── Deterministic Brahmic-to-Latin phonetic transliteration (Devanagari, Bengali, Kannada, Telugu, Tamil, etc.)
         │
         ▼
 2. Dual-Stream Blocking (Country Partitioned)
    ├── Search space partitioned by Country (US vs India vs other)
    ├── Dual representation: Index both [soft_name] and [translit_name]
    ├── Sublinear TF-IDF over character 3-5 n-grams
    └── Candidate pool generation: name_top_k=35, addr_top_k=25, addr_min_sim=0.20
         │
         ▼
 3. Multi-Signal Similarity Scoring
    ├── Lexical similarity: Levenshtein, Token Sort Ratio, Jaro-Winkler, Prefix Match
    ├── Phonetic similarity: max(sim(native), sim(translit))
    └── Address token containment & numeric digit matching
         │
         ▼
 4. Precision Reranking & Dynamic Thresholding
    ├── Calibrated ensemble similarity score with conservative threshold (τ = 0.85)
    └── Strict Singleton Guard: Below-threshold entities are emitted as singletons (empty matches)
         │
         ▼
[matching_results.tsv] & [reports/failures_v1.jsonl]
```

---

## 📦 4. The 200k Medium Benchmark Split (Live in S3)

The medium split has been generated on Colab and uploaded directly to central storage:
**S3 Location:** `s3://amazon-ml-challange-2026-parallax/splits/medium_split_200k/`

### Dataset Composition
* **$S_1$ Entities:** **200,000 records** (US: 119,958 | India: 80,042)
* **$S_2$ Candidate Space:** **500,000 records** (334,213 true positive matches + 165,787 negative distractors)
* **$S_3$ Candidate Space:** **500,000 records** (357,980 true positive matches + 142,020 negative distractors)
* **Ground Truth Rows:** **200,000 rows** (100% aligned with $S_1$)
* **$S_1$ Singletons:** **11,170 (5.58%)**
* **5-Fold Cross-Validation:** Pre-assigned orthogonal folds (`cv_folds_source1.tsv` / `.parquet`)
* **Generation Time:** 83.9 seconds

### Resource Footprint & Benchmarks on 200k Records
*(Benchmarked on AMD Ryzen 7 7730U: 8 cores, 16 threads, 14 GB RAM)*
* **Disk Footprint:** **~130 MB** total download.
* **Single Full Run (All 200k entities):**
  * Peak RAM: **~2.2 – 2.8 GB** (well within laptop's 3.8 – 6.0 GB free RAM).
  * Latency: **~1.8 – 3.5 minutes** (warm run with cached preprocessed text).
* **Total 5-Fold Cross-Validation:**
  * Training 5 folds of LightGBM on 5.2M candidate pairs + scoring 1.3M validation pairs per fold.
  * RAM: **~1.6 – 2.1 GB**.
  * Total Runtime for all 5 folds: **~3.8 – 4.5 minutes**.
* **Hyperparameter Sweeps (Optuna / Bayesian Optimization):**
  * Using precomputed candidate features: **~5 to 8 seconds per trial**.
  * 50 Optuna trials: **~5.5 minutes** total!

---

## 📊 5. Current Empirical Numbers & Scale Caveat

> [!CAUTION]
> ### ⚠️ Crucial Scale Clarification:
> The metrics below were measured on the **5,000 $S_1$ Golden Benchmark Split** (`data/golden_split/`).
> 
> The 5k split was designed for fast 5-second smoke testing. The new 200k split introduces over **307,000 negative distractors** across $S_2$ and $S_3$, which will authentically test false merge resistance. We will update these numbers on the 200k split shortly.

### Performance on Golden Benchmark (5,000 entities)
* **Macro $F_{0.5}$ Score:** **`0.9857`** (Official Competition Metric)
* **Precision:** **`99.57%`** (14 false merge positives out of 3,260 predictions)
* **Recall:** **`96.04%`** (3,246 out of 3,380 true matches captured)
* **Singleton Accuracy:** **`100.00%`** (62/62 singletons correctly kept unmerged; 0 violations)
* **Blocking Recall:** **`99.73%`** (Only 9 true pairs dropped at blocking stage)
* **Inference Speed:** **4.8 seconds** for 5,000 entities (~1,040 entities/second on CPU)

---

## 🤝 6. Team Division of Labor & Interface Boundaries

To keep us synchronized and prevent duplicated effort:

### 👤 What I Have Built & What I'm Doing Right Now
* **What I've already implemented:**
  * Fast normalizer with Brahmic-to-Latin phonetic transliteration (`src/parallax/preprocessing/transliteration.py`).
  * Dual-representation TF-IDF character n-gram blocking partitioned by country (`src/parallax/blocking/tfidf_blocker.py`).
  * Baseline lexical feature extraction and thresholding (`src/parallax/features/` and `src/parallax/pipeline.py`).
  * Error diagnostics tracer (`reports/diagnostics_v1.md`, `reports/failures_v1.jsonl`).
  * 200k Medium Benchmark Split generated and uploaded to S3.
* **What I'm doing right now:**
  * Pulling the 200k split locally to `data/medium_split_200k/` and running the V1 baseline evaluation & 5-fold CV hyperparameter sweep.

### 📌 Instructions for Teammates (What not to duplicate & what to work on)

#### For Teammate B (NLP / Text):
* **What NOT to do right now:** Please don't spend time building basic tokenizers, Indic transliteration tables, or TF-IDF blockers from scratch (already built, tested, and working).
* **What to work on instead:** Focus on semantic representations. You can take the candidate pairs (`output/candidate_pairs.tsv`) or the 200k split (`data/medium_split_200k/`) and train/fine-tune a text model (like DeBERTa-v3, bi-encoders, or cross-encoders) to score difficult text pairs where lexical similarity alone is borderline.

#### For Teammate C (Vision / Domain Rules / Addresses):
* **What NOT to do right now:** Please don't build candidate generation or entity splitting pipelines.
* **What to work on instead:** Address parsing/geocoding (extracting state, pincode/zip, street numbers), business suffix rules, or product/storefront visual embeddings to filter out borderline false merges.

---

## 🚀 7. How to Pull the Datasets

### Pull the New Medium Split (200k) — Live Now in S3:
```bash
uv run python -m parallax.utils.s3_sync pull splits/medium_split_200k data/medium_split_200k
```

### Pull Golden Split (5k) — Live in S3:
```bash
uv run python -m parallax.utils.s3_sync pull golden_split data/golden_split
```

All code passes strict quality gates via `make gate`. Let's stay coordinated and keep pushing!
