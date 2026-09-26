# Parallax: Neural Representation Track (E0–E3)

## 1. Overview & Experimental Design

This track evaluates whether learned text representations provide non-redundant signal beyond the frozen 28 baseline lexical features for business entity resolution.

### Experimental Arms (Strictly Controlled)
* **E0:** Baseline 28 features (reproducing $F_{0.5} \approx 0.9595 \pm 0.0035$).
* **E1:** 28 baseline + `indicxlit_name_similarity` (AI4Bharat neural transliteration ratio).
* **E2:** 28 baseline + `qwen_name_cosine` (Qwen3-Embedding-0.6B dense cosine).
* **E3:** 28 baseline + `indicxlit_name_similarity` + `qwen_name_cosine`.

### Non-Negotiable Invariants
1. **Candidate Pairs Frozen:** Candidate generation and candidate pair IDs are identical across all arms.
2. **Evaluation Protocol Frozen:** Identical 5-fold CV splits (`cv_folds_source1.tsv`), identical LightGBM parameters (`lr=0.08`, `leaves=31`, `depth=6`, `n_estimators=100`), identical threshold search, and identical singleton gating.
3. **No Per-Pair Inference:** Neural models are evaluated **once per unique entity/string**, cached to disk, and looked up via $O(1)$ operations during pairwise feature construction.

---

## 2. Directory Layout

```text
experiments/neural_text/
├── README.md                           # This execution manual
├── requirements.txt                    # Pinned Kaggle dependencies
├── config.yaml                         # Experiment configuration and hyperparameters
├── caches/                             # Persistent cache storage
│   ├── indicxlit_translit_cache_5k.parquet
│   ├── qwen_name_embeddings_5k.npz
│   ├── qwen_metadata_5k.json
│   └── augmented_features_5k.parquet
├── reports/                            # Generated experiment reports
│   └── neural_experiment_e0_e3_report.md
├── results/                            # Machine-readable metric JSONs
│   └── experiment_e0_e3_metrics.json
└── scripts/
    ├── detect_script.py                # Deterministic Unicode script/language classifier
    ├── generate_indicxlit_cache.py     # Batch IndicXlit transliterator
    ├── generate_qwen_cache.py          # Batch Qwen3-Embedding encoder
    ├── build_pairwise_features.py      # Pairwise feature matrix constructor
    ├── run_experiments.py              # Frozen 5-fold CV runner
    └── run_all.py                      # Master CLI orchestrator with dry-run protection
```

---

## 3. Kaggle Setup Instructions

### 3.1 Kaggle Environment Configuration
1. **Accelerator:** GPU (T4 $\times 2$ or P100) is strongly recommended for Qwen embedding generation (takes ~8 minutes on T4 vs ~4.6 hours on CPU).
2. **Internet:** Can be toggled **ON** initially to install dependencies and download models, or models can be mounted as offline Kaggle Datasets.

### 3.2 Required Model Datasets
* **Qwen3-Embedding-0.6B:**
  * If online: Pulled automatically from Hugging Face: `Qwen/Qwen3-Embedding-0.6B`.
  * If offline: Attach Kaggle Dataset containing model weights (e.g. `/kaggle/input/qwen3-embedding-06b`).
* **AI4Bharat IndicXlit:**
  * Installed via `pip install ai4bharat-transliteration`.
  * Model weights auto-download on first use or can be pointed to a local directory via `--model-dir`.

### 3.3 Dependency Installation
In the Kaggle notebook cell or terminal:
```bash
pip install -r experiments/neural_text/requirements.txt
```

---

## 4. Execution Workflow

### Step 0: Verification / Dry-Run (Fast & Safe)
Verify script wiring without triggering heavy inference:
```bash
python experiments/neural_text/scripts/run_all.py --scale 5000
```
*(By default, `run_all.py` runs with `--dry-run` and prints entity counts.)*

---

### Step 1: Generate IndicXlit Transliteration Cache
Precomputes Latin phonetic names for all entities with Indic script (~17K entities in 5K split):
```bash
python experiments/neural_text/scripts/generate_indicxlit_cache.py \
    --scale 5000 \
    --beam-width 4
```
* **Output:** `experiments/neural_text/caches/indicxlit_translit_cache_5k.parquet`
* **Runtime:** ~3 minutes on CPU.
* **Resume:** If interrupted, re-running automatically resumes from the last completed entity.

---

### Step 2: Generate Qwen3-Embedding-0.6B Cache
Encodes all unique business names in the candidate pool into L2-normalized dense vectors:
```bash
python experiments/neural_text/scripts/generate_qwen_cache.py \
    --scale 5000 \
    --device auto \
    --batch-size 64 \
    --embedding-dim 1024
```
* **Output:** `experiments/neural_text/caches/qwen_name_embeddings_5k.npz`
* **Runtime:** ~8–10 minutes on Kaggle T4 GPU (~4.6 hours on CPU).
* **Resume:** Re-running skips all entities already present in the `.npz` archive.

---

### Step 3: Construct Pairwise Neural Features
Performs vectorized dot products and string lookups to attach the new features to the frozen 28 baseline features:
```bash
python experiments/neural_text/scripts/build_pairwise_features.py \
    --scale 5000
```
* **Output:** `experiments/neural_text/caches/augmented_features_5k.parquet`
* **Runtime:** ~15 seconds.
* **Integrity Assertions:** Verifies row count (213,023), alignment, non-null values, and exact preservation of the baseline 28 columns.

---

### Step 4: Run E0–E3 5-Fold Cross-Validation
Trains and evaluates LightGBM across all 4 experimental arms on identical folds:
```bash
python experiments/neural_text/scripts/run_experiments.py \
    --augmented-features experiments/neural_text/caches/augmented_features_5k.parquet
```
* **Outputs:**
  * Metrics JSON: `experiments/neural_text/results/experiment_e0_e3_metrics.json`
  * Markdown Report: `experiments/neural_text/reports/neural_experiment_e0_e3_report.md`
* **Runtime:** ~30–45 seconds.

---

### Alternative: Single-Command End-to-End Execution
To run all stages sequentially on Kaggle:
```bash
python experiments/neural_text/scripts/run_all.py \
    --scale 5000 \
    --device auto \
    --run-experiment
```

---

## 5. Promotion Decision Gate

To consider any neural feature arm successful:
1. **Macro $F_{0.5}$ Improvement:** $\Delta F_{0.5} \ge +0.0020$ over E0 ($0.9595 \to \ge 0.9615$).
2. **Singleton Stability:** Singleton accuracy must not degrade by $> 0.5\%$ ($97.35\% \to \ge 96.85\%$).
3. **Fold Consistency:** The improvement must be positive in at least 4 out of 5 folds.
4. **False Merge Control:** New false merges must not exceed recovered classification false negatives.
