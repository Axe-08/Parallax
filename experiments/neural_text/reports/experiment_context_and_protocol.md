# Project Parallax: Neural Text Representation Track (E0–E3)
**Comprehensive Context, Scientific Objectives & Operational Handover Document**

---

## 1. Executive Summary & Competition Mission

**Parallax** is a high-precision, large-scale Business Entity Resolution system designed for the Amazon ML Challenge. The objective is to identify and resolve identical business entities across three heterogeneous data sources ($S_1$, $S_2$, $S_3$) within an unindexed, real-world universe exceeding **1,000,000 target entities**.

### The Evaluation Metric: Macro $F_{0.5}$
The governing metric of the benchmark is **Macro $F_{0.5}$** ($\beta = 0.5$):

$$F_{0.5} = (1 + 0.5^2) \frac{\text{Precision} \cdot \text{Recall}}{0.5^2 \cdot \text{Precision} + \text{Recall}} = 1.25 \frac{\text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

#### Core Metric Invariants:
1. **Precision is weighted $2\times$ over Recall:** A single False Positive (merging two distinct businesses) causes severe metric damage compared to a False Negative (missing a true match).
2. **Singleton Handling:** A substantial fraction of $S_1$ query entities have zero matching records in $S_2 \cup S_3$ (singletons). Systems must predict the empty set $\emptyset$ for them. Incorrectly predicting a match for a singleton incurs a double penalty: it degrades precision and collapses **Singleton Accuracy**.

---

## 2. Baseline Status & The 9.87M Distractor Lesson

### The Authoritative 5K Baseline (Arm E0)
All experiments are strictly anchored to the **stratified 5K Golden Split** ($N = 5,000$ $S_1$ query entities evaluated against the full 1,000,000 target pool):

- **Candidate Pairs:** `213,023` total pairs (an average of $42.6$ candidates per query entity).
- **Frozen Feature Vector:** `28` deterministic features encompassing token edit distances (Jaro-Winkler, Levenshtein, token-sort ratios), address containment metrics, numeric exact match indicators, and postal code match flags.
- **Authoritative Performance:**
  - $\text{Macro } F_{0.5} = \mathbf{0.9595 \pm 0.0035}$
  - $\text{Singleton Accuracy} = \mathbf{97.35\%}$
  - **Residual Error:** Exactly **1,306 Classification False Negatives (CFNs)** where the true matching entity was present in the 213,023 candidates but scored below the decision threshold $\tau^*$.

### The 9.87M Distractor Inflation Anomaly
During preliminary runs, executing an uncalibrated live blocker over the 1M target universe produced **9,868,449 candidate pairs** ($1,974$ candidates per query).
- **Metric Collapse:** This $46\times$ flood of false distractors degraded Macro $F_{0.5}$ from $0.9595$ down to $0.9238$ and caused an 80 GB RAM explosion.
- **Architectural Resolution:** The neural representation track **strictly freezes the candidate pairs** to the verified 213,023 pairs stored in [`baseline_artifacts/candidate_pairs_sample.parquet`](file:///d:/ML%20Challange/Parallax/baseline_artifacts/candidate_pairs_sample.parquet) and the corresponding baseline features in [`baseline_artifacts/features_sample.parquet`](file:///d:/ML%20Challange/Parallax/baseline_artifacts/features_sample.parquet). Both files are tracked in the repository to guarantee identical candidate universes across all environments.

---

## 3. Scientific Hypotheses & Experimental Arms (E0–E3)

### The Scientific Hypothesis
Lexical and token-based matchers fail on two specific linguistic failure modes that account for the 1,306 baseline classification errors:
1. **Cross-Script Transliteration Gaps:** Hindi/Devanagari, Bengali, Tamil, etc., business names spelled phonetically in Latin script with non-standard transliterations.
2. **Deep Semantic Paraphrasing:** Enterprise acronyms, legal suffixes, trade names vs. parent company names, and non-linear word reorderings.

To evaluate whether learned representations provide non-redundant signal to recover these 1,306 errors, we test four experimental arms using a strict 5-fold cross-validation harness:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PARALLAX EXPERIMENT ARMS                        │
├──────────┬──────────────┬──────────────────────────────────────────────┤
│ Arm      │ Feature Dim  │ Features Included                            │
├──────────┼──────────────┼──────────────────────────────────────────────┤
│ E0       │ 28 features  │ Authoritative frozen baseline                │
│ E1       │ 30 features  │ E0 + IndicXlit transliteration similarity     │
│ E2       │ 29 features  │ E0 + Qwen3 dense semantic cosine             │
│ E3       │ 31 features  │ E0 + IndicXlit + Qwen3 (Full Multi-modal)     │
└──────────┴──────────────┴──────────────────────────────────────────────┘
```

---

### In-Depth Breakdown of Arms

#### Arm E0: The Frozen Control Benchmark
- **Features (28):** All baseline lexical, address, numeric, and postal features.
- **Purpose:** Establishes the authoritative numerical benchmark under identical 5-fold CV splits ($5 \times 1,000$ balanced entities). All evaluation scripts assert exact fold size matching to eliminate fold variance.

#### Arm E1: AI4Bharat IndicXlit Transliteration
- **Mechanism:** Inspects Unicode codepoints across 9 Indic scripts (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam). Generates phonetic Latin transliterations using `indic-transliteration` / `AI4Bharat IndicXlit`.
- **Added Features (2):**
  1. `indicxlit_name_similarity`: Normalized Levenshtein ratio between the $S_1$ soft name and the candidate's transliterated name. Non-Indic candidates are assigned `np.nan`. LightGBM branches natively on `NaN`, treating non-Indic candidates with dedicated split logic without corrupting standard lexical features.
  2. `has_indicxlit_name`: Binary indicator ($1.0$ if the candidate was transliterated from an Indic script, $0.0$ otherwise).
- **Target:** Recovering true matching pairs where one source record uses native Indic script and the matching record uses Latin script.

#### Arm E2: Qwen3-Embedding-0.6B Dense Semantics
- **Mechanism:** High-capacity contextual encoder (`Qwen/Qwen3-Embedding-0.6B`) precomputing 1024-dimensional dense vectors.
- **Added Features (1):**
  1. `qwen_name_cosine`: Exact dot product between the $L_2$-normalized entity embeddings of the $S_1$ query and candidate record.
- **Target:** Resolving synonymy, trade-name variations, and multi-word semantic equivalence that token-based edit distances miss.

#### Arm E3: Combined Multi-Modal Representation
- **Features (31):** All 28 baseline features + `indicxlit_name_similarity` + `has_indicxlit_name` + `qwen_name_cosine`.
- **Target:** Testing whether phonetic transliteration and dense semantic embeddings provide orthogonal, non-overlapping signals that recover false negatives synergistically without inflating false merges.

---

## 4. Engineering Architecture & Resource Efficiency Guarantees

The neural pipeline was engineered to run seamlessly under constrained GPU memory and finite execution windows without any compromise in numerical precision:

| Engineering Dimension | Implementation | Computational Efficiency | Precision / Accuracy Impact |
|---|---|---|---|
| **Entity-Level Caching** | Encodes unique `entity_id` strings once to disk (`.npz` / `.parquet`). | Does **~25,000** model inferences total instead of **213,023** pair evaluations ($8.5\times$ compute reduction). | Mathematically identical to running inference on-the-fly. Zero degradation. |
| **FP16 Compressed Storage** | Embeddings saved as half-precision float (`float16`). | Cuts embedding storage from 102 MB to 51 MB; fits entirely in VRAM / RAM. | Vectors are upcast to `float32` before computing dot products; error $\le 10^{-7}$, well within machine epsilon. |
| **Chunked Dot Product** | `np.einsum("ij,ij->i")` in 250,000-pair slices. | Transient RAM during cosine calculation stays strictly **$< 100\text{ MB}$**, avoiding OOM spikes. | Exact algebraic dot product; identical to materializing the full 213K matrix. |
| **Candidate Anchoring** | Frozen candidate pool from [`baseline_artifacts/candidate_pairs_sample.parquet`](file:///d:/ML%20Challange/Parallax/baseline_artifacts/candidate_pairs_sample.parquet). | Eliminates uncalibrated live blocking that produces 9.87M distractors and 80 GB RAM blowouts. | Focuses strictly on the exact candidate universe where the baseline achieves Macro $F_{0.5} \approx 0.9595$. |
| **Missing Value Handling** | Non-Indic entities receive `np.nan` + `has_indicxlit_name=0.0`. | Prevents redundant duplicate calculations. | LightGBM handles `NaN` splits natively without corrupting the decision tree. |

---

## 5. Promotion Gate Criteria to 200K Scale

An experimental arm will be accepted and promoted to the full 200,000-scale dataset if and only if it satisfies all three quantitative gates:

1. **Statistically Significant Macro $F_{0.5}$ Gain:**
   $$\Delta \text{Macro } F_{0.5} \ge \mathbf{+0.0020} \quad \text{over E0 baseline}$$
2. **Singleton Stability:**
   $$\text{Singleton Accuracy drop} < \mathbf{0.5\%} \quad (\text{must remain } \ge 96.85\%)$$
3. **Feature Orthogonality:**
   $$\text{Pearson Correlation } \rho(\text{neural\_feature}, \text{soft\_name\_ratio}) < \mathbf{0.85}$$

---

## 6. Step-by-Step GPU Server Execution Protocol

### Step 1: Connect & Clone
```bash
# On your GPU server:
git clone -b research/neural-text-representation https://github.com/Axe-08/Parallax.git
cd Parallax
```

### Step 2: Set Up Python Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r experiments/neural_text/requirements.txt
```

### Step 3: Populate Dataset TSVs
Ensure the raw TSV files are placed in `data/medium_split_200k/`:
- `data/medium_split_200k/train_source1.tsv`
- `data/medium_split_200k/train_source2.tsv`
- `data/medium_split_200k/train_source3.tsv`
- `data/medium_split_200k/train_ground_truth.tsv`
- `data/medium_split_200k/cv_folds_source1.tsv`

*(Sync from AWS S3 if configured):*
```bash
aws s3 sync s3://amazon-ml-challange-2026-parallax/splits/medium_split_200k data/medium_split_200k
```

### Step 4: Execute the End-to-End Experiment
```bash
export PYTHONPATH="src:$PYTHONPATH"

# Run all 4 stages (Indic cache, Qwen cache, pairwise features, E0-E3 evaluation):
python experiments/neural_text/scripts/run_all.py \
    --scale 5000 \
    --device cuda \
    --batch-size 128 \
    --run-experiment
```

### Step 5: Review Generated Report & Metrics
Once complete, inspect the generated report:
```bash
cat experiments/neural_text/reports/neural_experiment_e0_e3_report.md
```
Verify whether Arm E1, E2, or E3 satisfies the promotion gate ($\Delta F_{0.5} \ge +0.0020$).
