# ⚡ Parallax — Amazon ML Challenge 2026 Engine

> **High-throughput, problem-agnostic competitive ML engineering framework designed for the Amazon ML Challenge 2026.**

Built with strict separation of control, deterministic data pipelines, multi-tiered compute orchestration (AWS Spot + S3 + Kaggle + College DGX), and rigorous local cross-validation guardrails.

---

## 🧭 Quick Links

* 📖 **Team Battle Manual:** [TEAM_MANUAL.md](file:///home/akshit/Projects/hackathon/Parallax/TEAM_MANUAL.md) *(Pre-flight checklists, AWS setup, Kaggle fallback, and Hour-by-Hour timeline)*
* 📜 **AI Constitution & Directives:** [GEMINI.md](file:///home/akshit/Projects/hackathon/Parallax/GEMINI.md)
* 🏛️ **Obsidian Vault Hub:** `~/Vault/1-Projects/Parallax/Parallax-Hub.md`

---

## 🛠️ Quickstart

### 1. Local / Remote Setup (1-Click)
```bash
# Clone and enter repo
git clone https://github.com/Axe-08/Parallax.git
cd Parallax

# Bootstrap environment via uv
chmod +x setup_env.sh
./setup_env.sh --gpu   # or --cpu for CPU-only instances
```

### 2. Run the Verification Gate (< 5 seconds)
```bash
make gate
# Runs ruff linter, mypy strict typechecker, and full pytest suite
```

---

## 📦 Project Architecture

```
Parallax/
├── .agents/rules/            # AGY SDK & agent behavioral invariants
├── data/
│   ├── raw/                  # Original competition CSVs (train.csv, test.csv)
│   ├── processed/            # Cleaned parquet data & embeddings
│   ├── golden_split/         # Mini-dataset (5k train, 1k val) for Day-1 shootout
│   └── images/               # Asynchronously fetched catalog images
├── notebooks/                # Exploratory data analysis & Kaggle prototypes
├── src/parallax/
│   ├── config.py             # Pydantic v2 path, S3, and CV split schemas
│   ├── core.py               # Core application entrypoint & health checks
│   ├── data/
│   │   ├── downloader.py     # Asynchronous image downloader with retry backoff
│   │   └── splitter.py       # Stratified/Group K-Fold & Golden Benchmark generator
│   ├── metrics/
│   │   └── evaluator.py      # F1, SMAPE, Exact Match, and diagnostic scoring
│   ├── models/
│   │   ├── base.py           # Abstract BaseModel interface
│   │   ├── rules.py          # Deterministic Regex Rule Engine & unit normalizer
│   │   └── ensemble.py       # Priority fallback, voting, and probability blending
│   ├── observability/
│   │   └── tracer.py         # Structured trace ID, latency, and guardrail spans
│   ├── submission/
│   │   ├── formatter.py      # Automated submission generator
│   │   └── validator.py      # Strict submission integrity verifier (zero NaNs, ID match)
│   └── utils/
│       └── s3_sync.py        # Central AWS S3 push/pull/list sync utility
├── submissions/              # Verified competition submission files
├── tests/                    # Unit tests & golden benchmark test harnesses
├── Makefile                  # Developer workflows (check, format, typecheck, test, gate)
├── pyproject.toml            # uv dependencies, ruff, mypy, and pytest config
├── setup_env.sh              # 1-click cloud/cluster bootstrap script
└── TEAM_MANUAL.md            # Comprehensive operational handbook
```

---

## 🚀 CLI Power Tools

### 1. Golden Benchmark Splitter
Generate a 5,000-row train / 1,000-row validation mini-dataset at Hour 0:
```bash
uv run python -m parallax.data.splitter \
  --input data/raw/train.csv \
  --target-col entity_value \
  --golden-train 5000 \
  --golden-val 1000
```

### 2. Asynchronous Asset Downloader
Download 100k+ catalog images with connection pooling and corruption checks:
```bash
uv run python -m parallax.data.downloader \
  --input data/raw/train.csv \
  --url-col image_link \
  --id-col index \
  --out-dir data/images/ \
  --concurrency 32
```

### 3. Submission Integrity Verifier
Run before uploading any CSV to the official leaderboard:
```bash
uv run python -m parallax.submission.validator \
  submissions/sub_001.csv \
  --sample-sub data/raw/sample_submission.csv
```

### 4. S3 Cloud Data Synchronization
Sync Golden Splits and artifacts between team members across AWS and Kaggle:
```bash
# Push golden split to central team S3
uv run python -m parallax.utils.s3_sync push data/golden_split/

# Pull on another machine / Kaggle
uv run python -m parallax.utils.s3_sync pull golden_split data/golden_split/

# List bucket contents
uv run python -m parallax.utils.s3_sync list
```

### 5. Local Metric Evaluator
Score any model output against ground truth:
```bash
uv run python -m parallax.metrics.evaluator \
  --ground-truth data/golden_split/golden_val.csv \
  --predictions submissions/val_preds.csv \
  --target-col entity_value
```

---

## ⚖️ Engineering Constitution

* **Separation of Control:** Probabilistic models are wrapped with deterministic regex and heuristic formatters.
* **Trust Local CV:** Never overfit to public leaderboard feedback; evaluate every model on the identical Golden Validation set.
* **Deterministic Toolchain:** Always manage dependencies via `uv`. Run `make gate` before pushing any commits.
