# Parallax V1 Performance & Failure Diagnostics Report

## 1. Executive Metrics Scorecard

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Macro F0.5 Score** | **`0.9543`** | Official Leaderboard Metric |
| Singleton Accuracy | `98.10%` | Score on true singletons (11170 total) |
| Non-Singleton F0.5 | `0.9528` | Score on entities with true matches |
| Ground Truth Matches | `692,193` | Total true positive pairs |
| Correctly Resolved | `626,794` | True matches captured by model |
| Total Predicted | `633,352` | Predicted pairs above threshold |

## 2. Failure Category Breakdown

| Error Type | Count | Severity / Impact |
| :--- | :--- | :--- |
| `BLOCKING_FALSE_NEGATIVE` | 42536 | High (Limits recall ceiling) |
| `CLASSIFICATION_FALSE_NEGATIVE` | 22863 | Moderate (Missed match) |
| `FALSE_MERGE_POSITIVE` | 6310 | Critical (Penalized 2x under F0.5) |
| `SINGLETON_VIOLATION` | 248 | Critical (Drops 1.0 entity score to 0.0) |

## 3. Sample Failure Case Studies

### BLOCKING_FALSE_NEGATIVE Examples

**Case #1 [S1: `S1-879276752` ↔ Cand: `S3-144339303`]**
- **S1 Name / Address:** `Bryan Square LLC` | `850 Gorman Road, Gatesville, TX`
- **Cand Name / Address:** `bryansquare.com` | `Gatesville, Texas, Gorman Rd`
- **Model Score:** `N/A`
- **Root Cause:** Blocking dropout: candidate was never retrieved in candidate_pairs.

**Case #2 [S1: `S1-906428444` ↔ Cand: `S3-129054940`]**
- **S1 Name / Address:** `Oncology Medicine Inc` | `711 9th Street, Etowah, TN`
- **Cand Name / Address:** `oncologymedicine.com` | `711 Ninth Street, Etwah, Tennessee`
- **Model Score:** `N/A`
- **Root Cause:** Blocking dropout: candidate was never retrieved in candidate_pairs.

**Case #3 [S1: `S1-485414585` ↔ Cand: `S3-122687142`]**
- **S1 Name / Address:** `Engleman Ford LLC` | `314 Murdock Road, Govans, MD`
- **Cand Name / Address:** `Center Engleman LLC` | `314 Murdock Rd, Baltimoer, Maryland`
- **Model Score:** `N/A`
- **Root Cause:** Blocking dropout: candidate was never retrieved in candidate_pairs.

### FALSE_MERGE_POSITIVE Examples

**Case #1 [S1: `S1-888185705` ↔ Cand: `S3-696885039`]**
- **S1 Name / Address:** `North Infra Private Limited` | `Khasara No-961/1, Ground Floor, Right Portion, Vill- Mahipalpur, New Delhi, South West Delhi, Delhi`
- **Cand Name / Address:** `North Infra Pbaet Limited` | `None`
- **Model Score:** `0.866`
- **Root Cause:** False merge between distinct businesses sharing name/address similarity.

**Case #2 [S1: `S1-931804994` ↔ Cand: `S2-17882815`]**
- **S1 Name / Address:** `Ace Engineering` | `509, A Wing, Sanjar Enclave, S.V.Road Kandivali West, Opp. Milap Cinema, Mumbai, Mumbai City, Maharashtra`
- **Cand Name / Address:** `Ace Engineering` | `None`
- **Model Score:** `0.866`
- **Root Cause:** False merge between distinct businesses sharing name/address similarity.

**Case #3 [S1: `S1-891104891` ↔ Cand: `S2-735512023`]**
- **S1 Name / Address:** `Dandekar Vidyalaya` | `Delhi, Shahdara, Office No 5 Central Marke, Block B Dilshad Garden, Shahdara`
- **Cand Name / Address:** `Dandekar Vidyalaya Industries` | `दिल्ली, OFFICE NO 9 CENTRAL MARKE, BLOCK B DILSHAD GARDEN, SHAHDARA`
- **Model Score:** `0.956`
- **Root Cause:** False merge between distinct businesses sharing name/address similarity.

### SINGLETON_VIOLATION Examples

**Case #1 [S1: `S1-650695468` ↔ Cand: `S2-112900389`]**
- **S1 Name / Address:** `IJA Agro Private Limited` | `C/O- Basari Mohan Sardar, Uttar Kajir Hat, Bishnupur, South 24 Parganas, West Bengal`
- **Cand Name / Address:** `বিজয় গ্লোবাল প্রাইভেট লিমিটেড` | `C/O BABLUHOSSAIN SARDAR, VILL-SANKIJAHAN, KULTALI, SOUTH 24 PARGANAS, West Bengal`
- **Model Score:** `0.985`
- **Root Cause:** False candidate exceeded decision threshold on a true singleton.

**Case #2 [S1: `S1-361295179` ↔ Cand: `S2-686276090`]**
- **S1 Name / Address:** `Gulf Voya` | `6 Edgewater Road, Agawam, MA`
- **Cand Name / Address:** `The Gulf Voya Inc` | `19 EDGEWATER ROAD, AGAWAM, MA`
- **Model Score:** `0.882`
- **Root Cause:** False candidate exceeded decision threshold on a true singleton.

**Case #3 [S1: `S1-490603633` ↔ Cand: `S3-266782272`]**
- **S1 Name / Address:** `New Developers Private Limited` | `Unit No.210, 2Nd Flr, Hammersmith Industrial Premises, Sitaladevi Temple Road, Mahim, West, Mumbai, Mumbai City, Maharashtra`
- **Cand Name / Address:** `New Developers Private Limited` | `Cambala Hill Road, Hill Road, Mumbai, MH`
- **Model Score:** `0.878`
- **Root Cause:** False candidate exceeded decision threshold on a true singleton.

