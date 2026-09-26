# Parallax V1 Performance & Failure Diagnostics Report

## 1. Executive Metrics Scorecard

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Macro F0.5 Score** | **`0.9326`** | Official Leaderboard Metric |
| Singleton Accuracy | `98.22%` | Score on true singletons (11170 total) |
| Non-Singleton F0.5 | `0.9297` | Score on entities with true matches |
| Ground Truth Matches | `692,193` | Total true positive pairs |
| Correctly Resolved | `601,079` | True matches captured by model |
| Total Predicted | `607,714` | Predicted pairs above threshold |

## 2. Failure Category Breakdown

| Error Type | Count | Severity / Impact |
| :--- | :--- | :--- |
| `BLOCKING_FALSE_NEGATIVE` | 73593 | High (Limits recall ceiling) |
| `CLASSIFICATION_FALSE_NEGATIVE` | 17521 | Moderate (Missed match) |
| `FALSE_MERGE_POSITIVE` | 6409 | Critical (Penalized 2x under F0.5) |
| `SINGLETON_VIOLATION` | 226 | Critical (Drops 1.0 entity score to 0.0) |

## 3. Sample Failure Case Studies

### BLOCKING_FALSE_NEGATIVE Examples

**Case #1 [S1: `S1-274126313` ↔ Cand: `S2-51486805`]**
- **S1 Name / Address:** `Obsidian, LLC` | `3907 Hamilton Road, Deer Park, WA`
- **Cand Name / Address:** `Obsidian, LLC Center` | `3907 HAMILTON RD, DEER PARK CIYT, WA`
- **Model Score:** `N/A`
- **Root Cause:** Blocking dropout: candidate was never retrieved in candidate_pairs.

**Case #2 [S1: `S1-274126313` ↔ Cand: `S3-850112871`]**
- **S1 Name / Address:** `Obsidian, LLC` | `3907 Hamilton Road, Deer Park, WA`
- **Cand Name / Address:** `Korbrixx D.B.A. Obsidian, LLC` | `3907 Hamilton Road, Deer Park, WA`
- **Model Score:** `N/A`
- **Root Cause:** Blocking dropout: candidate was never retrieved in candidate_pairs.

**Case #3 [S1: `S1-72444401` ↔ Cand: `S2-973800896`]**
- **S1 Name / Address:** `New Solutions` | `113/154, 1St Floor, Swaroop Nagar, Swarup Nagar, Kanpur Nagar, Uttar Pradesh`
- **Cand Name / Address:** `न्यू सॉल्यूशंस` | `113/154, SWARUP NAGAR, KANPUR NAGAR, Uttar Pradesh`
- **Model Score:** `N/A`
- **Root Cause:** Blocking dropout: candidate was never retrieved in candidate_pairs.

### FALSE_MERGE_POSITIVE Examples

**Case #1 [S1: `S1-931804994` ↔ Cand: `S2-17882815`]**
- **S1 Name / Address:** `Ace Engineering` | `509, A Wing, Sanjar Enclave, S.V.Road Kandivali West, Opp. Milap Cinema, Mumbai, Mumbai City, Maharashtra`
- **Cand Name / Address:** `Ace Engineering` | `None`
- **Model Score:** `0.911`
- **Root Cause:** False merge between distinct businesses sharing name/address similarity.

**Case #2 [S1: `S1-891104891` ↔ Cand: `S2-735512023`]**
- **S1 Name / Address:** `Dandekar Vidyalaya` | `Delhi, Shahdara, Office No 5 Central Marke, Block B Dilshad Garden, Shahdara`
- **Cand Name / Address:** `Dandekar Vidyalaya Industries` | `दिल्ली, OFFICE NO 9 CENTRAL MARKE, BLOCK B DILSHAD GARDEN, SHAHDARA`
- **Model Score:** `0.949`
- **Root Cause:** False merge between distinct businesses sharing name/address similarity.

**Case #3 [S1: `S1-930982377` ↔ Cand: `S2-332793940`]**
- **S1 Name / Address:** `Fortune Impex Private Limited` | `Shop No 12A 384, Sector 31 Noida, Noida, Ghaziabad, Uttar Pradesh`
- **Cand Name / Address:** `Fortune  Impex Private Limited` | `None`
- **Model Score:** `0.876`
- **Root Cause:** False merge between distinct businesses sharing name/address similarity.

### SINGLETON_VIOLATION Examples

**Case #1 [S1: `S1-490603633` ↔ Cand: `S3-266782272`]**
- **S1 Name / Address:** `New Developers Private Limited` | `Unit No.210, 2Nd Flr, Hammersmith Industrial Premises, Sitaladevi Temple Road, Mahim, West, Mumbai, Mumbai City, Maharashtra`
- **Cand Name / Address:** `New Developers Private Limited` | `Cambala Hill Road, Hill Road, Mumbai, MH`
- **Model Score:** `0.870`
- **Root Cause:** False candidate exceeded decision threshold on a true singleton.

**Case #2 [S1: `S1-325685862` ↔ Cand: `S3-334874481`]**
- **S1 Name / Address:** `Mustang (India) Infradevelopers Clinic` | `Surkasha Vihar Colony Gwalior Road, Agra, Uttar Pradesh`
- **Cand Name / Address:** `Dr Mustang (India) Infradevel0pers Clinic Holdings` | `#31 Surkasha Vihar Colony Gwalior Road, Agra Region, Agra, UP`
- **Model Score:** `0.997`
- **Root Cause:** False candidate exceeded decision threshold on a true singleton.

**Case #3 [S1: `S1-283072454` ↔ Cand: `S3-823484014`]**
- **S1 Name / Address:** `Mumbai Technologies Private Limited` | `Mumbai, Floor-6, Esha Ekta B G Kher Marg, Worli Naka. Worli, 602, Maharashtra, Mumbai`
- **Cand Name / Address:** `mumbai technologies private limited` | `Mumbai City, MH, Mumbai, 4Th Floor`
- **Model Score:** `0.891`
- **Root Cause:** False candidate exceeded decision threshold on a true singleton.

