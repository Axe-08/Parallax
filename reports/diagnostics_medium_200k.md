# Parallax V1 Performance & Failure Diagnostics Report

## 1. Executive Metrics Scorecard

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Macro F0.5 Score** | **`0.9738`** | Official Leaderboard Metric |
| Singleton Accuracy | `98.38%` | Score on true singletons (11170 total) |
| Non-Singleton F0.5 | `0.9732` | Score on entities with true matches |
| Ground Truth Matches | `692,193` | Total true positive pairs |
| Correctly Resolved | `654,817` | True matches captured by model |
| Total Predicted | `660,483` | Predicted pairs above threshold |

## 2. Failure Category Breakdown

| Error Type | Count | Severity / Impact |
| :--- | :--- | :--- |
| `BLOCKING_FALSE_NEGATIVE` | 12964 | High (Limits recall ceiling) |
| `CLASSIFICATION_FALSE_NEGATIVE` | 24412 | Moderate (Missed match) |
| `FALSE_MERGE_POSITIVE` | 5437 | Critical (Penalized 2x under F0.5) |
| `SINGLETON_VIOLATION` | 229 | Critical (Drops 1.0 entity score to 0.0) |

## 3. Sample Failure Case Studies

### BLOCKING_FALSE_NEGATIVE Examples

**Case #1 [S1: `S1-888185705` ↔ Cand: `S3-521406491`]**
- **S1 Name / Address:** `North Infra Private Limited` | `Khasara No-961/1, Ground Floor, Right Portion, Vill- Mahipalpur, New Delhi, South West Delhi, Delhi`
- **Cand Name / Address:** `Privmlfate North Infra Limited` | `Khasara No-961/1, New Delhi, Kapasheda, DL`
- **Model Score:** `N/A`
- **Root Cause:** Blocking dropout: candidate was never retrieved in candidate_pairs.

**Case #2 [S1: `S1-849274176` ↔ Cand: `S2-34421487`]**
- **S1 Name / Address:** `Aarvita Biosciences Private Limited` | `No.41, Ponnambalam Salai, K.K.Nagar, Chennai, Tamil Nadu`
- **Cand Name / Address:** `Jaxrizagild` | `41, CHENNAI CITY REGION, தமிழ்நாடு`
- **Model Score:** `N/A`
- **Root Cause:** Blocking dropout: candidate was never retrieved in candidate_pairs.

**Case #3 [S1: `S1-266344604` ↔ Cand: `S3-300641912`]**
- **S1 Name / Address:** `Dream Construction Private Limited` | `4A, 4Th Flr, Ram Krishna Abashan Phase-Ii Dr.B.C Roy Sarani, Jyangra, Baguiati, Kolkata, Parganas North, West Bengal`
- **Cand Name / Address:** `ড্রিম কনস্ট্রাকশন প্রাইভেট লিমিটেড` | `4A, Kolkata, Parganas North, পশ্চিমবঙ্গ`
- **Model Score:** `N/A`
- **Root Cause:** Blocking dropout: candidate was never retrieved in candidate_pairs.

### FALSE_MERGE_POSITIVE Examples

**Case #1 [S1: `S1-879276752` ↔ Cand: `S2-494125125`]**
- **S1 Name / Address:** `Bryan Square LLC` | `850 Gorman Road, Gatesville, TX`
- **Cand Name / Address:** `Bryan Square` | `None`
- **Model Score:** `0.905`
- **Root Cause:** False merge between distinct businesses sharing name/address similarity.

**Case #2 [S1: `S1-618980917` ↔ Cand: `S3-363655395`]**
- **S1 Name / Address:** `Alpha It` | `Maglam Aangan Villa Mahapura, Sanganer, Jaipur, Rajasthan, 133, Jaipur`
- **Cand Name / Address:** `Pragati  Food` | `133, Jaipur, RJ`
- **Model Score:** `0.897`
- **Root Cause:** False merge between distinct businesses sharing name/address similarity.

**Case #3 [S1: `S1-930982377` ↔ Cand: `S2-332793940`]**
- **S1 Name / Address:** `Fortune Impex Private Limited` | `Shop No 12A 384, Sector 31 Noida, Noida, Ghaziabad, Uttar Pradesh`
- **Cand Name / Address:** `Fortune  Impex Private Limited` | `None`
- **Model Score:** `0.894`
- **Root Cause:** False merge between distinct businesses sharing name/address similarity.

### SINGLETON_VIOLATION Examples

**Case #1 [S1: `S1-325685862` ↔ Cand: `S3-334874481`]**
- **S1 Name / Address:** `Mustang (India) Infradevelopers Clinic` | `Surkasha Vihar Colony Gwalior Road, Agra, Uttar Pradesh`
- **Cand Name / Address:** `Dr Mustang (India) Infradevel0pers Clinic Holdings` | `#31 Surkasha Vihar Colony Gwalior Road, Agra Region, Agra, UP`
- **Model Score:** `0.998`
- **Root Cause:** False candidate exceeded decision threshold on a true singleton.

**Case #2 [S1: `S1-527699181` ↔ Cand: `S2-463773061`]**
- **S1 Name / Address:** `Kight Prime Rio Clinic` | `1553 River Birch Run, Unit 110, Chesapeake City, VA`
- **Cand Name / Address:** `CARTWRIGHT PRIME RIO CLINIC INC` | `1553  RIVER BIRCH RUN, CHESAPEAKE CITY, VA`
- **Model Score:** `0.997`
- **Root Cause:** False candidate exceeded decision threshold on a true singleton.

**Case #3 [S1: `S1-987515496` ↔ Cand: `S2-291057871`]**
- **S1 Name / Address:** `Precision Apparel` | `2 Naquag Street, Rutland, MA`
- **Cand Name / Address:** `Precision Co Apparel` | `MA, RUTLAND, 6 NAQUAG ST`
- **Model Score:** `0.931`
- **Root Cause:** False candidate exceeded decision threshold on a true singleton.

