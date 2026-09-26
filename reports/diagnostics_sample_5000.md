# Parallax V1 Performance & Failure Diagnostics Report

## 1. Executive Metrics Scorecard

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Macro F0.5 Score** | **`0.9657`** | Official Leaderboard Metric |
| Singleton Accuracy | `97.64%` | Score on true singletons (296 total) |
| Non-Singleton F0.5 | `0.9651` | Score on entities with true matches |
| Ground Truth Matches | `17,278` | Total true positive pairs |
| Correctly Resolved | `16,108` | True matches captured by model |
| Total Predicted | `16,329` | Predicted pairs above threshold |

## 2. Failure Category Breakdown

| Error Type | Count | Severity / Impact |
| :--- | :--- | :--- |
| `BLOCKING_FALSE_NEGATIVE` | 298 | High (Limits recall ceiling) |
| `CLASSIFICATION_FALSE_NEGATIVE` | 872 | Moderate (Missed match) |
| `FALSE_MERGE_POSITIVE` | 214 | Critical (Penalized 2x under F0.5) |
| `SINGLETON_VIOLATION` | 7 | Critical (Drops 1.0 entity score to 0.0) |

## 3. Sample Failure Case Studies

### BLOCKING_FALSE_NEGATIVE Examples

**Case #1 [S1: `S1-795107512` ↔ Cand: `S2-417790576`]**
- **S1 Name / Address:** `Atlantic Charities` | `749 Adams Drive, Unit 3B, Newport News City, VA`
- **Cand Name / Address:** `Atlantic Chmaitens` | `None`
- **Model Score:** `N/A`
- **Root Cause:** Blocking dropout: candidate was never retrieved in candidate_pairs.

**Case #2 [S1: `S1-682507700` ↔ Cand: `S3-145402673`]**
- **S1 Name / Address:** `Ganga Service Private Limited` | `Maharashtra, 102, 1St Floor, A Wing, Sigma Emerald Building, Off Anand Nagar, Vishal Chsl, Santacruz, East, Mumbai, Mumbai City`
- **Cand Name / Address:** `Ganga Service Private Ltd - 3661126495` | `102, Mumbai, Mumbai City, MH`
- **Model Score:** `N/A`
- **Root Cause:** Blocking dropout: candidate was never retrieved in candidate_pairs.

**Case #3 [S1: `S1-574094676` ↔ Cand: `S3-319544818`]**
- **S1 Name / Address:** `Heritage Industries Group` | `Flat No. 04, Ground Floor, Plot No. 93, Ward 9Bd, Near Empire Hotel, Gandhidham, Kachchh, Gujarat`
- **Cand Name / Address:** `Heritage Group Services` | `None`
- **Model Score:** `N/A`
- **Root Cause:** Blocking dropout: candidate was never retrieved in candidate_pairs.

### FALSE_MERGE_POSITIVE Examples

**Case #1 [S1: `S1-795107512` ↔ Cand: `S3-115938956`]**
- **S1 Name / Address:** `Atlantic Charities` | `749 Adams Drive, Unit 3B, Newport News City, VA`
- **Cand Name / Address:** `Atlantic Charities Co` | `None`
- **Model Score:** `0.938`
- **Root Cause:** False merge between distinct businesses sharing name/address similarity.

**Case #2 [S1: `S1-795107512` ↔ Cand: `S2-831604097`]**
- **S1 Name / Address:** `Atlantic Charities` | `749 Adams Drive, Unit 3B, Newport News City, VA`
- **Cand Name / Address:** `Atlantic Chárities` | `None`
- **Model Score:** `0.965`
- **Root Cause:** False merge between distinct businesses sharing name/address similarity.

**Case #3 [S1: `S1-446962179` ↔ Cand: `S2-112485032`]**
- **S1 Name / Address:** `Nova L.L.C.` | `1226 Hwy 86, Calera, AL`
- **Cand Name / Address:** `NOVA L.L.C.` | `None`
- **Model Score:** `0.928`
- **Root Cause:** False merge between distinct businesses sharing name/address similarity.

### SINGLETON_VIOLATION Examples

**Case #1 [S1: `S1-707663250` ↔ Cand: `S2-173497984`]**
- **S1 Name / Address:** `Future Investment Private Limited` | `1405, Uttar Pradesh, Ghaziabad, 14Th Floor, Migsun Homz, Kaushambi`
- **Cand Name / Address:** `गुरु होटल इन्वेस्टमेंट प्राइवेट लिमिटेड` | `L, GHAZIABAD, Uttar Pradesh`
- **Model Score:** `0.977`
- **Root Cause:** False candidate exceeded decision threshold on a true singleton.

**Case #2 [S1: `S1-460214697` ↔ Cand: `S3-555025385`]**
- **S1 Name / Address:** `Urban It Private Limited` | `Silver Symphony37 Church Avenue Santacruz (W), Mumbai, Maharashtra`
- **Cand Name / Address:** `Urban It Private [Limited]` | `110, Mumbai City, Mumbai, महाराष्ट्र`
- **Model Score:** `0.906`
- **Root Cause:** False candidate exceeded decision threshold on a true singleton.

**Case #3 [S1: `S1-725644064` ↔ Cand: `S3-905115029`]**
- **S1 Name / Address:** `Womens Health Care LLC` | `54 Clifton Avenue, Hull, MA`
- **Cand Name / Address:** `Womens Health Care Corp` | `65. Clifton Ave, Hull, Massachusetts`
- **Model Score:** `0.948`
- **Root Cause:** False candidate exceeded decision threshold on a true singleton.

