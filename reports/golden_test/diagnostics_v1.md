# Parallax V1 Performance & Failure Diagnostics Report

## 1. Executive Metrics Scorecard

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Macro F0.5 Score** | **`0.9887`** | Official Leaderboard Metric |
| Singleton Accuracy | `100.00%` | Score on true singletons (62 total) |
| Non-Singleton F0.5 | `0.9880` | Score on entities with true matches |
| Ground Truth Matches | `3,380` | Total true positive pairs |
| Correctly Resolved | `3,314` | True matches captured by model |
| Total Predicted | `3,338` | Predicted pairs above threshold |

## 2. Failure Category Breakdown

| Error Type | Count | Severity / Impact |
| :--- | :--- | :--- |
| `BLOCKING_FALSE_NEGATIVE` | 11 | High (Limits recall ceiling) |
| `CLASSIFICATION_FALSE_NEGATIVE` | 55 | Moderate (Missed match) |
| `FALSE_MERGE_POSITIVE` | 24 | Critical (Penalized 2x under F0.5) |
| `SINGLETON_VIOLATION` | 0 | Critical (Drops 1.0 entity score to 0.0) |

## 3. Sample Failure Case Studies

### BLOCKING_FALSE_NEGATIVE Examples

**Case #1 [S1: `S1-694138875` ↔ Cand: `S3-675675021`]**
- **S1 Name / Address:** `ST Badminton Limited` | `Kerala, Ernakulam, Thaikavu, Junction, Arakkakadavu, Nkm Arcade`
- **Cand Name / Address:** `ST Limited Services` | `None`
- **Model Score:** `N/A`
- **Root Cause:** Blocking dropout: candidate was never retrieved in candidate_pairs.

**Case #2 [S1: `S1-322655095` ↔ Cand: `S3-474091907`]**
- **S1 Name / Address:** `Silver Global Pvt Ltd` | `No. 2, Old No. 1479/2 Pid No. 58-46-2, 4Th Cross, 4Th T Block, Jayanagar, Bangalore, Karnataka`
- **Cand Name / Address:** `Silver Pvt Ltd Center - [2735036094]` | `Bangalore, KA, No 2`
- **Model Score:** `N/A`
- **Root Cause:** Blocking dropout: candidate was never retrieved in candidate_pairs.

**Case #3 [S1: `S1-322655095` ↔ Cand: `S3-431613812`]**
- **S1 Name / Address:** `Silver Global Pvt Ltd` | `No. 2, Old No. 1479/2 Pid No. 58-46-2, 4Th Cross, 4Th T Block, Jayanagar, Bangalore, Karnataka`
- **Cand Name / Address:** `ಸಿಲ್ವರ್ ಗ್ಲೋಬಲ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` | `H.no 2, Bangalore, KA`
- **Model Score:** `N/A`
- **Root Cause:** Blocking dropout: candidate was never retrieved in candidate_pairs.

### FALSE_MERGE_POSITIVE Examples

**Case #1 [S1: `S1-846787006` ↔ Cand: `S2-336401529`]**
- **S1 Name / Address:** `Kolkata Destinations Private Limited` | `Kolkata, 109/40B, West Bengal, Howrah, Hazra Road`
- **Cand Name / Address:** `Kolkata Design Private [Ltd]` | `132 M G ROAD, KOLKATA, West Bengal`
- **Model Score:** `0.818`
- **Root Cause:** False merge between distinct businesses sharing name/address similarity.

**Case #2 [S1: `S1-846787006` ↔ Cand: `S2-601745205`]**
- **S1 Name / Address:** `Kolkata Destinations Private Limited` | `Kolkata, 109/40B, West Bengal, Howrah, Hazra Road`
- **Cand Name / Address:** `Kolkata Delsgign Private Limited` | `#132 M G ROAD, KOLKATA, West Bengal`
- **Model Score:** `0.872`
- **Root Cause:** False merge between distinct businesses sharing name/address similarity.

**Case #3 [S1: `S1-55241030` ↔ Cand: `S3-995287325`]**
- **S1 Name / Address:** `Vijay Global Private Limited` | `Banaikala, Joda, Kendujhar, Orissa`
- **Cand Name / Address:** `Vijay Global Pvt  Ltd` | `None`
- **Model Score:** `0.868`
- **Root Cause:** False merge between distinct businesses sharing name/address similarity.

