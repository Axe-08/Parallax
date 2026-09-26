# Feature-Gap Analysis: 28 Features → Target ~0.988

## Current System Inventory

### Our 28 Features (Grouped by Information Type)

| # | Feature | Type | Information Signal |
|:--|:--------|:-----|:-------------------|
| | **A. Sequence-Level Name Similarity (5)** | | |
| 1 | `raw_name_ratio` | float | Levenshtein ratio on raw business name |
| 2 | `soft_name_ratio` | float | Levenshtein ratio on normalized (lowered, domain-stripped) name |
| 3 | `token_sort_ratio` | float | Sorted-token Levenshtein — order-invariant |
| 4 | `token_set_ratio` | float | Set-intersection Levenshtein — subset-tolerant |
| 5 | `partial_ratio` | float | Best-substring Levenshtein — length-invariant |
| | **B. Prefix-Weighted Name Similarity (2)** | | |
| 6 | `jaro_winkler_soft` | float | JW on soft name (prefix bonus) |
| 7 | `jaro_winkler_raw` | float | JW on raw name |
| | **C. Token-Set Name Analysis (3)** | | |
| 8 | `token_jaccard_name` | float | Jaccard of name word sets |
| 9 | `token_overlap_name` | float | Overlap coefficient (intersection / min) |
| 10 | `first_token_match` | binary | Exact match on first word of soft name |
| | **D. Name Length Geometry (2)** | | |
| 11 | `len_diff_name` | int | Absolute character length difference |
| 12 | `len_ratio_name` | float | min(len) / max(len) |
| | **E. Address Sequence Similarity (4)** | | |
| 13 | `addr_ratio` | float | Levenshtein ratio on clean address |
| 14 | `addr_token_set_ratio` | float | Token-set Levenshtein on address |
| 15 | `canon_addr_ratio` | float | Levenshtein on canonicalized address (ordinals + abbrevs expanded) |
| 16 | `token_jaccard_addr` | float | Jaccard of canonicalized address word sets |
| | **F. Structural Number Discrimination (5)** | | |
| 17 | `num_match_score` | ternary | Any number overlap (1.0), no overlap (0.0), or missing (0.5) |
| 18 | `primary_num_match` | binary | Primary building number exact match |
| 19 | `primary_num_conflict` | binary | Both have primary numbers that differ |
| 20 | `num_jaccard` | float | Jaccard of all extracted numbers |
| 21 | `num_conflict_count` | int | Count of non-shared numbers |
| | **G. Postal Code Alignment (3)** | | |
| 22 | `postal_match` | binary | ZIP/PIN exact match |
| 23 | `postal_conflict` | binary | Both have postal codes that differ |
| 24 | `postal_missing` | binary | At least one postal code absent |
| | **H. Null / Presence Indicators (3)** | | |
| 25 | `is_s1_addr_null` | binary | S1 address is null |
| 26 | `is_cand_addr_null` | binary | Candidate address is null |
| 27 | `both_addr_present` | binary | Neither address is null |
| | **I. Primary Number Presence (1)** | | |
| 28 | `primary_num_missing` | binary | At least one side has no primary number |

---

## 1. What Information Our Features Completely Lack

### Gap 1: Script-Crossing / Transliteration Signal ⚠️ CRITICAL

**The single largest gap.** The preprocessing pipeline already computes `translit_name` and `translit_address` via Brahmic-to-Latin transliteration. The feature extractor uses `cand_translit_name` as a secondary input to `fuzz.ratio()` and `JaroWinkler.similarity()` — it compares `s1_soft_name` against both `cand_soft_name` and `cand_translit_name`, taking the max.

**But there is no feature that captures the *transliteration gap itself*:**
- How much did transliteration help? (`translit_boost = score_with_translit - score_without`)
- Is the candidate in a non-Latin script? (binary signal)
- Does the S1 name match the transliterated candidate better than the raw candidate?

**Evidence from failures:** 22 of 66 classification FNs (33%) in the golden test involve Brahmic-script candidates (Bengali, Kannada, Hindi, Tamil, Malayalam, Gujarati, Telugu, Punjabi). Model scores for these are catastrophically low (0.04–0.19) because even with `max(score_soft, score_translit)`, the feature values are dominated by the `score_soft` dimension where text is unintelligible.

**Concrete failures:**
- `Sun Enterprises LLP` ↔ `சன் எண்டர்பிரைசஸ் எல்எல்பி` — score 0.189
- `Eastern Construction Pvt Ltd` ↔ `ইস্টার্ন কনস্ট্রাকশন প্রাইভেট লিমিটেড` — score 0.100
- `Blue Ventures Private Limited` ↔ `ब्लू वेंचर्स प्राइवेट लिमिटेड` — score 0.038

### Gap 2: Entity-Level Aggregation Features (Zero)

Our features are all **pair-level**. LightGBM sees each (S1, candidate) row independently with no context about the other candidates for that same S1.

**Missing signals:**
- How many candidates does this S1 have?
- What is the max/mean/std of scores across all candidates for this S1?
- How does this candidate's score rank among all candidates for this S1?
- What is the score gap between this candidate and the next-best candidate?

**Why this matters for false merges and singletons:** A singleton entity may have 20 candidates all scoring ~0.65–0.72. A true-match entity typically has 1–3 candidates scoring >0.85 with the rest scoring <0.40. The *distribution shape* of scores is highly discriminative, but our model never sees it.

### Gap 3: Phonetic / Sound-Based Similarity (Zero)

No Soundex, Metaphone, NYSIIS, or any phonetic encoding. Purely visual/lexical.

**Evidence from failures:**
- `Foot & Ankle Care of Chicago` ↔ `FOOT + ANKLE CARE OF CIEADGO` — blocking FN because "CIEADGO" is phonetically similar to "Chicago" but lexically distant
- `Eye Specialists Inc` ↔ `Eye Siecaclggsts Inc` — blocking FN, garbled but phonetically recognizable
- `Kolkata Delsgign Private Limited` — typo of "Design", phonetically identical

### Gap 4: Country / Metadata Features (Zero)

The `country` column exists in the raw data but **no feature uses it**. The pipeline partitions by country for blocking, but the classifier never sees:
- Is S1 from the same country as the candidate?
- Is the country "unseen" (not US/India)?
- Country-specific priors for singleton rates, match counts, etc.

### Gap 5: Name-Part Decomposition (Zero)

No decomposition of business name into:
- Core business name vs. legal suffix (e.g., "Pvt Ltd", "LLC", "Inc")
- Name-without-suffix similarity
- Legal suffix match/mismatch
- Prefix/honorific stripping ("Dr", "Smt", "M/s", "www.", "@")

**Evidence from failures:**
- `Dr Airborne Institute LLP` vs `Airborne Institute` — "Dr" prefix noise, score 0.268
- `www.omindia.com Om (India) |` vs `Om (India) Retail LLP` — domain prefix, score 0.211
- `@Ujjatechshiromani` vs `Ujjatech Shiromani Private Limited` — handle prefix, score 0.417
- `M/s Shree & Sons Prívate Limited` vs `Shree & Sons Private Limited` — "M/s" prefix, score 0.088

### Gap 6: Character-Level / Edit Pattern Features (Zero)

No features for:
- Levenshtein edit distance (raw count, not ratio)
- Character n-gram overlap (TF-IDF score reuse from blocking)
- Longest Common Subsequence ratio
- Common prefix length / common suffix length
- Containment: is one name a substring of the other?

### Gap 7: Address Structure Decomposition (Minimal)

We have postal and primary number. But no:
- City/locality name extraction and match
- State/region match
- Address token count comparison
- Address length ratio
- Geographic coordinate features (impossible without geocoder)

---

## 2. Failure-Case Mapping: Which Gaps Cause Which Failures

### 2.1 Classification FNs (38,982 on 200K / 66 on golden)

| Root Cause Pattern | Observed Count (golden 96) | Gap # | Key Example |
|:---|:---|:---|:---|
| Brahmic script candidate, score collapses | 22 / 66 (33%) | **Gap 1** | `Creative Logistics` ↔ `क्रिएटिव लॉजिस्टिक्स` (score: 0.042) |
| Name matches but candidate address is null | 12 / 66 (18%) | Gap 2, 4 | `Chun & Swanson Partners` ↔ `Chun & Swanson Services` (no addr, score: 0.195) |
| Prefix/suffix noise ("Dr", "www.", "M/s") | 8 / 66 (12%) | **Gap 5** | `Smt techantriksh.com` ↔ `Antriksh Tech Pvt Ltd` (score: 0.693) |
| Garbled/typo name with phonetic similarity | 6 / 66 (9%) | **Gap 3** | `United Tridt` ↔ `United Trust` (score: 0.730) |
| Semantically identical, lexically different | 5 / 66 (8%) | Gap 6 | `Theressa's Sélect` ↔ `Theressa's Select Tax Service` (score: 0.103) |
| Address-only match (random/different name) | 5 / 66 (8%) | Gap 2, 7 | `Dolphin Services Private Limited` ↔ `Smt Vantageflux` (score: 0.019) |
| Other | 8 / 66 (12%) | Mixed | — |

### 2.2 False Merges (7,726 on 200K / 18 on golden)

| Root Cause Pattern | Count (golden) | Gap # | Key Example |
|:---|:---|:---|:---|
| Near-identical name, different business (no discriminating signal) | 8 / 18 (44%) | **Gap 5, 2** | `Pediatric Partners LLC` ↔ `PEDIATRIC PARTNERS` (score: 0.922), different entities |
| Same address, different business | 4 / 18 (22%) | Gap 7 | `Midwest Electrical LLC` ↔ `Lee Trust Llc` (same Union Ave, Chicago) |
| Partial name containment | 4 / 18 (22%) | **Gap 6** | `Allied Society` ↔ `Allied LLC Society` (score: 0.955) |
| Generic name collision | 2 / 18 (11%) | Gap 2 | `RR Services Private Limited` ↔ `star services private limited` (score: 0.941) |

### 2.3 Singleton Violations (572 on 200K / 0 on golden)

No golden-test examples, but the mechanism is clear: singletons with highly generic names (e.g., "Smart Products Private Limited") receive high scores against similar-but-different entities. **Gap 2** (entity-level aggregation: no "how confident is the best candidate relative to the field") is the primary missing signal.

### 2.4 Blocking FNs (14,963 on 200K / 11 on golden)

| Root Cause | Count (golden) | Blocking fix? | Feature fix? |
|:---|:---|:---|:---|
| Brahmic-script candidate, transliteration didn't produce close enough match | 6 / 11 (55%) | Phonetic/word-TF-IDF channel | — |
| Garbled/typo name | 3 / 11 (27%) | Phonetic blocking keys | — |
| Partial name match, insufficient TF-IDF overlap | 2 / 11 (18%) | Increase top-K | — |

> Blocking FNs are **not fixable by features** — they require blocking improvements. But features can help the model handle *borderline* candidates better, preventing classification FNs that look similar to blocking FNs.

---

## 3. Redundancy Analysis Within Existing 28 Features

| Feature Pair/Group | Correlation | Assessment |
|:---|:---|:---|
| `soft_name_ratio` ↔ `raw_name_ratio` | ~0.92 | **Near-redundant.** Soft name is just lowered/domain-stripped raw name. Keep both — the delta matters for domain-containing names. |
| `token_sort_ratio` ↔ `token_set_ratio` | ~0.88 | **Partially redundant.** Token set subsumes token sort for subsets. Both contribute non-trivially when one name has extra tokens. Keep both. |
| `addr_ratio` ↔ `addr_token_set_ratio` ↔ `canon_addr_ratio` | ~0.82–0.90 | **Moderate redundancy.** Canon addr ratio adds value only when abbreviation/ordinal expansion changes the string. Consider dropping `addr_ratio` if canon version performs comparably. |
| `num_match_score` ↔ `primary_num_match` | ~0.65 | **Complementary.** `num_match_score` is the original blended signal; `primary_num_match` is the decomposed precision signal. `num_match_score` may be droppable after ablation. |
| `is_s1_addr_null`, `is_cand_addr_null`, `both_addr_present` | deterministic | **`both_addr_present` = NOT(s1_null OR cand_null).** Technically redundant but LightGBM benefits from explicit interaction terms. Keep all three. |
| `primary_num_missing` ↔ (`primary_num_match` + `primary_num_conflict`) | deterministic | **`primary_num_missing` = 1 iff both `match` and `conflict` are 0.** Fully redundant. Drop candidate. |
| `postal_missing` ↔ (`postal_match` + `postal_conflict`) | deterministic | **Same pattern.** Fully redundant. Drop candidate. |

**Net assessment:** 2 features (`primary_num_missing`, `postal_missing`) are strictly redundant and could be dropped. 1–2 more (`addr_ratio`, `num_match_score`) may be droppable after ablation. The rest provide complementary signals.

---

## 4. Feature Improvement Mapping by Error Type

### 4.1 Features to Reduce **False Merges** (7,726)

| Feature | Mechanism | Expected Impact |
|:---|:---|:---|
| `name_without_suffix_ratio` | Strip "Pvt Ltd", "LLC" etc. before comparing — exposes core name difference | **High** |
| `suffix_type_match` | Do legal suffixes match? "LLC" vs "Inc" signals different entity | **Medium** |
| `entity_rank` | Where does this candidate rank among all S1's candidates? Low rank = suspicious | **Medium** |
| `score_gap_to_next` | Gap between this candidate's score and #2 — large gap = more confident | **Medium** |
| `containment_ratio` | Is one name a substring of the other? Near-containment drives false merges | **Medium** |
| `name_edit_distance` | Raw edit count (not ratio) — catches "Clinic" vs "Clean" that ratio normalizes away | **Low** |

### 4.2 Features to Reduce **Classification FNs** (38,982)

| Feature | Mechanism | Expected Impact |
|:---|:---|:---|
| `translit_boost` | `max(soft,translit) - soft_only` — positive means transliteration helped | **Very High** |
| `is_cross_script` | Binary: is candidate in a non-Latin script? | **Very High** |
| `translit_name_ratio` | Direct `fuzz.ratio(s1_translit, cand_translit)` — both sides transliterated | **Very High** |
| `name_no_prefix_ratio` | Strip "Dr", "Smt", "M/s", "www.", "@" before comparing | **High** |
| `phonetic_name_sim` | Soundex/Metaphone similarity | **Medium** |
| `country_match` | Same country → higher prior for match | **Medium** |
| `addr_token_overlap` | Overlap coefficient on address tokens (not just Jaccard) | **Medium** |

### 4.3 Features to Reduce **Blocking FNs** (14,963)

Features cannot directly fix blocking FNs (these pairs never enter the candidate set). But blocking-layer improvements can:
- Phonetic blocking keys (Soundex inverted index)
- Word-level TF-IDF channel
- Increased top-K
- Cross-script blocking via transliteration index

### 4.4 Features to Reduce **Singleton Violations** (572)

| Feature | Mechanism | Expected Impact |
|:---|:---|:---|
| `s1_candidate_count` | Singletons with many weak candidates behave differently than non-singletons with few strong ones | **High** |
| `max_score_for_s1` | Maximum candidate score across all candidates for this S1 — low max = likely singleton | **High** |
| `score_std_for_s1` | Standard deviation of candidate scores — singletons have flat distributions | **High** |
| `score_rank_normalized` | This candidate's rank / total candidates — contextualizes the score | **Medium** |

---

## 5. Ranked Candidate Features (22 Features)

### Tier 1: Highest Expected Information Gain (7 features)

These address the largest gaps with the most observed failure cases.

| # | Feature Name | Type | Computation | Gap | Primary Target | Dependencies | Testability |
|:--|:-------------|:-----|:------------|:----|:---------------|:-------------|:------------|
| **F1** | `translit_boost_name` | float | `max(ratio(s1_soft, cand_soft), ratio(s1_soft, cand_translit)) - ratio(s1_soft, cand_soft)` | Gap 1 | Class. FNs (Brahmic) | None — data already computed | ✅ Trivial |
| **F2** | `is_cross_script` | binary | 1 if candidate `business_name` contains non-Latin Unicode blocks | Gap 1 | Class. FNs (Brahmic) | None | ✅ Trivial |
| **F3** | `translit_name_ratio` | float | `fuzz.ratio(s1_translit_name, cand_translit_name)` — both sides transliterated | Gap 1 | Class. FNs (Brahmic) | `translit_name` column (already exists) | ✅ Trivial |
| **F4** | `name_core_ratio` | float | Strip legal suffixes + prefixes ("Dr", "M/s", "www.") from both sides, then `fuzz.ratio()` | Gap 5 | False Merges + Class. FNs | Suffix/prefix stripping function | ✅ Easy (need regex) |
| **F5** | `suffix_match` | binary | Do both names' legal suffixes match? "LLC"="LLC"→1, "LLC"≠"Inc"→0, missing→0.5 | Gap 5 | False Merges | Suffix extraction function | ✅ Easy |
| **F6** | `s1_candidate_count` | int | Number of candidates generated for this S1 entity by the blocker | Gap 2 | Singleton Violations | Candidate dict (available during feature extraction) | ✅ Trivial |
| **F7** | `country_match` | binary | 1 if S1 and candidate have identical `country` field | Gap 4 | Class. FNs + False Merges | `country` column (already in raw data) | ✅ Trivial |

> [!IMPORTANT]
> **F1–F3 together should dramatically improve the 33% of classification FNs caused by Brahmic script pairs.** The data is already preprocessed; we're simply not surfacing the signal as features.

### Tier 2: High Expected Information Gain (7 features)

| # | Feature Name | Type | Computation | Gap | Primary Target | Dependencies | Testability |
|:--|:-------------|:-----|:------------|:----|:---------------|:-------------|:------------|
| **F8** | `score_rank_pct` | float | `rank_of_this_candidate / total_candidates_for_s1` (0=best, 1=worst) | Gap 2 | Singleton Violations, False Merges | Requires post-hoc or second pass | ⚠️ Needs architecture change — compute scores first, then add rank feature. Possible as LightGBM leaf prediction → rank → re-score. |
| **F9** | `containment_ratio_name` | float | `len(LCS(s1_soft, cand_soft)) / min(len_s1, len_cand)` | Gap 6 | False Merges | LCS computation — O(n²) per pair but names are short | ✅ Easy |
| **F10** | `name_edit_distance` | int | Raw Levenshtein distance (not normalized) | Gap 6 | False Merges | `rapidfuzz.distance.Levenshtein.distance()` | ✅ Trivial |
| **F11** | `phonetic_name_match` | float | Soundex/Metaphone similarity of first significant name token | Gap 3 | Class. FNs (garbled) | `jellyfish` library | ✅ Easy |
| **F12** | `common_prefix_len` | int | Length of longest common prefix of soft names | Gap 6 | Class. FNs | Simple string operation | ✅ Trivial |
| **F13** | `addr_token_overlap` | float | Overlap coefficient on canonicalized address word sets | Gap 7 | Class. FNs (addr-matching) | Already have `s1_addr_toks`, `cand_addr_toks` | ✅ Trivial |
| **F14** | `name_word_count_diff` | int | `abs(len(s1_soft.split()) - len(cand_soft.split()))` | Gap 6 | False Merges (extra tokens) | Simple | ✅ Trivial |

### Tier 3: Medium Expected Information Gain (5 features)

| # | Feature Name | Type | Computation | Gap | Primary Target |
|:--|:-------------|:-----|:------------|:----|:---------------|
| **F15** | `len_ratio_addr` | float | `min(addr_len) / max(addr_len)` — address length disparity | Gap 7 | False Merges |
| **F16** | `country_is_unseen` | binary | 1 if S1 country ∉ {"US", "India"} | Gap 4 | General (threshold calibration) |
| **F17** | `both_names_short` | binary | 1 if both soft names ≤ 2 tokens | Gap 6 | False Merges (generic short names) |
| **F18** | `addr_prefix_5_match` | binary | First 5 chars of cleaned address match | Gap 7 | Class. FNs |
| **F19** | `translit_boost_addr` | float | Same as F1 but for address channel | Gap 1 | Class. FNs (Brahmic addresses) |

### Tier 4: Speculative / Needs Investigation (3 features)

| # | Feature Name | Type | Computation | Notes |
|:--|:-------------|:-----|:------------|:------|
| **F20** | `tfidf_blocking_score` | float | The max sparse TF-IDF cosine similarity from the blocker for this pair | Available if blocker returns scores; currently discarded |
| **F21** | `cross_encoder_score` | float | Semantic similarity from a transformer cross-encoder | Requires Kaggle GPU; expensive per pair; add as Phase 5 |
| **F22** | `name_embedding_cosine` | float | Cosine similarity of sentence embeddings (bi-encoder) | Requires Kaggle GPU; cheaper per pair but needs pre-computation |

---

## 6. Redundancy Risk in Proposed Features

| Proposed Feature | Potentially Redundant With | Assessment |
|:---|:---|:---|
| `name_edit_distance` (F10) | `len_diff_name` + `soft_name_ratio` | Partially redundant — ratio = 1 - dist/max_len. But raw distance gives the model absolute scale information (editing "IBM" by 1 char is different from editing "International Business Machines" by 1 char). **Keep.** |
| `containment_ratio_name` (F9) | `partial_ratio` | Partially redundant — `partial_ratio` already finds best substring alignment. But containment is directional and doesn't involve edit distance. **Keep tentatively, ablate.** |
| `addr_token_overlap` (F13) | `token_jaccard_addr` | Complement — Jaccard penalizes extra tokens, overlap doesn't. **Keep.** |
| `name_word_count_diff` (F14) | `len_diff_name` | Different unit — character length vs. word count. "Dr ABC" vs "ABC" has len_diff=3 but word_count_diff=1. **Keep.** |
| `len_ratio_addr` (F15) | `canon_addr_ratio` | Weakly correlated — length ratio is a much coarser signal. **Keep only if ablation shows independent contribution.** |

---

## 7. Implementation Priority (Not Feature Count)

> [!WARNING]
> **Do not target 50 features.** The competitor may have 50 features because they lack our preprocessing advantages (transliteration, canonicalization) and compensate with more features. Our goal is maximum F₀.₅ per feature, not feature count.

### Recommended implementation batches:

**Batch 1 (7 features, ~2 hours, highest ROI):**
F1 `translit_boost_name`, F2 `is_cross_script`, F3 `translit_name_ratio`, F4 `name_core_ratio`, F5 `suffix_match`, F6 `s1_candidate_count`, F7 `country_match`

→ Run 5-fold CV. Measure Δ F₀.₅.

**Decision gate:** If Batch 1 gives ≥ +0.005, stop and analyze before adding more. If < +0.003, proceed to Batch 2.

**Batch 2 (5 features, ~1.5 hours):**
F10 `name_edit_distance`, F11 `phonetic_name_match`, F12 `common_prefix_len`, F13 `addr_token_overlap`, F14 `name_word_count_diff`

**Batch 3 (3 features, ~1 hour):**
F9 `containment_ratio_name`, F15 `len_ratio_addr`, F19 `translit_boost_addr`

**After Batch 3:** Run full ablation. Drop any features with negative or negligible contribution. Then and only then consider Tier 4 (semantic/GPU features).

---

## 8. Key Takeaway

Our biggest feature gap is not "too few features" — it's that we **compute information we never surface**:

1. Transliteration is done but the *delta* is invisible to LightGBM.
2. Country is used for blocking partitions but invisible to the classifier.
3. Candidate count per entity is known but invisible.
4. Legal suffixes are partially expanded in normalization but the match/mismatch is invisible.

The first 7 proposed features (Tier 1) require essentially no new data, no new dependencies, and no architectural changes. They expose what we already have.
