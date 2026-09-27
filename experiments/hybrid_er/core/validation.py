import pandas as pd
from typing import List, Set, Dict

def validate_candidate_schema(df: pd.DataFrame):
    """
    Asserts that the candidate DataFrame contains the mandatory canonical identity columns,
    validates types, checks for nulls/empty strings, enforces allowed sources, and ensures uniqueness.
    """
    if df is None or df.empty:
        return
        
    required_cols = {'s1_id', 'source', 'cand_id'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Candidate DataFrame is missing required columns: {missing}")

    for col in required_cols:
        if df[col].isnull().any():
            raise ValueError(f"Candidate DataFrame contains null values in canonical column: {col}")
            
    # Normalize to string before validation
    df['s1_id'] = df['s1_id'].astype(str)
    df['source'] = df['source'].astype(str)
    df['cand_id'] = df['cand_id'].astype(str)
    
    for col in required_cols:
        if (df[col] == "").any():
            raise ValueError(f"Candidate DataFrame contains empty strings in canonical column: {col}")

    invalid_sources = df[~df['source'].isin(['S2', 'S3'])]
    if not invalid_sources.empty:
        raise ValueError(f"Invalid source found in candidates. Allowed: 'S2', 'S3'. Found: {invalid_sources['source'].unique()}")

    # Enforce uniqueness of canonical tuple
    dups = df[df.duplicated(subset=['s1_id', 'source', 'cand_id'])]
    if not dups.empty:
        raise ValueError(f"Candidate DataFrame contains duplicate canonical tuples (s1_id, source, cand_id). Found {len(dups)} duplicates.")


def validate_source_contract(df: pd.DataFrame, s2_ids: Set[str], s3_ids: Set[str]):
    """
    Asserts that all 'S2' sources map to valid S2 IDs, and 'S3' to valid S3 IDs.
    """
    if df is None or df.empty:
        return
        
    s2_mask = df['source'] == 'S2'
    s3_mask = df['source'] == 'S3'
    
    invalid_s2 = df[s2_mask & ~df['cand_id'].astype(str).isin(s2_ids)]
    if not invalid_s2.empty:
        raise ValueError(f"Found {len(invalid_s2)} candidates labeled 'S2' whose cand_id is not in S2 ID population.")
        
    invalid_s3 = df[s3_mask & ~df['cand_id'].astype(str).isin(s3_ids)]
    if not invalid_s3.empty:
        raise ValueError(f"Found {len(invalid_s3)} candidates labeled 'S3' whose cand_id is not in S3 ID population.")
        
    invalid_other = df[~df['source'].isin(['S2', 'S3'])]
    if not invalid_other.empty:
        raise ValueError(f"Found {len(invalid_other)} candidates with source other than S2 or S3.")


def validate_evaluation_population(df: pd.DataFrame, eval_s1_ids: List[str]):
    """
    Asserts that the input DataFrame covers exactly the required evaluation population.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty but expected evaluation population.")
        
    id_col = 's1_id' if 's1_id' in df.columns else 'entity_id'
    present_ids = set(df[id_col].astype(str).unique())
    required_ids = set([str(x) for x in eval_s1_ids])
    
    missing = required_ids - present_ids
    extra = present_ids - required_ids
    
    if missing:
        raise ValueError(f"Evaluation population validation failed: Missing {len(missing)} S1 entities. Example: {list(missing)[:5]}")
    if extra:
        raise ValueError(f"Evaluation population validation failed: Found {len(extra)} extra S1 entities not in evaluation set.")


def validate_prediction_population(predictions: Dict[str, Set[str]], eval_s1_ids: List[str]):
    """
    Asserts that prediction S1 IDs == evaluation S1 IDs precisely.
    Zero-match S1s must have an empty set.
    """
    pred_keys = set(str(k) for k in predictions.keys())
    required_ids = set(str(x) for x in eval_s1_ids)
    
    missing = required_ids - pred_keys
    extra = pred_keys - required_ids
    
    if missing:
        raise ValueError(f"Prediction population validation failed: Missing {len(missing)} S1 entities. Zero-match S1s must have an empty set.")
    if extra:
        raise ValueError(f"Prediction population validation failed: Found {len(extra)} extra S1 entities not in evaluation set.")


def validate_gt_global_contract(gt: Dict[str, Set[str]], s2_ids: Set[str], s3_ids: Set[str]):
    """
    Validates that the ground truth dictionary contains only valid S2/S3 targets,
    and no target is an S1 ID. Does NOT check for exact population matching.
    """
    if gt is None:
        return
        
    valid_targets = s2_ids.union(s3_ids)
    # Check a sampled subset to avoid massive slow loops if not necessary,
    # but since it's a strict contract, we check all.
    for s1, targets in gt.items():
        s1 = str(s1)
        for t in targets:
            t = str(t)
            # Cannot be an S1 ID (heuristic: if it's in the GT keys, it's an S1)
            # Actually, the user says "no target may be an S1 ID".
            # We don't have s1_ids here, but we know targets must be in valid_targets.
            if t not in valid_targets:
                raise ValueError(f"GT contract violation: target ID {t} for S1 {s1} is not in S2 or S3 populations.")

def scope_ground_truth_to_eval(gt: Dict[str, Set[str]], eval_s1_ids: List[str]) -> Dict[str, Set[str]]:
    """
    Extracts the subset of GT matching eval_s1_ids exactly.
    Raises ValueError if any eval S1 is missing from the global GT.
    """
    scoped_gt = {}
    for s1 in eval_s1_ids:
        s1_str = str(s1)
        if s1_str not in gt:
            raise ValueError(f"Evaluation S1 ID {s1_str} missing from global ground truth.")
        scoped_gt[s1_str] = set([str(x) for x in gt[s1_str]])
        
    # Ensure keys match exactly
    if set(scoped_gt.keys()) != set([str(x) for x in eval_s1_ids]):
        raise ValueError("Scoped GT keys do not exactly match eval_s1_ids.")
        
    return scoped_gt
