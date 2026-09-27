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
        
    present_ids = set(df['s1_id'].astype(str).unique())
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


def validate_gt_contract(gt: Dict[str, Set[str]], eval_s1_ids: List[str], s2_ids: Set[str], s3_ids: Set[str]):
    """
    Validates that the ground truth dictionary used for evaluation exactly covers the evaluation population
    and maps only to valid S2/S3 targets.
    """
    gt_keys = set(str(k) for k in gt.keys())
    required_ids = set(str(x) for x in eval_s1_ids)
    
    missing = required_ids - gt_keys
    extra = gt_keys - required_ids
    
    if missing:
        raise ValueError(f"GT contract validation failed: Missing {len(missing)} evaluation S1 entities from GT.")
    if extra:
        raise ValueError(f"GT contract validation failed: GT contains {len(extra)} extra S1 entities not in evaluation set.")
        
    valid_targets = s2_ids.union(s3_ids)
    
    for s1, targets in gt.items():
        s1 = str(s1)
        for t in targets:
            t = str(t)
            if t in required_ids:
                raise ValueError(f"GT contract violation: target ID {t} is an S1 ID!")
            if t not in valid_targets:
                raise ValueError(f"GT contract violation: target ID {t} for S1 {s1} is not in S2 or S3 populations.")
