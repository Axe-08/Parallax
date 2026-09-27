import pandas as pd
from typing import List, Set

def validate_candidate_schema(df: pd.DataFrame):
    """
    Asserts that the candidate DataFrame contains the mandatory canonical identity columns.
    """
    required_cols = {'s1_id', 'source', 'cand_id'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Candidate DataFrame is missing required columns: {missing}")

def validate_source_contract(df: pd.DataFrame, s2_ids: Set[str], s3_ids: Set[str]):
    """
    Asserts that all 'S2' sources map to valid S2 IDs, and 'S3' to valid S3 IDs.
    """
    if df.empty:
        return
        
    s2_mask = df['source'] == 'S2'
    s3_mask = df['source'] == 'S3'
    
    invalid_s2 = df[s2_mask & ~df['cand_id'].astype(str).isin(s2_ids)]
    if not invalid_s2.empty:
        raise ValueError(f"Found {len(invalid_s2)} candidates labeled 'S2' whose cand_id is not in s2_df")
        
    invalid_s3 = df[s3_mask & ~df['cand_id'].astype(str).isin(s3_ids)]
    if not invalid_s3.empty:
        raise ValueError(f"Found {len(invalid_s3)} candidates labeled 'S3' whose cand_id is not in s3_df")

def validate_evaluation_population(df: pd.DataFrame, eval_s1_ids: List[str]):
    """
    Asserts that the DataFrame covers exactly the required evaluation population,
    and no more/less. Zero-match S1s should be present in final predictions.
    """
    present_ids = set(df['s1_id'].astype(str).unique())
    required_ids = set([str(x) for x in eval_s1_ids])
    
    missing = required_ids - present_ids
    extra = present_ids - required_ids
    
    if missing:
        raise ValueError(f"Evaluation population validation failed: Missing {len(missing)} S1 entities.")
    if extra:
        raise ValueError(f"Evaluation population validation failed: Found {len(extra)} extra S1 entities not in evaluation set.")
