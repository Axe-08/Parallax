import pandas as pd

def mine_hard_negatives(candidate_df: pd.DataFrame, ground_truth: dict, max_neg_per_pos: int = 3) -> pd.DataFrame:
    """
    Given a candidate union dataframe, mines hard negatives for training.
    Hard negatives are those with high retrieval scores/ranks but not in ground truth.
    candidate_df must have a 'score_e0' or 'score_bge' to rank by.
    """
    df = candidate_df.copy()
    df['s1_id'] = df['s1_id'].astype(str)
    df['cand_id'] = df['cand_id'].astype(str)
    
    # Identify true matches
    def is_match(row):
        s1 = row['s1_id']
        c = row['cand_id']
        if s1 in ground_truth:
            return 1 if c in ground_truth[s1] else 0
        return 0
        
    df['is_match'] = df.apply(is_match, axis=1)
    
    positives = df[df['is_match'] == 1]
    negatives = df[df['is_match'] == 0]
    
    # Sort negatives by a combined score if available to get the "hardest" ones
    sort_col = None
    if 'score_e0' in negatives.columns:
        sort_col = 'score_e0'
    elif 'score_bge' in negatives.columns:
        sort_col = 'score_bge'
        
    if sort_col:
        negatives = negatives.sort_values(['s1_id', sort_col], ascending=[True, False])
        
    # Group by S1 and take top max_neg_per_pos
    hard_negatives = negatives.groupby('s1_id').head(max_neg_per_pos)
    
    return pd.concat([positives, hard_negatives], ignore_index=True)
