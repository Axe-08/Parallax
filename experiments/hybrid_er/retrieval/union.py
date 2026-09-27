import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def merge_candidate_tables(candidate_dfs: list, e0_features_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Merges multiple candidate dataframes into a single deduplicated table.
    Each input dataframe MUST have:
    - s1_id (str)
    - source (str: 'S2' or 'S3')
    - cand_id (str)
    - blocker (str) - e.g., 'e0', 'bge', 'channel_f'
    - rank (int) - rank within that blocker
    - score (float) - similarity score within that blocker
    
    Returns a unified candidate table with columns:
    s1_id, cand_id, source, 
    found_by_e0, rank_e0, score_e0,
    found_by_bge, rank_bge, score_bge,
    etc.
    """
    if not candidate_dfs:
        return pd.DataFrame()
        
    # Standardize column types before concat
    std_dfs = []
    for df in candidate_dfs:
        if df is None or len(df) == 0:
            continue
        d = df.copy()
        d['s1_id'] = d['s1_id'].astype(str)
        d['cand_id'] = d['cand_id'].astype(str)
        if 'source' not in d.columns:
            raise ValueError(f"Candidate DataFrame is missing required 'source' column. Columns: {list(d.columns)}")
            
        d['source'] = d['source'].astype(str)
        invalid_sources = d[~d['source'].isin(['S2', 'S3'])]
        if not invalid_sources.empty:
            raise ValueError(f"Invalid source found in candidates. Allowed: 'S2', 'S3'. Found: {invalid_sources['source'].unique()}")
            
        if 'blocker' not in d.columns:
            d['blocker'] = 'unknown'
        else:
            d['blocker'] = d['blocker'].astype(str)
        if 'rank' not in d.columns:
            d['rank'] = 1
        if 'score' not in d.columns:
            d['score'] = 1.0
        std_dfs.append(d)
        
    if not std_dfs:
        return pd.DataFrame()
        
    master_df = pd.concat(std_dfs, ignore_index=True)
    
    # We want to pivot this so each blocker is a set of columns
    # If the same blocker found the same candidate twice (e.g. from multiple views),
    # we take the best rank and highest score.
    grouped = master_df.groupby(['s1_id', 'source', 'cand_id', 'blocker']).agg(
        rank=('rank', 'min'),
        score=('score', 'max')
    ).reset_index()
    
    # Pivot
    pivoted = grouped.pivot(
        index=['s1_id', 'source', 'cand_id'],
        columns='blocker',
        values=['rank', 'score']
    )
    
    # Flatten multi-level columns
    pivoted.columns = [f"{col[0]}_{col[1]}" for col in pivoted.columns]
    pivoted = pivoted.reset_index()
    
    # Add found_by_X flags
    blockers = master_df['blocker'].unique()
    for b in blockers:
        pivoted[f'found_by_{b}'] = pivoted[f'rank_{b}'].notna().astype(int)
        # Fill missing ranks with a large number (e.g. 9999) and scores with 0
        pivoted[f'rank_{b}'] = pivoted[f'rank_{b}'].fillna(9999)
        pivoted[f'score_{b}'] = pivoted[f'score_{b}'].fillna(0.0)
        
    # If E0 features are provided, join them
    if e0_features_df is not None:
        e0 = e0_features_df.copy()
        e0['s1_id'] = e0['s1_id'].astype(str)
        e0['cand_id'] = e0['cand_id'].astype(str)
        e0['source'] = e0['source'].astype(str)
        pivoted = pivoted.merge(e0, on=['s1_id', 'source', 'cand_id'], how='left')
        
    return pivoted

def save_candidates(df: pd.DataFrame, filepath: str):
    """
    Saves candidates to chunked parquet format.
    """
    table = pa.Table.from_pandas(df)
    pq.write_table(table, filepath)

def load_candidates(filepath: str) -> pd.DataFrame:
    """
    Loads candidates from parquet.
    """
    table = pq.read_table(filepath)
    return table.to_pandas()
