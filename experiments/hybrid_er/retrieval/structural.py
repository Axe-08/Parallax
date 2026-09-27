import pandas as pd
from typing import Tuple

def extract_structural_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts structural keys from a dataframe (S1, S2, or S3).
    Since domain/handle/email are not in the schema, we rely on numeric blocks
    and rare tokens from business names if available, but primarily exact/near-exact names.
    """
    # For now, let's create a normalized exact name key
    keys_df = df.copy()
    
    # Exact normalized name key
    keys_df['key_exact_name'] = keys_df['business_name'].fillna('').astype(str).str.lower().str.replace(r'[\r\n\t\s]+', ' ', regex=True).str.strip()
    
    # Exact normalized address key
    keys_df['key_exact_addr'] = keys_df['business_address'].fillna('').astype(str).str.lower().str.replace(r'[\r\n\t\s]+', ' ', regex=True).str.strip()
    
    return keys_df

def run_structural_retrieval(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds candidates using exact structural keys.
    """
    s1_keys = extract_structural_keys(s1_df)
    s2_keys = extract_structural_keys(s2_df)
    s3_keys = extract_structural_keys(s3_df)
    
    candidates = []
    
    # Match on Exact Name (ignoring empty strings)
    for tgt_df, source_name in [(s2_keys, 'S2'), (s3_keys, 'S3')]:
        valid_tgt = tgt_df[tgt_df['key_exact_name'] != '']
        valid_s1 = s1_keys[s1_keys['key_exact_name'] != '']
        
        matches = valid_s1.merge(valid_tgt, on='key_exact_name', suffixes=('_s1', '_tgt'))
        
        for _, row in matches.iterrows():
            candidates.append({
                's1_id': str(row['entity_id_s1']),
                'source': source_name,
                'cand_id': str(row['entity_id_tgt']),
                'blocker': 'channel_f_name',
                'rank': 1,
                'score': 1.0
            })
            
    # Match on Exact Address (ignoring empty)
    for tgt_df, source_name in [(s2_keys, 'S2'), (s3_keys, 'S3')]:
        valid_tgt = tgt_df[tgt_df['key_exact_addr'] != '']
        # Also require at least > 10 chars to avoid matching purely on "unknown" or "na"
        valid_tgt = valid_tgt[valid_tgt['key_exact_addr'].str.len() > 10]
        
        valid_s1 = s1_keys[s1_keys['key_exact_addr'] != '']
        valid_s1 = valid_s1[valid_s1['key_exact_addr'].str.len() > 10]
        
        matches = valid_s1.merge(valid_tgt, on='key_exact_addr', suffixes=('_s1', '_tgt'))
        
        for _, row in matches.iterrows():
            candidates.append({
                's1_id': str(row['entity_id_s1']),
                'source': source_name,
                'cand_id': str(row['entity_id_tgt']),
                'blocker': 'channel_f_addr',
                'rank': 1,
                'score': 1.0
            })
            
    return pd.DataFrame(candidates) if candidates else pd.DataFrame(columns=['s1_id', 'source', 'cand_id', 'blocker', 'rank', 'score'])
