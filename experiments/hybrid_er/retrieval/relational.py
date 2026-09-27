import pandas as pd
from typing import Dict, List, Set

def build_s2_s3_graph(s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> Dict[str, List[tuple]]:
    """
    Builds a bipartite graph between S2 and S3 targets based on exact key overlaps.
    Does NOT use ground truth.
    Returns an adjacency list: { "S2:entity_id": [("S3", "entity_id2"), ...], ... }
    """
    adj = {}
    
    # Precompute keys
    s2_keys = s2_df.copy()
    s2_keys['key_name'] = s2_keys['business_name'].fillna('').astype(str).str.lower().str.replace(r'[\r\n\t\s]+', ' ', regex=True).str.strip()
    s2_keys['key_addr'] = s2_keys['business_address'].fillna('').astype(str).str.lower().str.replace(r'[\r\n\t\s]+', ' ', regex=True).str.strip()
    
    s3_keys = s3_df.copy()
    s3_keys['key_name'] = s3_keys['business_name'].fillna('').astype(str).str.lower().str.replace(r'[\r\n\t\s]+', ' ', regex=True).str.strip()
    s3_keys['key_addr'] = s3_keys['business_address'].fillna('').astype(str).str.lower().str.replace(r'[\r\n\t\s]+', ' ', regex=True).str.strip()
    
    # 1. Exact Name match edges
    v_s2_n = s2_keys[s2_keys['key_name'] != '']
    v_s3_n = s3_keys[s3_keys['key_name'] != '']
    matches_n = v_s2_n.merge(v_s3_n, on='key_name', suffixes=('_s2', '_s3'))
    
    for _, row in matches_n.iterrows():
        s2_node = f"S2:{row['entity_id_s2']}"
        s3_node = f"S3:{row['entity_id_s3']}"
        
        if s2_node not in adj: adj[s2_node] = set()
        if s3_node not in adj: adj[s3_node] = set()
        
        adj[s2_node].add(("S3", str(row['entity_id_s3'])))
        adj[s3_node].add(("S2", str(row['entity_id_s2'])))

    # 2. Exact Address match edges (> 10 chars)
    v_s2_a = s2_keys[(s2_keys['key_addr'] != '') & (s2_keys['key_addr'].str.len() > 10)]
    v_s3_a = s3_keys[(s3_keys['key_addr'] != '') & (s3_keys['key_addr'].str.len() > 10)]
    matches_a = v_s2_a.merge(v_s3_a, on='key_addr', suffixes=('_s2', '_s3'))
    
    for _, row in matches_a.iterrows():
        s2_node = f"S2:{row['entity_id_s2']}"
        s3_node = f"S3:{row['entity_id_s3']}"
        
        if s2_node not in adj: adj[s2_node] = set()
        if s3_node not in adj: adj[s3_node] = set()
        
        adj[s2_node].add(("S3", str(row['entity_id_s3'])))
        adj[s3_node].add(("S2", str(row['entity_id_s2'])))
        
    # Convert sets to lists
    return {k: list(v) for k, v in adj.items()}

def run_relational_expansion(initial_candidates_df: pd.DataFrame, adj_graph: Dict[str, List[tuple]]) -> pd.DataFrame:
    """
    Takes an initial set of candidates (e.g. from E0 or BGE) and performs a 1-hop expansion
    using the target-target graph.
    """
    empty = pd.DataFrame(columns=['s1_id', 'source', 'cand_id', 'blocker', 'rank', 'score'])
    if initial_candidates_df.empty:
        return empty

    expanded = []
    for s1, source, tgt in zip(
        initial_candidates_df['s1_id'].astype(str),
        initial_candidates_df['source'].astype(str),
        initial_candidates_df['cand_id'].astype(str),
    ):
        for exp_source, exp_tgt in adj_graph.get(f"{source}:{tgt}", ()):
            expanded.append((s1, exp_source, exp_tgt))

    if not expanded:
        return empty

    # One row per canonical candidate; score = number of seed candidates that reach it.
    df = pd.DataFrame(expanded, columns=['s1_id', 'source', 'cand_id'])
    df = df.groupby(['s1_id', 'source', 'cand_id'], sort=False).size().rename('score').reset_index()
    df['score'] = df['score'].astype(float)
    df['blocker'] = 'channel_e_graph'
    df['rank'] = df.groupby('s1_id')['score'].rank(ascending=False, method='min')
    return df[['s1_id', 'source', 'cand_id', 'blocker', 'rank', 'score']]
