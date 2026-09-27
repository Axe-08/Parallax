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
    if initial_candidates_df.empty:
        return pd.DataFrame()
        
    expanded = []
    
    for _, row in initial_candidates_df.iterrows():
        s1 = str(row['s1_id'])
        source = str(row['source'])
        tgt = str(row['cand_id'])
        
        node_key = f"{source}:{tgt}"
        
        if node_key in adj_graph:
            for exp_source, exp_tgt in adj_graph[node_key]:
                expanded.append({
                    's1_id': s1,
                    'source': exp_source,
                    'cand_id': exp_tgt,
                    'blocker': 'channel_e_graph',
                    'rank': 1,
                    'score': 1.0 # Edge weight
                })
                
    return pd.DataFrame(expanded) if expanded else pd.DataFrame(columns=['s1_id', 'source', 'cand_id', 'blocker', 'rank', 'score'])
