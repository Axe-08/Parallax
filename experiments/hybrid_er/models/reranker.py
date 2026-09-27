import torch
import numpy as np
import pandas as pd
from typing import List
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = None, batch_size: int = 128):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, max_length=256, device=self.device)

    def score_pairs(self, pairs: List[List[str]], show_progress_bar: bool = True) -> np.ndarray:
        """
        Takes a list of [query, document] pairs and returns semantic scores.
        """
        if not pairs:
            return np.array([])
            
        is_cuda = "cuda" in str(self.device)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda" if is_cuda else "cpu", dtype=torch.float16 if is_cuda else torch.float32, enabled=is_cuda):
                scores = self.model.predict(
                    pairs, 
                    batch_size=self.batch_size, 
                    show_progress_bar=show_progress_bar
                )
        if is_cuda:
            torch.cuda.empty_cache()
        return scores

def run_reranking_on_candidates(candidate_df: pd.DataFrame, s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame, serializer_fn, device: str = None) -> pd.DataFrame:
    """
    Given a candidate union table, generate string pairs and run the cross-encoder.
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Reranker")
    
    # Pre-serialize
    s1_dict = {str(row['entity_id']): serializer_fn(row) for _, row in s1_df.iterrows()}
    s2_dict = {str(row['entity_id']): serializer_fn(row) for _, row in s2_df.iterrows()}
    s3_dict = {str(row['entity_id']): serializer_fn(row) for _, row in s3_df.iterrows()}
    
    tgt_dicts = {'S2': s2_dict, 'S3': s3_dict}
    
    pairs = []
    valid_indices = []
    
    for idx, row in candidate_df.iterrows():
        s1 = str(row['s1_id'])
        source = str(row['source'])
        tgt = str(row['cand_id'])
        
        q_str = s1_dict.get(s1, "")
        t_str = tgt_dicts.get(source, {}).get(tgt, "")
        
        # We only score if we found the strings
        if q_str and t_str:
            pairs.append([q_str, t_str])
            valid_indices.append(idx)
            
    logger.info(f"Reranking {len(pairs)} pairs...")
    
    if not pairs:
        res = candidate_df.copy()
        res['reranker_score'] = 0.0
        return res
        
    reranker = Reranker(device=device)
    scores = reranker.score_pairs(pairs)
    
    res = candidate_df.copy()
    res['reranker_score'] = 0.0
    res.loc[valid_indices, 'reranker_score'] = scores
    
    return res
