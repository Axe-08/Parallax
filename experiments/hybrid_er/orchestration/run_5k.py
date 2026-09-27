import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parallax.data.contracts import load_ground_truth_dict
from parallax.metrics.evaluator import evaluate_resolution_predictions

# Internal imports
from experiments.hybrid_er.core.serialization import serialize_full, serialize_name_only, serialize_address_only
from experiments.hybrid_er.retrieval.neural_bge import DenseRetriever, FaissIndexManager
from experiments.hybrid_er.retrieval.structural import run_structural_retrieval
from experiments.hybrid_er.retrieval.relational import build_s2_s3_graph, run_relational_expansion
from experiments.hybrid_er.retrieval.union import merge_candidate_tables
from experiments.hybrid_er.models.reranker import run_reranking_on_candidates
from experiments.hybrid_er.models.meta_blocker import train_meta_blocker, filter_top_m
from experiments.hybrid_er.models.fusion import train_fusion_model, predict_fusion, build_fusion_features
from experiments.hybrid_er.training.hard_negatives import mine_hard_negatives

def load_data():
    print("Loading 5K Data...")
    gt = load_ground_truth_dict("data/medium_split_200k/train_ground_truth.tsv")
    
    # We only have the sample files for 5K in baseline_artifacts
    s1_df = pd.read_parquet('baseline_artifacts/features_sample.parquet')
    eval_s1_ids = list(s1_df['s1_id'].astype(str).unique())
    
    # We load the full targets because retrieval acts on all targets
    s2_df = pd.read_csv('data/medium_split_200k/train_source2.tsv', sep='\t')
    s3_df = pd.read_csv('data/medium_split_200k/train_source3.tsv', sep='\t')
    
    # We also load the actual S1 string representations
    full_s1 = pd.read_csv('data/medium_split_200k/train_source1.tsv', sep='\t')
    s1_eval_df = full_s1[full_s1['entity_id'].astype(str).isin(eval_s1_ids)].copy()
    
    # Load E0 baseline candidate pairs
    e0_cands = pd.read_parquet('baseline_artifacts/candidate_pairs_sample.parquet')
    
    return s1_eval_df, s2_df, s3_df, e0_cands, gt, eval_s1_ids

def select_best_gpu(requested_gpu: str = "auto") -> str:
    import torch
    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU.")
        return "cpu"
        
    num_devices = torch.cuda.device_count()
    if requested_gpu != "auto" and requested_gpu != "cpu":
        gpu_id = int(requested_gpu)
        print(f"Using explicitly specified GPU: cuda:{gpu_id}")
        return f"cuda:{gpu_id}"
        
    if requested_gpu == "cpu":
        return "cpu"
        
    print(f"Inspecting memory across {num_devices} CUDA devices:")
    best_device = 0
    max_free_bytes = -1
    for i in range(num_devices):
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(i)
            free_gb = free_bytes / (1024**3)
            total_gb = total_bytes / (1024**3)
            print(f"  GPU {i}: {free_gb:.2f} GB free / {total_gb:.2f} GB total")
            if free_bytes > max_free_bytes:
                max_free_bytes = free_bytes
                best_device = i
        except Exception as e:
            print(f"  GPU {i}: Error querying memory ({e})")
            
    chosen = f"cuda:{best_device}"
    print(f"--> Automatically selected {chosen} with {max_free_bytes / (1024**3):.2f} GB free VRAM.")
    return chosen

def run_5k_pipeline(gpu: str = "auto", batch_size: int = 128):
    device = select_best_gpu(gpu)
    s1_df, s2_df, s3_df, e0_cands, gt, eval_s1_ids = load_data()
    
    print("1. Running Structural Retrieval (Channel F)...")
    structural_cands = run_structural_retrieval(s1_df, s2_df, s3_df)
    
    print("2. Building Relational Graph (Channel E)...")
    s2_s3_graph = build_s2_s3_graph(s2_df, s3_df)
    print("3. Running Relational Expansion...")
    # Expand E0 and Structural candidates
    e0_formatted = e0_cands.copy()
    e0_formatted['blocker'] = 'e0'
    e0_formatted['rank'] = 1
    e0_formatted['score'] = 1.0 # Pseudo score
    
    # Determine source for E0 candidates ('S2' or 'S3')
    e0_formatted['source'] = e0_formatted['cand_id'].astype(str).apply(
        lambda x: 'S2' if x.startswith('S2') else ('S3' if x.startswith('S3') else 'S2')
    )
    
    union_for_expansion = merge_candidate_tables([e0_formatted, structural_cands])
    # melt it back to list
    to_expand = union_for_expansion[['s1_id', 'source', 'cand_id']].copy()
    relational_cands = run_relational_expansion(to_expand, s2_s3_graph)
    
    print(f"4. Running Dense Neural Retrieval (BGE-M3 on {device})...")
    retriever = DenseRetriever(device=device, batch_size=batch_size)
    
    # Pre-serialize
    s1_texts = [serialize_full(row) for _, row in s1_df.iterrows()]
    s2_texts = [serialize_full(row) for _, row in s2_df.iterrows()]
    s3_texts = [serialize_full(row) for _, row in s3_df.iterrows()]
    
    s1_embs = retriever.encode(s1_texts)
    s2_embs = retriever.encode(s2_texts)
    s3_embs = retriever.encode(s3_texts)
    
    # Build FAISS
    index_s2 = FaissIndexManager(1024, "FlatIP")
    index_s2.add(s2_embs, [str(x) for x in s2_df['entity_id']])
    
    index_s3 = FaissIndexManager(1024, "FlatIP")
    index_s3.add(s3_embs, [str(x) for x in s3_df['entity_id']])
    
    # Search
    d_s2, _, e_s2 = index_s2.search(s1_embs, k=50)
    d_s3, _, e_s3 = index_s3.search(s1_embs, k=50)
    
    neural_cands = []
    for i, s1_row in enumerate(s1_df.itertuples()):
        s1_id = str(s1_row.entity_id)
        
        for rank, (score, cand) in enumerate(zip(d_s2[i], e_s2[i])):
            if cand:
                neural_cands.append({'s1_id': s1_id, 'source': 'S2', 'cand_id': cand, 'blocker': 'bge_dense', 'rank': rank+1, 'score': score})
                
        for rank, (score, cand) in enumerate(zip(d_s3[i], e_s3[i])):
            if cand:
                neural_cands.append({'s1_id': s1_id, 'source': 'S3', 'cand_id': cand, 'blocker': 'bge_dense', 'rank': rank+1, 'score': score})
                
    neural_cands_df = pd.DataFrame(neural_cands)
    
    print("5. Evaluating Ablation Matrix...")
    # E0 is baseline
    e0_eval = evaluate_resolution_predictions(gt, {s: set(e0_cands[e0_cands['s1_id'] == s]['cand_id'].astype(str)) for s in eval_s1_ids})
    print(f"E0 Macro F0.5: {e0_eval.macro_f05:.4f}")
    
    # E0 + BGE
    e0_bge = merge_candidate_tables([e0_formatted, neural_cands_df])
    preds = e0_bge.groupby('s1_id')['cand_id'].apply(lambda x: set(x)).to_dict()
    res = evaluate_resolution_predictions(gt, preds)
    print(f"E0 + BGE Macro F0.5: {res.macro_f05:.4f}")
    
    # Full Union (E0 + BGE + E + F)
    full_union = merge_candidate_tables([e0_formatted, neural_cands_df, structural_cands, relational_cands])
    preds = full_union.groupby('s1_id')['cand_id'].apply(lambda x: set(x)).to_dict()
    res = evaluate_resolution_predictions(gt, preds)
    print(f"E0 + BGE + E + F Macro F0.5 (Candidate Union Recall Bound): {res.macro_f05:.4f}")
    print(f"Candidate Count: {len(full_union)}")
    
    print(f"6. Reranking Full Union (on {device})...")
    scored_union = run_reranking_on_candidates(full_union, s1_df, s2_df, s3_df, serialize_full, device=device)
    
    # Filter by reranker score naive threshold for ablation check
    top_reranked = scored_union[scored_union['reranker_score'] > 0.0]
    preds = top_reranked.groupby('s1_id')['cand_id'].apply(lambda x: set(x)).to_dict()
    res = evaluate_resolution_predictions(gt, preds)
    print(f"+ Reranker (Naive > 0) Macro F0.5: {res.macro_f05:.4f}")
    
    print("7. Meta-Blocker Compression Evaluation...")
    # Train dummy meta-blocker on just E0 and BGE features to see if it preserves recall
    meta_model = train_meta_blocker(scored_union, target_col='found_by_e0') # Just a dummy target for now, real one uses GT
    
    print("Pipeline Execution Complete. (This is a skeleton smoke test)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5K Hybrid ER Validation Pipeline")
    parser.add_argument("--gpu", type=str, default="auto", help="GPU index (e.g. '2') or 'auto' to pick card with most free VRAM")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for embedding")
    args = parser.parse_args()
    
    run_5k_pipeline(gpu=args.gpu, batch_size=args.batch_size)
