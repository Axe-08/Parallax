import pandas as pd
from typing import Dict, Set, List
from parallax.metrics.evaluator import evaluate_resolution_predictions

class CandidateRecallReport:
    def __init__(self, canonical_positives: int, candidate_positives: int, blocking_fn: int, candidate_recall: float, candidate_count: int, multiplier: float):
        self.canonical_positives = canonical_positives
        self.candidate_positives = candidate_positives
        self.blocking_fn = blocking_fn
        self.candidate_recall = candidate_recall
        self.candidate_count = candidate_count
        self.multiplier = multiplier

    def __repr__(self):
        return (f"Candidate Recall Ceiling / Candidate Coverage\n"
                f"Canonical Positives: {self.canonical_positives}\n"
                f"Candidate Positives: {self.candidate_positives}\n"
                f"Blocking FN:         {self.blocking_fn}\n"
                f"Candidate Recall:    {self.candidate_recall:.2%}\n"
                f"Total Candidates:    {self.candidate_count}\n"
                f"Cand Multiplier:     {self.multiplier:.2f}x (cands / S1)")

def evaluate_candidate_recall(gt: Dict[str, Set[str]], candidate_preds_dict: Dict[str, Set[str]], eval_s1_ids: List[str]) -> CandidateRecallReport:
    """
    Evaluates the candidate-set ceiling (P0.1).
    This is NOT a matcher evaluation. It only measures whether the true positives exist in the candidate pool.
    """
    canonical_positives = 0
    candidate_positives = 0
    
    for s1_id in eval_s1_ids:
        s1_str = str(s1_id)
        true_targets = gt.get(s1_str, set())
        canonical_positives += len(true_targets)
        
        cand_targets = candidate_preds_dict.get(s1_str, set())
        candidate_positives += len(true_targets.intersection(cand_targets))
        
    blocking_fn = canonical_positives - candidate_positives
    recall = candidate_positives / canonical_positives if canonical_positives > 0 else 1.0
    
    total_cands = sum(len(v) for v in candidate_preds_dict.values())
    multiplier = total_cands / len(eval_s1_ids) if eval_s1_ids else 0.0
    
    return CandidateRecallReport(
        canonical_positives=canonical_positives,
        candidate_positives=candidate_positives,
        blocking_fn=blocking_fn,
        candidate_recall=recall,
        candidate_count=total_cands,
        multiplier=multiplier
    )

def evaluate_matcher_predictions(gt: Dict[str, Set[str]], preds_dict: Dict[str, Set[str]], eval_s1_ids: List[str]):
    """
    Evaluates the final matcher predictions.
    Delegates to the trusted Parallax evaluator to calculate TP, FP, CFN, BFN, and Macro F0.5.
    Ensures that empty predictions for zero-match S1s are properly represented.
    """
    from experiments.hybrid_er.core.validation import validate_prediction_population
    
    # 1. Construct evaluation-scoped GT (P0-A)
    gt_eval = {}
    for s1 in eval_s1_ids:
        s1_str = str(s1)
        if s1_str not in gt:
            raise ValueError(f"Evaluation S1 ID {s1_str} missing from ground truth.")
        gt_eval[s1_str] = set(str(x) for x in gt[s1_str])
        
    if set(gt_eval.keys()) != set([str(x) for x in eval_s1_ids]):
        raise ValueError("gt_eval keys do not exactly match eval_s1_ids.")

    # 2. Ensure every S1 in eval_s1_ids has an entry, even if empty
    safe_preds = {str(s1): set() for s1 in eval_s1_ids}
    for k, v in preds_dict.items():
        if str(k) not in safe_preds:
            raise ValueError(f"Prediction contains extra S1 ID {k} not in evaluation population.")
        safe_preds[str(k)] = set(str(x) for x in v)
        
    validate_prediction_population(safe_preds, eval_s1_ids)
            
    return evaluate_resolution_predictions(gt_eval, safe_preds)
