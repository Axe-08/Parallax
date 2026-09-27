"""
Two strictly separated evaluation levels:

1. Candidate generation (evaluate_candidate_recall): recall ceiling, blocking FNs and
   candidate cost. This is NOT a matcher metric and never reports Macro F0.5.
2. Matcher (evaluate_matcher_predictions): final predictions scored by the trusted
   Parallax evaluator (per-S1 Macro F0.5), plus the project error taxonomy.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from parallax.metrics.evaluator import evaluate_resolution_predictions

from experiments.hybrid_er.core.validation import (
    scope_ground_truth_to_eval,
    validate_prediction_population,
    validate_prediction_targets,
    validate_predictions_subset_of_candidates,
)


@dataclass
class CandidateRecallReport:
    n_s1: int
    canonical_positives: int
    candidate_positives: int
    blocking_fn: int
    candidate_recall: float
    candidate_count: int
    multiplier: float
    max_candidates_per_s1: int
    s1_without_candidates: int
    zero_match_s1_with_candidates: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"Candidate Recall Ceiling / Candidate Coverage (NOT a matcher metric)\n"
            f"S1 evaluated:        {self.n_s1}\n"
            f"Canonical Positives: {self.canonical_positives}\n"
            f"Candidate Positives: {self.candidate_positives}\n"
            f"Blocking FN:         {self.blocking_fn}\n"
            f"Candidate Recall:    {self.candidate_recall:.2%}\n"
            f"Total Candidates:    {self.candidate_count}\n"
            f"Cand Multiplier:     {self.multiplier:.2f}x (cands / S1)\n"
            f"Max Cands per S1:    {self.max_candidates_per_s1}"
        )


@dataclass
class MatcherReport:
    macro_f05: float
    singleton_score: float
    non_singleton_f05: float
    n_s1: int
    n_zero_match_s1: int
    zero_match_correct_empty: int
    canonical_positives: int
    predicted_pairs: int
    tp: int
    fp: int
    precision: float
    recall: float
    # Error taxonomy
    false_merge_positive: int
    singleton_violation: int
    classification_fn: int | None
    blocking_fn: int | None
    candidate_positives: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def __repr__(self) -> str:
        cfn = "n/a" if self.classification_fn is None else self.classification_fn
        bfn = "n/a" if self.blocking_fn is None else self.blocking_fn
        return (
            f"Matcher Evaluation (trusted per-S1 Macro F0.5)\n"
            f"Macro F0.5:          {self.macro_f05:.4f}\n"
            f"S1 evaluated:        {self.n_s1} (zero-match: {self.n_zero_match_s1}, "
            f"correctly empty: {self.zero_match_correct_empty})\n"
            f"TP / FP:             {self.tp} / {self.fp}\n"
            f"Precision / Recall:  {self.precision:.4f} / {self.recall:.4f}\n"
            f"CFN / BFN:           {cfn} / {bfn}\n"
            f"Singleton viol.:     {self.singleton_violation}"
        )


def evaluate_candidate_recall(
    gt: Mapping[str, set[str]],
    candidate_preds_dict: Mapping[str, set[str]],
    eval_s1_ids: Sequence[str],
) -> CandidateRecallReport:
    """
    Evaluates the candidate-set ceiling over exactly `eval_s1_ids`.
    This is NOT a matcher evaluation. It only measures whether the true positives exist in
    the candidate pool, and at what candidate cost.
    """
    gt_eval = scope_ground_truth_to_eval(gt, eval_s1_ids)
    extra = {str(k) for k in candidate_preds_dict.keys()} - set(gt_eval.keys())
    if extra:
        raise ValueError(f"Candidate dict contains {len(extra)} S1 IDs outside the evaluation population.")

    canonical_positives = 0
    candidate_positives = 0
    counts = []
    s1_without = 0
    zero_match_with = 0
    for s1_str, true_targets in gt_eval.items():
        cand_targets = {str(c) for c in candidate_preds_dict.get(s1_str, set())}
        canonical_positives += len(true_targets)
        candidate_positives += len(true_targets & cand_targets)
        counts.append(len(cand_targets))
        if not cand_targets:
            s1_without += 1
        if not true_targets and cand_targets:
            zero_match_with += 1

    total_cands = sum(counts)
    n = len(gt_eval)
    return CandidateRecallReport(
        n_s1=n,
        canonical_positives=canonical_positives,
        candidate_positives=candidate_positives,
        blocking_fn=canonical_positives - candidate_positives,
        candidate_recall=candidate_positives / canonical_positives if canonical_positives > 0 else 1.0,
        candidate_count=total_cands,
        multiplier=total_cands / n if n else 0.0,
        max_candidates_per_s1=max(counts) if counts else 0,
        s1_without_candidates=s1_without,
        zero_match_s1_with_candidates=zero_match_with,
    )


def evaluate_matcher_predictions(
    gt: Mapping[str, set[str]],
    preds_dict: Mapping[str, set[str]],
    eval_s1_ids: Sequence[str],
    candidates: Mapping[str, set[str]] | None = None,
    s2_ids: set[str] | None = None,
    s3_ids: set[str] | None = None,
) -> MatcherReport:
    """
    Evaluates final matcher predictions over exactly `eval_s1_ids`.
    Delegates Macro F0.5 to the trusted Parallax evaluator on evaluation-scoped GT.

    - S1s absent from preds_dict are scored as empty predictions (valid for zero-match S1s).
    - Extra S1s in preds_dict raise.
    - If `candidates` is given, predictions must be ⊆ candidates, and CFN/BFN are split.
    - If `s2_ids`/`s3_ids` are given, every predicted target must be a real S2/S3 entity.
    """
    gt_eval = scope_ground_truth_to_eval(gt, eval_s1_ids)

    safe_preds: dict[str, set[str]] = {s1: set() for s1 in gt_eval}
    for k, v in preds_dict.items():
        if str(k) not in safe_preds:
            raise ValueError(f"Prediction contains extra S1 ID {k} not in evaluation population.")
        safe_preds[str(k)] = {str(x) for x in v}
    validate_prediction_population(safe_preds, eval_s1_ids)

    if s2_ids is not None and s3_ids is not None:
        validate_prediction_targets(safe_preds, s2_ids, s3_ids)
    if candidates is not None:
        validate_predictions_subset_of_candidates(safe_preds, candidates)

    rep = evaluate_resolution_predictions(gt_eval, safe_preds)

    tp = rep.total_correct_pairs
    fp = rep.total_predicted_pairs - tp
    zero_match = [s for s, v in gt_eval.items() if not v]
    violations = sum(1 for s in zero_match if safe_preds[s])

    cand_pos = cfn = bfn = None
    if candidates is not None:
        cand_pos = sum(len(v & {str(c) for c in candidates.get(s, set())}) for s, v in gt_eval.items())
        cfn = cand_pos - tp
        bfn = rep.total_true_pairs - cand_pos

    return MatcherReport(
        macro_f05=rep.macro_f05,
        singleton_score=rep.singleton_score,
        non_singleton_f05=rep.non_singleton_f05,
        n_s1=rep.total_entities,
        n_zero_match_s1=len(zero_match),
        zero_match_correct_empty=len(zero_match) - violations,
        canonical_positives=rep.total_true_pairs,
        predicted_pairs=rep.total_predicted_pairs,
        tp=tp,
        fp=fp,
        precision=tp / rep.total_predicted_pairs if rep.total_predicted_pairs else 0.0,
        recall=tp / rep.total_true_pairs if rep.total_true_pairs else 1.0,
        false_merge_positive=fp,
        singleton_violation=violations,
        classification_fn=cfn,
        blocking_fn=bfn,
        candidate_positives=cand_pos,
    )
