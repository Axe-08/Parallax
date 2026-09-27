import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.hybrid_er.core.data import read_records
from experiments.hybrid_er.core.validation import (
    assign_source_by_membership,
    validate_gt_global_contract,
    validate_predictions_subset_of_candidates,
)
from experiments.hybrid_er.evaluation.matcher_harness import (
    deterministic_group_partition,
    label_pairs,
    run_grouped_matcher,
)
from experiments.hybrid_er.evaluation.metrics import evaluate_candidate_recall, evaluate_matcher_predictions
from experiments.hybrid_er.retrieval.relational import run_relational_expansion
from experiments.hybrid_er.retrieval.union import candidates_to_dict, merge_candidate_tables

S2 = {"S2-1", "S2-2", "S2-3"}
S3 = {"S3-1", "S3-2"}


def _chan(rows, blocker="b", rank=1, score=0.5):
    df = pd.DataFrame(rows, columns=["s1_id", "source", "cand_id"])
    df["blocker"] = blocker
    df["rank"] = rank
    df["score"] = score
    return df


# --- union boundary -----------------------------------------------------------------

def test_union_requires_population_ids():
    with pytest.raises(TypeError):
        merge_candidate_tables([_chan([("1", "S2", "S2-1")])])
    with pytest.raises(ValueError, match="requires s2_ids and s3_ids"):
        merge_candidate_tables([_chan([("1", "S2", "S2-1")])], s2_ids=None, s3_ids=None)


def test_union_rejects_mislabeled_source():
    with pytest.raises(ValueError, match="labeled 'S2' whose cand_id is not in S2"):
        merge_candidate_tables([_chan([("1", "S2", "S3-1")])], s2_ids=S2, s3_ids=S3)


def test_union_rejects_invalid_source_value():
    with pytest.raises(ValueError, match="Invalid source"):
        merge_candidate_tables([_chan([("1", "S4", "S2-1")])], s2_ids=S2, s3_ids=S3)


def test_union_rejects_missing_blocker_columns():
    df = pd.DataFrame({"s1_id": ["1"], "source": ["S2"], "cand_id": ["S2-1"]})
    with pytest.raises(ValueError, match="missing required columns"):
        merge_candidate_tables([df], s2_ids=S2, s3_ids=S3)


def test_union_rejects_duplicate_within_blocker():
    df = _chan([("1", "S2", "S2-1"), ("1", "S2", "S2-1")])
    with pytest.raises(ValueError, match="duplicate canonical tuples"):
        merge_candidate_tables([df], s2_ids=S2, s3_ids=S3)


def test_union_rejects_same_blocker_from_two_tables():
    a = _chan([("1", "S2", "S2-1")], blocker="bge")
    b = _chan([("1", "S2", "S2-1")], blocker="bge")
    with pytest.raises(ValueError, match="duplicate canonical tuples"):
        merge_candidate_tables([a, b], s2_ids=S2, s3_ids=S3)


def test_union_preserves_per_view_provenance():
    full = _chan([("1", "S2", "S2-1"), ("1", "S3", "S3-1")], blocker="bge_full", rank=[1, 2], score=[0.9, 0.8])
    name = _chan([("1", "S2", "S2-1")], blocker="bge_name", rank=5, score=0.4)
    e0 = _chan([("1", "S3", "S3-1")], blocker="e0", rank=np.nan, score=np.nan)
    out = merge_candidate_tables([full, name, e0], s2_ids=S2, s3_ids=S3).set_index("cand_id")
    assert len(out) == 2
    assert out.loc["S2-1", "found_by_bge_full"] == 1 and out.loc["S2-1", "found_by_bge_name"] == 1
    assert out.loc["S2-1", "score_bge_name"] == 0.4 and out.loc["S2-1", "score_bge_full"] == 0.9
    # Channel without a score still counts as found; not-found gets the sentinel.
    assert out.loc["S3-1", "found_by_e0"] == 1 and np.isnan(out.loc["S3-1", "score_e0"])
    assert out.loc["S2-1", "found_by_e0"] == 0 and out.loc["S2-1", "rank_e0"] == 9999


def test_relational_expansion_emits_unique_rows_with_support():
    seeds = pd.DataFrame({"s1_id": ["1", "1"], "source": ["S2", "S2"], "cand_id": ["S2-1", "S2-2"]})
    graph = {"S2:S2-1": [("S3", "S3-1")], "S2:S2-2": [("S3", "S3-1"), ("S3", "S3-2")]}
    out = run_relational_expansion(seeds, graph).set_index("cand_id")
    assert out.loc["S3-1", "score"] == 2.0 and out.loc["S3-2", "score"] == 1.0
    merge_candidate_tables([out.reset_index()], s2_ids=S2, s3_ids=S3)  # passes the union contract


# --- source assignment ---------------------------------------------------------------

def test_assign_source_by_membership():
    df = pd.DataFrame({"s1_id": ["1", "1"], "cand_id": ["S2-1", "S3-2"]})
    out = assign_source_by_membership(df, S2, S3)
    assert out["source"].tolist() == ["S2", "S3"]
    with pytest.raises(ValueError, match="neither S2 nor S3"):
        assign_source_by_membership(pd.DataFrame({"cand_id": ["S9-1"]}), S2, S3)
    with pytest.raises(ValueError, match="both S2 and S3"):
        assign_source_by_membership(pd.DataFrame({"cand_id": ["X"]}), {"X"}, {"X"})


# --- global GT contract --------------------------------------------------------------

def test_gt_global_contract_population_checks():
    gt = {"S1-1": {"S2-1"}, "S1-2": set()}
    validate_gt_global_contract(gt, S2, S3, s1_ids={"S1-1", "S1-2"})
    with pytest.raises(ValueError, match="not in the S1 population"):
        validate_gt_global_contract(gt, S2, S3, s1_ids={"S1-1"})
    with pytest.raises(ValueError, match="have no GT row"):
        validate_gt_global_contract(gt, S2, S3, s1_ids={"S1-1", "S1-2", "S1-3"})
    with pytest.raises(ValueError, match="is an S1 ID"):
        validate_gt_global_contract({"S1-1": {"S1-2"}, "S1-2": set()}, S2, S3, s1_ids={"S1-1", "S1-2"})


# --- candidate-level metrics ---------------------------------------------------------

GT = {"a": {"S2-1", "S3-1"}, "b": set(), "c": {"S2-2"}, "z": {"S2-3"}}  # z is outside eval


def test_candidate_recall_counts():
    cands = {"a": {"S2-1", "S2-2", "S3-2"}, "b": {"S3-1"}, "c": set()}
    r = evaluate_candidate_recall(GT, cands, ["a", "b", "c"])
    assert (r.canonical_positives, r.candidate_positives, r.blocking_fn) == (3, 1, 2)
    assert (r.candidate_count, r.max_candidates_per_s1, r.s1_without_candidates) == (4, 3, 1)
    assert r.zero_match_s1_with_candidates == 1
    assert r.multiplier == pytest.approx(4 / 3)


def test_candidate_recall_rejects_extra_and_missing_s1():
    with pytest.raises(ValueError, match="outside the evaluation population"):
        evaluate_candidate_recall(GT, {"z": {"S2-3"}}, ["a"])
    with pytest.raises(ValueError, match="missing from global ground truth"):
        evaluate_candidate_recall(GT, {}, ["a", "nope"])


# --- matcher metrics / taxonomy ------------------------------------------------------

def test_matcher_taxonomy():
    cands = {"a": {"S2-1", "S3-1", "S2-2"}, "b": {"S3-2"}, "c": set()}
    preds = {"a": {"S2-1", "S2-2"}, "b": {"S3-2"}, "c": set()}
    r = evaluate_matcher_predictions(GT, preds, ["a", "b", "c"], candidates=cands, s2_ids=S2, s3_ids=S3)
    assert (r.tp, r.fp) == (1, 2)
    assert r.classification_fn == 1  # S3-1 was a candidate but not predicted
    assert r.blocking_fn == 1  # S2-2 for c never retrieved
    assert r.singleton_violation == 1 and r.zero_match_correct_empty == 0
    # a: P=.5 R=.5 -> .5 ; b: violation -> 0 ; c: empty on non-empty GT -> 0
    assert r.macro_f05 == pytest.approx(0.5 / 3)


def test_matcher_rejects_prediction_outside_candidates_and_invalid_target():
    with pytest.raises(ValueError, match="not in the candidate set"):
        evaluate_matcher_predictions(GT, {"a": {"S2-1"}}, ["a"], candidates={"a": set()})
    with pytest.raises(ValueError, match="not in S2 or S3 populations"):
        evaluate_matcher_predictions(GT, {"a": {"S1-9"}}, ["a"], s2_ids=S2, s3_ids=S3)


def test_subset_validator_accepts_subset():
    validate_predictions_subset_of_candidates({"a": {"x"}, "b": set()}, {"a": {"x", "y"}})


def test_candidates_to_dict_rejects_foreign_s1():
    df = _chan([("q", "S2", "S2-1")])
    with pytest.raises(ValueError, match="outside the evaluation population"):
        candidates_to_dict(df, ["a"])
    assert candidates_to_dict(df, ["q", "r"]) == {"q": {"S2-1"}, "r": set()}


# --- deterministic grouped matcher ---------------------------------------------------

def test_group_partition_is_order_independent_and_disjoint():
    ids = [f"S1-{i}" for i in range(50)]
    p1 = deterministic_group_partition(ids, 3, seed=7)
    p2 = deterministic_group_partition(list(reversed(ids)) + ids[:5], 3, seed=7)
    assert p1 == p2
    assert set().union(*p1) == set(ids)
    assert sum(len(g) for g in p1) == len(ids)


def test_group_partition_independent_of_hash_seed():
    code = (
        "import sys; sys.path[:0] = [r'%s', r'%s'];"
        "from experiments.hybrid_er.evaluation.matcher_harness import deterministic_group_partition as d;"
        "ids = set('S1-%%d' %% i for i in range(200));"
        "print([sorted(g)[:3] for g in d(ids, 3, 1)])"
    ) % (PROJECT_ROOT, PROJECT_ROOT / "src")
    outs = set()
    for h in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=h)
        outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout)
    assert len(outs) == 1


def _synthetic_pairs(n_s1=60, seed=0):
    rng = np.random.default_rng(seed)
    rows, gt = [], {}
    for i in range(n_s1):
        s1 = f"S1-{i}"
        gt[s1] = set()
        for j in range(6):
            cand = f"S2-{i}-{j}"
            pos = (j < 2) and (i % 10 != 0)  # every 10th S1 is zero-match
            if pos:
                gt[s1].add(cand)
            rows.append((s1, cand, rng.normal(2.0 if pos else 0.0, 1.0), rng.normal()))
    df = pd.DataFrame(rows, columns=["s1_id", "cand_id", "f1", "f2"])
    folds = {f"S1-{i}": i % 5 for i in range(n_s1)}
    return df, gt, folds


def test_grouped_matcher_is_deterministic_and_leak_free():
    df, gt, folds = _synthetic_pairs()
    params = {"objective": "binary", "verbose": -1, "n_jobs": 1, "deterministic": True, "min_data_in_leaf": 5}
    r1 = run_grouped_matcher(df, ["f1", "f2"], gt, folds, seed=3, params=params, num_boost_round=20)
    r2 = run_grouped_matcher(df.copy(), ["f1", "f2"], gt, folds, seed=3, params=params, num_boost_round=20)
    assert r1.predictions == r2.predictions and r1.taus == r2.taus
    np.testing.assert_allclose(r1.oof_prob, r2.oof_prob)
    assert set(r1.predictions) == set(gt)
    for f in r1.folds:
        assert f.n_train_s1 + f.n_val_s1 == len(gt)
    # every prediction is within that S1's own candidate rows
    cands = df.groupby("s1_id")["cand_id"].apply(set).to_dict()
    validate_predictions_subset_of_candidates(r1.predictions, cands)
    assert label_pairs(df, gt).sum() == sum(len(v) for v in gt.values())


def test_grouped_matcher_rejects_unassigned_s1():
    df, gt, folds = _synthetic_pairs()
    folds.pop("S1-3")
    with pytest.raises(ValueError, match="no outer fold assignment"):
        run_grouped_matcher(df, ["f1"], gt, folds)


# --- raw record reading --------------------------------------------------------------

def test_read_records_keeps_raw_strings(tmp_path):
    p = tmp_path / "train_source2.tsv"
    p.write_text(
        'entity_id\tbusiness_name\tbusiness_address\tcountry\n'
        'S2-1\tNA\t\tUS\n'
        'S2-2\t"Quoted" Co\t1 Main St\tFrance\n',
        encoding="utf-8",
    )
    df = read_records(p)
    assert df["business_name"].tolist() == ["NA", '"Quoted" Co']
    assert df["business_address"].tolist() == ["", "1 Main St"]
    assert df["country"].tolist() == ["US", "France"]
