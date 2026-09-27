import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to sys path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.hybrid_er.core.validation import (
    validate_candidate_schema,
    validate_source_contract,
    validate_evaluation_population,
    validate_prediction_population,
    validate_gt_contract
)
from experiments.hybrid_er.evaluation.metrics import evaluate_matcher_predictions
from experiments.hybrid_er.core.cv import get_s1_grouped_folds

def test_schema_missing_source():
    df = pd.DataFrame({'s1_id': ['1'], 'cand_id': ['2']})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_candidate_schema(df)

def test_schema_invalid_source():
    df = pd.DataFrame({'s1_id': ['1'], 'source': ['S4'], 'cand_id': ['2']})
    with pytest.raises(ValueError, match="Invalid source found in candidates"):
        validate_candidate_schema(df)

def test_source_s2_with_s3_cand():
    df = pd.DataFrame({'s1_id': ['1'], 'source': ['S2'], 'cand_id': ['s3_1']})
    s2_ids = {'s2_1'}
    s3_ids = {'s3_1'}
    with pytest.raises(ValueError, match="labeled 'S2' whose cand_id is not in S2"):
        validate_source_contract(df, s2_ids, s3_ids)

def test_source_s3_with_s2_cand():
    df = pd.DataFrame({'s1_id': ['1'], 'source': ['S3'], 'cand_id': ['s2_1']})
    s2_ids = {'s2_1'}
    s3_ids = {'s3_1'}
    with pytest.raises(ValueError, match="labeled 'S3' whose cand_id is not in S3"):
        validate_source_contract(df, s2_ids, s3_ids)

def test_schema_duplicate_canonical():
    df = pd.DataFrame({
        's1_id': ['1', '1'],
        'source': ['S2', 'S2'],
        'cand_id': ['2', '2']
    })
    with pytest.raises(ValueError, match="duplicate canonical tuples"):
        validate_candidate_schema(df)

def test_schema_null_empty_fields():
    df = pd.DataFrame({'s1_id': ['1'], 'source': ['S2'], 'cand_id': [None]})
    with pytest.raises(ValueError, match="null values"):
        validate_candidate_schema(df)
        
    df2 = pd.DataFrame({'s1_id': [''], 'source': ['S2'], 'cand_id': ['2']})
    with pytest.raises(ValueError, match="empty strings"):
        validate_candidate_schema(df2)

def test_eval_population_missing():
    df = pd.DataFrame({'s1_id': ['1'], 'source': ['S2'], 'cand_id': ['2']})
    with pytest.raises(ValueError, match="Missing 1 S1 entities"):
        validate_evaluation_population(df, ['1', '2'])

def test_eval_population_extra():
    df = pd.DataFrame({'s1_id': ['1', '2', '3'], 'source': ['S2', 'S2', 'S2'], 'cand_id': ['1', '2', '3']})
    with pytest.raises(ValueError, match="Found 1 extra S1 entities"):
        validate_evaluation_population(df, ['1', '2'])

def test_pred_population_missing():
    preds = {'1': {'s2_1'}}
    with pytest.raises(ValueError, match="Missing 1 S1 entities"):
        validate_prediction_population(preds, ['1', '2'])

def test_pred_population_extra():
    preds = {'1': {'s2_1'}, '2': set(), '3': {'s3_1'}}
    with pytest.raises(ValueError, match="Found 1 extra S1 entities"):
        validate_prediction_population(preds, ['1', '2'])

def test_zero_match_s1_is_valid():
    preds = {'1': {'s2_1'}, '2': set()}
    # Should not raise
    validate_prediction_population(preds, ['1', '2'])

def test_gt_target_outside_s2_s3():
    gt = {'1': {'s4_1'}}
    s2_ids = {'s2_1'}
    s3_ids = {'s3_1'}
    with pytest.raises(ValueError, match="not in S2 or S3 populations"):
        validate_gt_contract(gt, ['1'], s2_ids, s3_ids)

def test_gt_missing_eval_s1():
    gt = {'1': {'s2_1'}}
    s2_ids = {'s2_1'}
    s3_ids = {'s3_1'}
    with pytest.raises(ValueError, match="Missing 1 evaluation S1"):
        validate_gt_contract(gt, ['1', '2'], s2_ids, s3_ids)

def test_evaluate_matcher_predictions_scoping():
    # Full GT has 200k, we evaluate only 2
    full_gt = {
        '1': {'s2_1'},
        '2': set(),
        '3': {'s3_1'} # Extra one
    }
    preds = {
        '1': {'s2_1'},
        '2': set()
    }
    # If the evaluator tries to evaluate '3', it will either crash or the recall/FP will be wrong.
    # Actually, evaluate_matcher_predictions should strip '3' before sending to base evaluator.
    res = evaluate_matcher_predictions(full_gt, preds, ['1', '2'])
    # S1=1: TP=1
    # S1=2: TP=0, TN(implicit)
    assert res.macro_f05 == 1.0 # Perfect match on the scoped 2

def test_cv_intersection_is_empty():
    df = pd.DataFrame({
        's1_id': ['1', '1', '2', '2', '3', '3', '4', '5'],
        'feat': [0]*8
    })
    folds = list(get_s1_grouped_folds(df, n_splits=3))
    assert len(folds) == 3
    for train_idx, val_idx in folds:
        train_s1 = set(df.iloc[train_idx]['s1_id'])
        val_s1 = set(df.iloc[val_idx]['s1_id'])
        assert train_s1.intersection(val_s1) == set()
