import lightgbm as lgb
import pandas as pd
import numpy as np

def train_fusion_model(train_df: pd.DataFrame, features: list, target_col: str = 'is_match', params: dict = None) -> lgb.Booster:
    """
    Trains the final fusion model.
    """
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'verbose': -1,
            'random_state': 42
        }
        
    dtrain = lgb.Dataset(train_df[features], label=train_df[target_col])
    model = lgb.train(params, dtrain, num_boost_round=100)
    return model

def predict_fusion(model: lgb.Booster, test_df: pd.DataFrame, features: list) -> np.ndarray:
    return model.predict(test_df[features])

def build_fusion_features(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs the final feature vector.
    Fills NaNs for provenance columns.
    """
    df = candidate_df.copy()
    
    # Base feature list
    features = []
    
    # 1. Reranker
    if 'reranker_score' in df.columns:
        features.append('reranker_score')
        
    # 2. E0 probability
    if 'xgb_prob' in df.columns:
        features.append('xgb_prob')
        
    # 3. Provenance flags
    for col in df.columns:
        if col.startswith('found_by_'):
            features.append(col)
        if col.startswith('score_') and not col.startswith('score_xgb'):
            features.append(col)
        if col.startswith('rank_'):
            features.append(col)
            
    # Keep only available features
    final_features = [f for f in features if f in df.columns]
    
    # Fill NAs
    for f in final_features:
        df[f] = df[f].fillna(-999.0)
        
    return df, final_features
