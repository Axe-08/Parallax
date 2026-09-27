import pandas as pd
import lightgbm as lgb
from typing import List

def extract_meta_features(candidate_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts features for the meta-blocker.
    """
    df = candidate_df.copy()
    features = []
    
    for col in df.columns:
        if col.startswith('found_by_') or col.startswith('score_') or col.startswith('rank_'):
            features.append(col)
            
    # Keep only available features
    final_features = [f for f in features if f in df.columns]
    
    # Fill NAs
    for f in final_features:
        df[f] = df[f].fillna(-999.0)
        
    return df, final_features

def train_meta_blocker(train_df: pd.DataFrame, target_col: str = 'is_match', params: dict = None) -> lgb.Booster:
    """
    Trains the meta-blocker (LightGBM).
    Must be trained on OOF downstream retrieval features.
    """
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.1,
            'num_leaves': 31,
            'verbose': -1,
            'random_state': 42
        }
        
    df, features = extract_meta_features(train_df)
    dtrain = lgb.Dataset(df[features], label=df[target_col])
    model = lgb.train(params, dtrain, num_boost_round=100)
    return model

def filter_top_m(candidate_df: pd.DataFrame, model: lgb.Booster, m: int = 50) -> pd.DataFrame:
    """
    Scores candidates and retains top M per S1.
    """
    df, features = extract_meta_features(candidate_df)
    if not features:
        return candidate_df.groupby('s1_id').head(m).reset_index(drop=True)
        
    df['meta_score'] = model.predict(df[features])
    
    # Sort and keep top M
    filtered = df.sort_values(['s1_id', 'meta_score'], ascending=[True, False]).groupby('s1_id').head(m)
    return filtered.reset_index(drop=True)
