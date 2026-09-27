import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

def get_s1_grouped_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42, target_col: str = None):
    """
    Yields (train_idx, val_idx) splits enforcing that the same s1_id never appears
    in both training and validation sets (P0.5 / P0.6).
    """
    if 's1_id' not in df.columns:
        raise ValueError("DataFrame must contain 's1_id' for grouped cross-validation.")
        
    groups = df['s1_id'].astype(str)
    
    if target_col and target_col in df.columns:
        y = df[target_col]
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return cv.split(df, y, groups=groups)
    else:
        cv = GroupKFold(n_splits=n_splits)
        return cv.split(df, groups=groups)
