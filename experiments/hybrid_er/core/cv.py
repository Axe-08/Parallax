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


def split_s1_groups(
    s1_ids, n_train: int, n_val: int, seed: int, exclude=()
) -> tuple[list[str], list[str]]:
    """
    Draws disjoint train/validation S1 groups (sizes n_train, n_val) from `s1_ids` minus
    `exclude`, independent of input order and PYTHONHASHSEED. All rows/pairs of an S1 then
    follow its group, so no S1 can appear on both sides.
    """
    import numpy as np

    excluded = {str(s) for s in exclude}
    pool = np.array(sorted({str(s) for s in s1_ids} - excluded), dtype=object)
    if n_train + n_val > len(pool):
        raise ValueError(f"Requested {n_train}+{n_val} S1s but only {len(pool)} are available.")
    rng = np.random.default_rng(seed)
    rng.shuffle(pool)
    val = sorted(pool[:n_val].tolist())
    train = sorted(pool[n_val : n_val + n_train].tolist())
    return train, val
