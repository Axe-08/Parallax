"""
Parallax Dataset Splitter & Golden Benchmark Generator
======================================================
Problem-agnostic dataset splitting engine for K-Fold, Stratified, Group,
and Golden Benchmark mini-datasets for rapid Day-1 prototyping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from parallax.config import get_config
from parallax.observability.tracer import PipelineTracer


def create_kfold_splits(
    df: pd.DataFrame,
    target_col: str | None = None,
    group_col: str | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign a fold integer column [0, n_splits-1] to each row."""
    df = df.copy().reset_index(drop=True)
    df["fold"] = -1

    if group_col and group_col in df.columns:
        splitter = GroupKFold(n_splits=n_splits)
        groups = df[group_col].to_numpy()
        splits = list(splitter.split(df, groups=groups))
    elif target_col and target_col in df.columns and (df[target_col].nunique() < len(df) * 0.5):
        # Use stratified split for categorical / classification targets
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(df, y=df[target_col].astype(str).to_numpy()))
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(df))

    for fold_idx, (_, val_idx) in enumerate(splits):
        df.iloc[val_idx, df.columns.get_loc("fold")] = fold_idx

    return df


def create_golden_benchmark(
    df: pd.DataFrame,
    target_col: str | None = None,
    train_size: int = 5000,
    val_size: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a high-signal stratified Golden Benchmark train and validation split."""
    total_requested = train_size + val_size
    if len(df) < total_requested:
        # Scale down proportionally if dataset is smaller than requested
        ratio = train_size / total_requested
        train_size = int(len(df) * ratio)
        val_size = len(df) - train_size

    # Sample candidate pool
    if target_col and target_col in df.columns and (df[target_col].nunique() < len(df) * 0.5):
        # Stratified sampling
        candidate_pool = (
            df.groupby(target_col, group_keys=False)
            .apply(lambda x: x.sample(frac=min(1.0, total_requested / len(df)), random_state=seed))
            .sample(n=min(total_requested, len(df)), random_state=seed)
        )
    else:
        candidate_pool = df.sample(n=min(total_requested, len(df)), random_state=seed)

    candidate_pool = candidate_pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df = candidate_pool.iloc[:val_size].copy().reset_index(drop=True)
    train_df = candidate_pool.iloc[val_size : val_size + train_size].copy().reset_index(drop=True)

    return train_df, val_df


def run_splitter(
    input_file: Path,
    output_dir: Path,
    target_col: str | None = None,
    group_col: str | None = None,
    golden_train: int = 5000,
    golden_val: int = 1000,
    n_splits: int = 5,
    seed: int = 42,
) -> None:
    """Execute splitting and export Golden Split and 5-Fold manifests."""
    tracer = PipelineTracer("dataset_splitter")

    with tracer.span("load_raw_data"):
        if input_file.suffix == ".parquet":
            df = pd.read_parquet(input_file)
        else:
            df = pd.read_csv(input_file)

    output_dir.mkdir(parents=True, exist_ok=True)

    with tracer.span("generate_golden_split"):
        golden_train_df, golden_val_df = create_golden_benchmark(
            df=df,
            target_col=target_col,
            train_size=golden_train,
            val_size=golden_val,
            seed=seed,
        )
        golden_train_path = output_dir / "golden_train.parquet"
        golden_val_path = output_dir / "golden_val.parquet"
        golden_train_df.to_parquet(golden_train_path, index=False)
        golden_val_df.to_parquet(golden_val_path, index=False)

        # Also write CSV copies for easy inspection
        golden_train_df.to_csv(output_dir / "golden_train.csv", index=False)
        golden_val_df.to_csv(output_dir / "golden_val.csv", index=False)

    with tracer.span("generate_full_folds"):
        folded_df = create_kfold_splits(
            df=df,
            target_col=target_col,
            group_col=group_col,
            n_splits=n_splits,
            seed=seed,
        )
        full_folded_path = output_dir / "train_folded.parquet"
        folded_df.to_parquet(full_folded_path, index=False)

    # Save metadata summary
    metadata = {
        "source_rows": len(df),
        "target_col": target_col,
        "golden_train_rows": len(golden_train_df),
        "golden_val_rows": len(golden_val_df),
        "n_splits": n_splits,
        "seed": seed,
    }
    (output_dir / "golden_metadata.json").write_text(json.dumps(metadata, indent=2))

    tracer.finish(success=True)
    print(f"\n✅ Splits successfully generated at {output_dir}:")
    print(f"   Golden Train: {golden_train_path} ({len(golden_train_df)} rows)")
    print(f"   Golden Val:   {golden_val_path} ({len(golden_val_df)} rows)")
    print(f"   Full Folded:  {full_folded_path} ({len(folded_df)} rows, {n_splits} folds)")


def main() -> None:
    """CLI Entry point."""
    cfg = get_config()
    parser = argparse.ArgumentParser(description="Parallax Golden Split & K-Fold Splitter")
    parser.add_argument("--input", "-i", type=Path, default=cfg.paths.raw_data_dir / "train.csv")
    parser.add_argument("--output-dir", "-o", type=Path, default=cfg.paths.golden_split_dir)
    parser.add_argument("--target-col", default=None, help="Target column name")
    parser.add_argument("--group-col", default=None, help="Group column for GroupKFold")
    parser.add_argument(
        "--golden-train", type=int, default=5000, help="Number of golden train rows"
    )
    parser.add_argument("--golden-val", type=int, default=1000, help="Number of golden val rows")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_splitter(
        input_file=args.input,
        output_dir=args.output_dir,
        target_col=args.target_col,
        group_col=args.group_col,
        golden_train=args.golden_train,
        golden_val=args.golden_val,
        n_splits=args.n_splits,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
