"""
Split loading for the hybrid ER pipeline.

Populations are discovered from the files, never hard-coded. Records are read as raw
strings (QUOTE_NONE, no NA coercion) so that e.g. a business literally named "NA" is not
turned into a missing value, and quote characters are preserved exactly as the official
line-based submission validator sees them.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from parallax.data.contracts import load_ground_truth_dict

from experiments.hybrid_er.core.validation import validate_gt_global_contract

logger = logging.getLogger(__name__)

RECORD_COLUMNS = ("entity_id", "business_name", "business_address", "country")


def _count_data_lines(path: Path) -> int:
    with open(path, "rb") as f:
        n = sum(1 for line in f if line.strip())
    return n - 1  # header


def read_records(path: Path | str) -> pd.DataFrame:
    """Reads one *_sourceN.tsv file as raw strings and checks it parsed one row per line."""
    path = Path(path)
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )
    missing = set(RECORD_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    n_lines = _count_data_lines(path)
    if len(df) != n_lines:
        raise ValueError(f"{path}: parsed {len(df)} rows but file has {n_lines} data lines")
    if (df["entity_id"].str.strip() == "").any():
        raise ValueError(f"{path}: empty entity_id")
    if df["entity_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate entity_id")
    return df[list(RECORD_COLUMNS)]


@dataclass
class SplitData:
    name: str
    s1: pd.DataFrame
    s2: pd.DataFrame
    s3: pd.DataFrame
    gt: dict[str, set[str]] | None
    s1_ids: set[str] = field(init=False)
    s2_ids: set[str] = field(init=False)
    s3_ids: set[str] = field(init=False)

    def __post_init__(self) -> None:
        self.s1_ids = set(self.s1["entity_id"])
        self.s2_ids = set(self.s2["entity_id"])
        self.s3_ids = set(self.s3["entity_id"])
        overlap = (self.s1_ids & self.s2_ids) | (self.s1_ids & self.s3_ids) | (self.s2_ids & self.s3_ids)
        if overlap:
            raise ValueError(f"{self.name}: {len(overlap)} entity IDs appear in more than one source population")

    def counts(self) -> dict[str, int]:
        out = {"s1": len(self.s1), "s2": len(self.s2), "s3": len(self.s3)}
        if self.gt is not None:
            out["gt_s1"] = len(self.gt)
            out["gt_pairs"] = sum(len(v) for v in self.gt.values())
            out["gt_zero_match_s1"] = sum(1 for v in self.gt.values() if not v)
        return out


def load_split(split_dir: Path | str, prefix: str = "train", with_gt: bool = True) -> SplitData:
    """
    Loads {prefix}_source{1,2,3}.tsv (+ {prefix}_ground_truth.tsv) from `split_dir` and
    validates the global GT contract against the discovered populations.
    """
    split_dir = Path(split_dir)
    s1 = read_records(split_dir / f"{prefix}_source1.tsv")
    s2 = read_records(split_dir / f"{prefix}_source2.tsv")
    s3 = read_records(split_dir / f"{prefix}_source3.tsv")
    gt = load_ground_truth_dict(split_dir / f"{prefix}_ground_truth.tsv") if with_gt else None
    data = SplitData(name=f"{split_dir.name}/{prefix}", s1=s1, s2=s2, s3=s3, gt=gt)
    if gt is not None:
        validate_gt_global_contract(gt, data.s2_ids, data.s3_ids, s1_ids=data.s1_ids)
    logger.info("Loaded split %s: %s", data.name, data.counts())
    return data
