"""
Parallax Competition Data Contracts
===================================
Strictly typed Pydantic v2 domain schemas and TSV serializers/deserializers
for the Amazon ML Challenge 2026 Business Entity Resolution task.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from enum import Enum
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class BusinessRecord(BaseModel):
    """Pydantic v2 representation of an input entity record from S1, S2, or S3."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entity_id: str = Field(
        ...,
        description="Unique identifier with source prefix (e.g. S1-00001, S2-00047, S3-00812)",
    )
    business_name: str = Field(
        ...,
        description="Name of the business entity as given in raw source",
    )
    business_address: str | None = Field(
        default=None,
        description="Address of the business entity (may be null/missing)",
    )
    country: str = Field(
        ...,
        description="Country label (open set: US, India, France, etc.)",
    )


class CandidatePair(BaseModel):
    """A candidate pair generated during the blocking stage."""

    model_config = ConfigDict(frozen=True)

    s1_id: str = Field(..., description="Source 1 entity identifier")
    cand_id: str = Field(..., description="Candidate entity identifier from S2 or S3")
    score: float | None = Field(default=None, description="Optional blocking/similarity score")
    rank: int | None = Field(default=None, description="Rank within the candidate list")


class ResolutionResult(BaseModel):
    """Final entity resolution prediction for a single Source 1 entity."""

    model_config = ConfigDict(frozen=True)

    source1_entity_id: str = Field(..., description="Source 1 entity identifier")
    matched_entity_ids: list[str] = Field(
        default_factory=list,
        description="Ordered list of matching S2/S3 entity IDs (empty for singletons)",
    )


class FailureType(str, Enum):
    """Taxonomy of entity resolution errors for automated failure diagnostics."""

    BLOCKING_FALSE_NEGATIVE = "BLOCKING_FALSE_NEGATIVE"
    CLASSIFICATION_FALSE_NEGATIVE = "CLASSIFICATION_FALSE_NEGATIVE"
    FALSE_MERGE_POSITIVE = "FALSE_MERGE_POSITIVE"
    SINGLETON_VIOLATION = "SINGLETON_VIOLATION"


class FailureRecord(BaseModel):
    """Structured diagnostic log entry for a resolution failure."""

    model_config = ConfigDict(frozen=True)

    failure_type: FailureType = Field(..., description="Category of error")
    s1_id: str = Field(..., description="Source 1 entity identifier")
    cand_id: str | None = Field(
        default=None, description="Candidate entity identifier if applicable"
    )
    s1_name: str = Field(..., description="Source 1 business name")
    s1_address: str | None = Field(default=None, description="Source 1 address")
    cand_name: str | None = Field(default=None, description="Candidate business name if applicable")
    cand_address: str | None = Field(default=None, description="Candidate address if applicable")
    model_score: float | None = Field(
        default=None, description="Classifier probability or ranking score"
    )
    root_cause_hint: str = Field(..., description="Estimated root-cause explanation")


# --- TSV Reader & Writer Helpers ---


def load_business_records_df(file_path: Path | str) -> pd.DataFrame:
    """
    Read an entity TSV file (*_source1.tsv, *_source2.tsv, *_source3.tsv)
    with strict tab separator and utf-8 encoding.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Record file not found: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype={"entity_id": str, "business_name": str, "business_address": str, "country": str},
        keep_default_na=False,
    )
    # Convert empty address strings to None
    df["business_address"] = df["business_address"].apply(
        lambda x: None if (not x or x == "nan" or str(x).strip() == "") else str(x).strip()
    )
    return df


def load_ground_truth_dict(file_path: Path | str) -> dict[str, set[str]]:
    """
    Load train_ground_truth.tsv into a dictionary mapping source1_entity_id -> set of matched IDs.
    Singletons map to an empty set.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    gt_dict: dict[str, set[str]] = {}
    with open(path, encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n")
        parts = [p.strip() for p in header.split("\t")]
        if "source1_entity_id" not in parts or "matched_entity_ids" not in parts:
            raise ValueError(f"Invalid ground truth header: {parts}")

        s1_idx = parts.index("source1_entity_id")
        mids_idx = parts.index("matched_entity_ids")

        for line in f:
            raw = line.rstrip("\r\n")
            if not raw:
                continue
            cols = raw.split("\t")
            s1_id = cols[s1_idx].strip()
            if len(cols) > mids_idx and cols[mids_idx].strip() and cols[mids_idx].strip() != "nan":
                mids = {m.strip() for m in cols[mids_idx].split(",") if m.strip()}
            else:
                mids = set()
            gt_dict[s1_id] = mids

    return gt_dict


def write_candidate_pairs_tsv(
    output_path: Path | str,
    candidates: Mapping[str, Collection[str]],
) -> None:
    """
    Write candidate_pairs.tsv matching official competition format:
    source1_entity_id \\t candidate_entity_ids (comma-separated, no quotes).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in sorted(candidates.keys()):
            cand_list = sorted(list(candidates[s1_id]))
            cands_str = ",".join(cand_list)
            f.write(f"{s1_id}\t{cands_str}\n")


def write_matching_results_tsv(
    output_path: Path | str,
    predictions: Mapping[str, set[str] | list[str]],
) -> None:
    """
    Write matching_results.tsv matching official competition format:
    source1_entity_id \\t matched_entity_ids (comma-separated, no quotes).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in sorted(predictions.keys()):
            matches = sorted(list(predictions[s1_id]))
            matches_str = ",".join(matches)
            f.write(f"{s1_id}\t{matches_str}\n")
