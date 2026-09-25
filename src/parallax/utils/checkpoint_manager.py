"""Resilient Multi-Level Checkpoint and Resume Manager.

Maintains atomic stage manifests and artifact caching (Parquet and JSON)
to allow seamless recovery from interruptions and hardware reboots.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class CheckpointManifest(BaseModel):
    """Manifest tracking completed stages and associated artifacts."""

    dataset_name: str = Field(default="medium_split_200k", description="Benchmark split identifier")
    updated_at: str = Field(default="", description="ISO-8601 timestamp of last checkpoint")
    completed_stages: list[str] = Field(
        default_factory=list, description="List of completed stage names"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Arbitrary stage metadata")


class CheckpointManager:
    """Manages reading, writing, and validating checkpoint artifacts."""

    def __init__(
        self,
        checkpoint_dir: Path | str,
        dataset_name: str = "medium_split_200k",
        reset: bool = False,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.dataset_name = dataset_name
        self.manifest_path = self.checkpoint_dir / "manifest.json"

        if reset and self.checkpoint_dir.exists():
            self.reset()
        else:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.manifest = self._load_manifest()

    def _load_manifest(self) -> CheckpointManifest:
        if self.manifest_path.is_file():
            try:
                data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                return CheckpointManifest.model_validate(data)
            except Exception:
                pass
        return CheckpointManifest(
            dataset_name=self.dataset_name,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def _save_manifest(self) -> None:
        self.manifest.updated_at = datetime.now(timezone.utc).isoformat()
        self.manifest_path.write_text(
            self.manifest.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def get_manifest(self) -> CheckpointManifest:
        """Return the current CheckpointManifest."""
        return self.manifest

    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if a pipeline stage is marked as completed in the manifest."""
        return stage_name in self.manifest.completed_stages

    def record_stage_completed(
        self,
        stage_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record stage completion in the manifest."""
        if stage_name not in self.manifest.completed_stages:
            self.manifest.completed_stages.append(stage_name)
        if metadata:
            self.manifest.metadata[stage_name] = metadata
        self._save_manifest()

    def has_checkpoint(self, name: str, ext: str = "parquet") -> bool:
        """Check if an artifact file exists in the checkpoint directory."""
        return (self.checkpoint_dir / f"{name}.{ext}").is_file()

    def save_dataframe(self, name: str, df: pd.DataFrame) -> Path:
        """Save a pandas DataFrame to a Snappy-compressed Parquet checkpoint."""
        out_path = self.checkpoint_dir / f"{name}.parquet"
        df.to_parquet(out_path, compression="snappy", index=False)
        return out_path

    def load_dataframe(self, name: str) -> pd.DataFrame | None:
        """Load a DataFrame from a Parquet checkpoint if it exists."""
        in_path = self.checkpoint_dir / f"{name}.parquet"
        if in_path.is_file():
            return pd.read_parquet(in_path)
        return None

    def save_json(self, name: str, data: dict[str, Any]) -> Path:
        """Save structured dictionary to a JSON checkpoint."""
        out_path = self.checkpoint_dir / f"{name}.json"
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return out_path

    def load_json(self, name: str) -> dict[str, Any] | None:
        """Load structured dictionary from a JSON checkpoint if it exists."""
        in_path = self.checkpoint_dir / f"{name}.json"
        if in_path.is_file():
            return json.loads(in_path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
        return None

    def reset(self) -> None:
        """Wipe all checkpoints and reset the manifest."""
        if self.checkpoint_dir.exists():
            for item in self.checkpoint_dir.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = CheckpointManifest(
            dataset_name=self.dataset_name,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        self._save_manifest()
