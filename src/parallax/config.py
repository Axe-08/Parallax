"""
Parallax Configuration Module
=============================
Strictly typed configuration schemas using Pydantic v2.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


def _default_root() -> Path:
    return Path(os.getenv("PARALLAX_ROOT", Path.cwd()))


class PathConfig(BaseModel):
    """Local directory layout configuration."""

    project_root: Path = Field(default_factory=_default_root)
    raw_data_dir: Path = Field(default_factory=lambda: _default_root() / "data" / "raw")
    processed_data_dir: Path = Field(default_factory=lambda: _default_root() / "data" / "processed")
    golden_split_dir: Path = Field(
        default_factory=lambda: _default_root() / "data" / "golden_split"
    )
    images_dir: Path = Field(default_factory=lambda: _default_root() / "data" / "images")
    submissions_dir: Path = Field(default_factory=lambda: _default_root() / "submissions")
    checkpoints_dir: Path = Field(default_factory=lambda: _default_root() / "checkpoints")

    def ensure_directories(self) -> None:
        """Create all necessary directories if they do not exist."""
        for path in [
            self.raw_data_dir,
            self.processed_data_dir,
            self.golden_split_dir,
            self.images_dir,
            self.submissions_dir,
            self.checkpoints_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


class S3Config(BaseModel):
    """Central AWS S3 storage configuration."""

    bucket_name: str = Field(
        default_factory=lambda: os.getenv("PARALLAX_S3_BUCKET", "amazon-ml-2026-parallax-shared")
    )
    region: str = Field(default_factory=lambda: os.getenv("PARALLAX_AWS_REGION", "us-east-1"))
    prefix: str = Field(default="challenge-data")
    profile: str | None = Field(default_factory=lambda: os.getenv("AWS_PROFILE", None))


class SplitConfig(BaseModel):
    """Configuration for Golden Benchmark and cross-validation splits."""

    n_splits: int = Field(default=5, ge=2, description="Number of CV folds")
    golden_train_size: int = Field(default=5000, ge=100, description="Golden train rows")
    golden_val_size: int = Field(default=1000, ge=50, description="Golden validation rows")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")
    target_col: str = Field(default="entity_value", description="Target prediction column")
    group_col: str | None = Field(default=None, description="Group column for GroupKFold")


class ParallaxConfig(BaseModel):
    """Unified application configuration."""

    paths: PathConfig = Field(default_factory=PathConfig)
    s3: S3Config = Field(default_factory=S3Config)
    split: SplitConfig = Field(default_factory=SplitConfig)
    device: str = Field(default_factory=lambda: os.getenv("PARALLAX_DEVICE", "cuda"))
    debug: bool = Field(default=False)


def get_config() -> ParallaxConfig:
    """Retrieve default configuration with environment overrides."""
    cfg = ParallaxConfig()
    cfg.paths.ensure_directories()
    return cfg
