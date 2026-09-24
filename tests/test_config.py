"""Unit tests for Parallax configuration."""

from parallax.config import ParallaxConfig, PathConfig, S3Config, SplitConfig, get_config


def test_default_config_instantiation():
    cfg = get_config()
    assert isinstance(cfg, ParallaxConfig)
    assert isinstance(cfg.paths, PathConfig)
    assert isinstance(cfg.s3, S3Config)
    assert isinstance(cfg.split, SplitConfig)
    assert cfg.s3.region == "us-east-1"
    assert cfg.split.golden_train_size == 5000
    assert cfg.split.golden_val_size == 1000
