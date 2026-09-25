"""Unit tests for the resilient CheckpointManager and CheckpointManifest."""

from pathlib import Path

import pandas as pd

from parallax.utils.checkpoint_manager import CheckpointManager, CheckpointManifest


def test_checkpoint_manager_dataframe_roundtrip(tmp_path: Path) -> None:
    """Verify DataFrame save and load via snappy Parquet."""
    mgr = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")

    df = pd.DataFrame(
        {
            "s1_id": ["s1_1", "s1_2", "s1_3"],
            "cand_id": ["c_1", "c_2", "c_3"],
            "prob": [0.95, 0.42, 0.88],
        }
    )

    assert not mgr.has_checkpoint("test_stage")
    mgr.save_dataframe("test_stage", df)
    assert mgr.has_checkpoint("test_stage")

    loaded_df = mgr.load_dataframe("test_stage")
    assert loaded_df is not None
    assert len(loaded_df) == 3
    assert list(loaded_df["s1_id"]) == ["s1_1", "s1_2", "s1_3"]
    assert list(loaded_df["cand_id"]) == ["c_1", "c_2", "c_3"]
    assert loaded_df["prob"].iloc[0] == 0.95


def test_checkpoint_manager_json_roundtrip(tmp_path: Path) -> None:
    """Verify JSON dictionary save and load."""
    mgr = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")

    payload = {
        "fold": 1,
        "optimal_tau": 0.82,
        "macro_f05": 0.9412,
        "nested": {"key": "val", "count": 42},
    }

    assert not mgr.has_checkpoint("fold_1_metrics", ext="json")
    mgr.save_json("fold_1_metrics", payload)
    assert mgr.has_checkpoint("fold_1_metrics", ext="json")

    loaded_json = mgr.load_json("fold_1_metrics")
    assert loaded_json is not None
    assert loaded_json["fold"] == 1
    assert loaded_json["optimal_tau"] == 0.82
    assert loaded_json["nested"]["count"] == 42


def test_checkpoint_manager_stage_completion_tracking(tmp_path: Path) -> None:
    """Verify recording and querying stage completion in manifest."""
    mgr = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")

    assert not mgr.is_stage_completed("blocking")
    mgr.record_stage_completed("blocking", metadata={"total_pairs": 150000})
    assert mgr.is_stage_completed("blocking")

    manifest = mgr.get_manifest()
    assert isinstance(manifest, CheckpointManifest)
    assert "blocking" in manifest.completed_stages
    assert manifest.metadata["blocking"]["total_pairs"] == 150000

    # Test persistence of manifest across new CheckpointManager instance
    mgr2 = CheckpointManager(checkpoint_dir=tmp_path / "checkpoints")
    assert mgr2.is_stage_completed("blocking")


def test_checkpoint_manager_reset(tmp_path: Path) -> None:
    """Verify reset clears all saved checkpoints and manifest."""
    chk_dir = tmp_path / "checkpoints"
    mgr = CheckpointManager(checkpoint_dir=chk_dir)

    mgr.save_dataframe("df_test", pd.DataFrame({"x": [1, 2]}))
    mgr.save_json("json_test", {"status": "ok"})
    mgr.record_stage_completed("test_stage")

    assert mgr.has_checkpoint("df_test")
    assert mgr.has_checkpoint("json_test", ext="json")
    assert mgr.is_stage_completed("test_stage")

    # Reset
    mgr.reset()
    assert not mgr.has_checkpoint("df_test")
    assert not mgr.has_checkpoint("json_test", ext="json")
    assert not mgr.is_stage_completed("test_stage")


def test_checkpoint_manager_missing_files(tmp_path: Path) -> None:
    """Verify graceful handling when requested checkpoints do not exist."""
    mgr = CheckpointManager(checkpoint_dir=tmp_path / "empty_checkpoints")
    assert mgr.load_dataframe("non_existent") is None
    assert mgr.load_json("non_existent") is None
    assert not mgr.has_checkpoint("non_existent")
