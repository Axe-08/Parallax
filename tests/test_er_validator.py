"""Integration tests for official submission validator."""

from pathlib import Path

from utils.validate_submission import validate


def test_validator_pass_scenario(tmp_path: Path):
    test_dir = tmp_path / "test"
    test_dir.mkdir(parents=True)

    # Create dummy test files
    (test_dir / "test_source1.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-100\tAcme\t500 Market\tUS\n"
        "S1-200\tBeta\t600 Market\tUS\n",
        encoding="utf-8",
    )
    (test_dir / "test_source2.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-100\tAcme Corp\t500 Market St\tUS\n",
        encoding="utf-8",
    )
    (test_dir / "test_source3.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S3-200\tBeta Inc\t600 Market St\tUS\n",
        encoding="utf-8",
    )

    matching_file = tmp_path / "matching_results.tsv"
    matching_file.write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-100\tS2-100\nS1-200\t\n",  # singleton
        encoding="utf-8",
    )

    cand_file = tmp_path / "candidate_pairs.tsv"
    cand_file.write_text(
        "source1_entity_id\tcandidate_entity_ids\nS1-100\tS2-100,S3-200\nS1-200\t\n",
        encoding="utf-8",
    )

    errors, warnings = validate(
        matching_path=str(matching_file),
        candidate_path=str(cand_file),
        test_dir=str(test_dir),
        check_ids=True,
    )

    assert len(errors) == 0


def test_validator_catches_self_match(tmp_path: Path):
    test_dir = tmp_path / "test"
    test_dir.mkdir(parents=True)
    (test_dir / "test_source1.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-100\tAcme\t500 Market\tUS\n",
        encoding="utf-8",
    )

    matching_file = tmp_path / "matching_results.tsv"
    matching_file.write_text(
        "source1_entity_id\tmatched_entity_ids\nS1-100\tS1-100\n",  # Self-match error!
        encoding="utf-8",
    )

    errors, _ = validate(
        matching_path=str(matching_file),
        candidate_path=None,
        test_dir=str(test_dir),
    )
    assert any("self-match" in e.lower() for e in errors)
