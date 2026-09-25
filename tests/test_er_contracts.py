"""Tests for Parallax Data Contracts."""

from pathlib import Path

from parallax.data.contracts import (
    BusinessRecord,
    CandidatePair,
    write_candidate_pairs_tsv,
    write_matching_results_tsv,
)


def test_business_record_validation():
    rec = BusinessRecord(
        entity_id="S1-12345",
        business_name="Acme Corp",
        business_address="500 Market St",
        country="US",
    )
    assert rec.entity_id == "S1-12345"
    assert rec.country == "US"


def test_candidate_pair_serialization():
    cand = CandidatePair(s1_id="S1-1", cand_id="S2-2", score=0.95, rank=1)
    assert cand.s1_id == "S1-1"
    assert cand.score == 0.95


def test_tsv_roundtrip(tmp_path: Path):
    cands_file = tmp_path / "candidate_pairs.tsv"
    matches_file = tmp_path / "matching_results.tsv"

    test_cands = {"S1-001": {"S2-10", "S3-20"}, "S1-002": set()}
    write_candidate_pairs_tsv(cands_file, test_cands)

    assert cands_file.is_file()
    content = cands_file.read_text(encoding="utf-8").splitlines()
    assert content[0] == "source1_entity_id\tcandidate_entity_ids"
    assert "S1-001\tS2-10,S3-20" in content
    assert "S1-002\t" in content

    test_matches = {"S1-001": {"S2-10"}, "S1-002": set()}
    write_matching_results_tsv(matches_file, test_matches)
    assert matches_file.is_file()
