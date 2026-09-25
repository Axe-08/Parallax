"""
Official ML Challenge 2026 Submission Validator
Checks matching_results.tsv and candidate_pairs.tsv against submission rules.
"""

from __future__ import annotations

import argparse
import os
import sys

MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")


def read_ids(tsv_path: str) -> set[str]:
    """Read the first column of a TSV file, skipping the header."""
    ids: set[str] = set()
    with open(tsv_path, encoding="utf-8") as f:
        header = f.readline()
        if not header:
            return ids
        for line in f:
            parts = line.split("\t", 1)
            if parts and parts[0].strip():
                ids.add(parts[0].strip())
    return ids


def load_match_targets(test_dir: str, warnings: list[str]) -> set[str] | None:
    """Load valid S2 and S3 IDs from test files."""
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    if not os.path.isfile(s2_path) or not os.path.isfile(s3_path):
        warnings.append(
            f"Could not find {s2_path} or {s3_path}; skipping match ID existence check."
        )
        return None
    valid: set[str] = set()
    valid.update(read_ids(s2_path))
    valid.update(read_ids(s3_path))
    return valid


def examples(s: set[str], n: int = 3) -> str:
    """Format sample elements from a set."""
    sample = sorted(s)[:n]
    out = ", ".join(sample)
    if len(s) > n:
        out += f", ... ({len(s)} total)"
    return out


def validate_id_list_file(
    path: str,
    expected_header: tuple[str, str],
    col_label: str,
    required: set[str],
    valid_ids: set[str] | None,
    errors: list[str],
) -> dict[str, set[str]] | None:
    """Validate TSV file containing an ID list column."""
    name = os.path.basename(path)
    if not os.path.isfile(path):
        errors.append(f"{name}: file not found at {path}.")
        return None

    mapping: dict[str, set[str]] = {}
    seen: set[str] = set()
    dup_rows: set[str] = set()
    intra_dupes: set[str] = set()
    self_matches: set[str] = set()
    wrong_prefix: set[str] = set()
    unknown: set[str] = set()
    n_rows = 0
    empties = 0

    with open(path, encoding="utf-8") as f:
        first = f.readline()
        if not first:
            errors.append(f"{name}: file is empty.")
            return None
        parts = first.rstrip("\r\n").split("\t")
        if tuple(parts) != expected_header:
            errors.append(f"{name}: header must be '\\t'.join({expected_header}), found {parts}.")
            return None

        for line_no, line in enumerate(f, 2):
            raw = line.rstrip("\r\n")
            if not raw:
                continue
            cols = raw.split("\t")
            if len(cols) == 1:
                s1, rest = cols[0].strip(), ""
            elif len(cols) == 2:
                s1, rest = cols[0].strip(), cols[1].strip()
            else:
                errors.append(
                    f"{name}: line {line_no} has {len(cols)} columns; "
                    "expected 2 tab-separated columns."
                )
                continue

            n_rows += 1
            if s1 in seen:
                dup_rows.add(s1)
            seen.add(s1)

            ids = [i.strip() for i in rest.split(",") if i.strip()] if rest else []
            if not ids:
                empties += 1
                mapping[s1] = set()
                continue
            if len(ids) != len(set(ids)):
                intra_dupes.add(s1)
            id_set = set(ids)
            mapping[s1] = id_set
            for mid in id_set:
                if mid.startswith("S1-"):
                    self_matches.add(mid)
                elif not mid.startswith(("S2-", "S3-")):
                    wrong_prefix.add(mid)
                elif valid_ids is not None and mid not in valid_ids:
                    unknown.add(mid)

    findings = [
        (
            dup_rows,
            "{name}: duplicate source1_entity_id row(s): {ex}. "
            "Each S1 entity may appear on only one row.",
        ),
        (
            intra_dupes,
            "{name}: repeated ID inside a {col} list for: {ex}. "
            "No duplicate IDs are allowed within a list.",
        ),
        (
            self_matches,
            "{name}: {col} contains Source-1 IDs (self-matches): {ex}. "
            "Only S2-/S3- IDs are allowed.",
        ),
        (wrong_prefix, "{name}: {col} contains IDs without an S2-/S3- prefix: {ex}."),
        (unknown, "{name}: {col} references IDs not in the test Source-2/3 files: {ex}."),
        (
            required - seen,
            "{name}: required S1 entity(ies) missing: {ex}. "
            "Every entity in test_source1.tsv needs a row (empty = no match).",
        ),
        (seen - required, "{name}: row(s) using an S1 ID that is not in the test set: {ex}."),
    ]
    for offenders, message in findings:
        if offenders:
            errors.append(message.format(name=name, ex=examples(offenders), col=col_label))

    print(f"  {name}: {n_rows} rows ({empties} empty, {n_rows - empties} non-empty).")
    return mapping


def validate(
    matching_path: str,
    candidate_path: str | None,
    test_dir: str,
    check_ids: bool = False,
) -> tuple[list[str], list[str]]:
    """Validate submission outputs against challenge rules."""
    errors: list[str] = []
    warnings: list[str] = []

    source1 = os.path.join(test_dir, "test_source1.tsv")
    if not os.path.isfile(source1):
        errors.append(f"Test source1 file not found: {source1} (check --test-dir).")
        return errors, warnings
    required = read_ids(source1)
    print(f"  required S1 entities: {len(required)}")

    valid_ids = load_match_targets(test_dir, warnings) if check_ids else None

    matched = validate_id_list_file(
        matching_path, MATCHING_HEADER, "matched_entity_ids", required, valid_ids, errors
    )

    candidate = None
    if candidate_path and os.path.isfile(candidate_path):
        candidate = validate_id_list_file(
            candidate_path, CANDIDATE_HEADER, "candidate_entity_ids", required, valid_ids, errors
        )
    elif candidate_path:
        warnings.append(f"{candidate_path} not found — skipping candidate_pairs.tsv checks.")

    if matched is not None and candidate is not None:
        offenders = {s1 for s1, mids in matched.items() if mids - candidate.get(s1, set())}
        if offenders:
            warnings.append(
                f"{len(offenders)} S1 entity(ies) have matched IDs not present in "
                f"candidate_pairs.tsv, e.g. {examples(offenders)}. Final matches "
                "normally come from your blocking candidates — double-check these."
            )

    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ML Challenge 2026 submission.")
    parser.add_argument("--matching", "-m", default="output/matching_results.tsv")
    parser.add_argument("--candidate", "-c", default=None)
    parser.add_argument("--test-dir", "-t", default="dataset/test")
    parser.add_argument("--check-ids", action="store_true")
    args = parser.parse_args()

    candidate_path = args.candidate or "output/candidate_pairs.tsv"
    print("ML Challenge 2026 — submission validator")
    print(f"  test dir: {args.test_dir}")

    try:
        errors, warnings = validate(
            args.matching, candidate_path, args.test_dir, check_ids=args.check_ids
        )
    except Exception as exc:
        print(f"\nFAIL — Exception during validation: {exc}")
        return 1

    print()
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        print(f"FAIL — {len(errors)} issue(s) to fix before submitting:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        return 1
    print("PASS — no blocking issues found. Safe to submit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
