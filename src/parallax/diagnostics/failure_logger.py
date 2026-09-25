"""
Parallax Failure Diagnostics & Error Logging Engine
===================================================
Automated root-cause error triaging:
- Identifies and records all false negatives, false merges, and singleton violations
- Writes structured JSONL logs for automated ingestion (reports/failures_v1.jsonl)
- Synthesizes an executive markdown diagnostics scorecard (reports/diagnostics_v1.md).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from parallax.data.contracts import FailureRecord, FailureType
from parallax.metrics.evaluator import EvaluationReport


class FailureDiagnosticsLogger:
    """Audits entity resolution predictions and logs structured error records."""

    def __init__(self, reports_dir: Path | str = "reports") -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def analyze_and_log_failures(
        self,
        ground_truth: Mapping[str, set[str]],
        candidates: Mapping[str, set[str]],
        predictions: Mapping[str, set[str]],
        scored_pairs_df: pd.DataFrame,
        s1_df: pd.DataFrame,
        target_df: pd.DataFrame,
        report: EvaluationReport,
        log_filename: str = "failures_v1.jsonl",
        summary_filename: str = "diagnostics_v1.md",
    ) -> list[FailureRecord]:
        """
        Analyze predictions against ground truth and candidate sets,
        writing out failures_v1.jsonl and diagnostics_v1.md.
        """
        s1_dict = s1_df.set_index("entity_id").to_dict("index")
        tgt_dict = target_df.set_index("entity_id").to_dict("index")

        prob_lookup: dict[tuple[str, str], float] = {}
        if len(scored_pairs_df) > 0 and "s1_id" in scored_pairs_df and "cand_id" in scored_pairs_df:
            for _, row in scored_pairs_df.iterrows():
                prob_lookup[(str(row["s1_id"]), str(row["cand_id"]))] = float(row.get("prob", 0.0))

        failures: list[FailureRecord] = []

        for s1_id, true_set in ground_truth.items():
            s1_row = s1_dict.get(s1_id, {})
            s1_name = str(s1_row.get("business_name", s1_id))
            s1_addr = s1_row.get("business_address")

            cand_set = candidates.get(s1_id, set())
            pred_set = predictions.get(s1_id, set())

            # 1. Singleton Violations
            if len(true_set) == 0 and len(pred_set) > 0:
                for false_cand in pred_set:
                    c_row = tgt_dict.get(false_cand, {})
                    c_name = str(c_row.get("business_name", false_cand))
                    c_addr = c_row.get("business_address")
                    score = prob_lookup.get((s1_id, false_cand))

                    failures.append(
                        FailureRecord(
                            failure_type=FailureType.SINGLETON_VIOLATION,
                            s1_id=s1_id,
                            cand_id=false_cand,
                            s1_name=s1_name,
                            s1_address=str(s1_addr) if s1_addr else None,
                            cand_name=c_name,
                            cand_address=str(c_addr) if c_addr else None,
                            model_score=score,
                            root_cause_hint=(
                                "False candidate exceeded decision threshold on a true singleton."
                            ),
                        )
                    )

            # 2. False Merge Positives (Non-Singletons)
            if len(true_set) > 0:
                false_positives = pred_set - true_set
                for fp in false_positives:
                    c_row = tgt_dict.get(fp, {})
                    c_name = str(c_row.get("business_name", fp))
                    c_addr = c_row.get("business_address")
                    score = prob_lookup.get((s1_id, fp))

                    failures.append(
                        FailureRecord(
                            failure_type=FailureType.FALSE_MERGE_POSITIVE,
                            s1_id=s1_id,
                            cand_id=fp,
                            s1_name=s1_name,
                            s1_address=str(s1_addr) if s1_addr else None,
                            cand_name=c_name,
                            cand_address=str(c_addr) if c_addr else None,
                            model_score=score,
                            root_cause_hint=(
                                "False merge between distinct businesses sharing "
                                "name/address similarity."
                            ),
                        )
                    )

            # 3. False Negatives (Missed True Matches)
            if len(true_set) > 0:
                missed = true_set - pred_set
                for fn in missed:
                    c_row = tgt_dict.get(fn, {})
                    c_name = str(c_row.get("business_name", fn))
                    c_addr = c_row.get("business_address")
                    score = prob_lookup.get((s1_id, fn))

                    if fn not in cand_set:
                        ftype = FailureType.BLOCKING_FALSE_NEGATIVE
                        hint = "Blocking dropout: candidate was never retrieved in candidate_pairs."
                    else:
                        ftype = FailureType.CLASSIFICATION_FALSE_NEGATIVE
                        score_val = f"{score:.3f}" if score is not None else "0.0"
                        hint = f"Model score ({score_val}) fell below decision threshold."

                    failures.append(
                        FailureRecord(
                            failure_type=ftype,
                            s1_id=s1_id,
                            cand_id=fn,
                            s1_name=s1_name,
                            s1_address=str(s1_addr) if s1_addr else None,
                            cand_name=c_name,
                            cand_address=str(c_addr) if c_addr else None,
                            model_score=score,
                            root_cause_hint=hint,
                        )
                    )

        # Write JSONL
        jsonl_path = self.reports_dir / log_filename
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for fail in failures:
                f.write(fail.model_dump_json() + "\n")

        # Write Markdown Scorecard
        self._write_markdown_report(summary_filename, report, failures)

        return failures

    def _write_markdown_report(
        self,
        filename: str,
        report: EvaluationReport,
        failures: list[FailureRecord],
    ) -> None:
        """Render a readable diagnostics report."""
        md_path = self.reports_dir / filename

        by_type: dict[str, int] = {}
        for fail_item in failures:
            by_type[fail_item.failure_type.value] = (
                by_type.get(fail_item.failure_type.value, 0) + 1
            )

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# Parallax V1 Performance & Failure Diagnostics Report\n\n")
            f.write("## 1. Executive Metrics Scorecard\n\n")
            f.write("| Metric | Value | Description |\n")
            f.write("| :--- | :--- | :--- |\n")
            f.write(
                f"| **Macro F0.5 Score** | **`{report.macro_f05:.4f}`** | "
                "Official Leaderboard Metric |\n"
            )
            f.write(
                f"| Singleton Accuracy | `{report.singleton_score * 100:.2f}%` | "
                f"Score on true singletons ({report.total_singletons} total) |\n"
            )
            f.write(
                f"| Non-Singleton F0.5 | `{report.non_singleton_f05:.4f}` | "
                "Score on entities with true matches |\n"
            )
            f.write(
                f"| Ground Truth Matches | `{report.total_true_pairs:,}` | "
                "Total true positive pairs |\n"
            )
            f.write(
                f"| Correctly Resolved | `{report.total_correct_pairs:,}` | "
                "True matches captured by model |\n"
            )
            f.write(
                f"| Total Predicted | `{report.total_predicted_pairs:,}` | "
                "Predicted pairs above threshold |\n\n"
            )

            f.write("## 2. Failure Category Breakdown\n\n")
            f.write("| Error Type | Count | Severity / Impact |\n")
            f.write("| :--- | :--- | :--- |\n")
            f.write(
                f"| `BLOCKING_FALSE_NEGATIVE` | "
                f"{by_type.get(FailureType.BLOCKING_FALSE_NEGATIVE.value, 0)} | "
                "High (Limits recall ceiling) |\n"
            )
            f.write(
                f"| `CLASSIFICATION_FALSE_NEGATIVE` | "
                f"{by_type.get(FailureType.CLASSIFICATION_FALSE_NEGATIVE.value, 0)} | "
                "Moderate (Missed match) |\n"
            )
            f.write(
                f"| `FALSE_MERGE_POSITIVE` | "
                f"{by_type.get(FailureType.FALSE_MERGE_POSITIVE.value, 0)} | "
                "Critical (Penalized 2x under F0.5) |\n"
            )
            f.write(
                f"| `SINGLETON_VIOLATION` | "
                f"{by_type.get(FailureType.SINGLETON_VIOLATION.value, 0)} | "
                "Critical (Drops 1.0 entity score to 0.0) |\n\n"
            )

            f.write("## 3. Sample Failure Case Studies\n\n")
            for ftype in [
                FailureType.BLOCKING_FALSE_NEGATIVE,
                FailureType.FALSE_MERGE_POSITIVE,
                FailureType.SINGLETON_VIOLATION,
            ]:
                subset = [item for item in failures if item.failure_type == ftype][:3]
                if subset:
                    f.write(f"### {ftype.value} Examples\n\n")
                    for i, fail in enumerate(subset, 1):
                        f.write(
                            f"**Case #{i} [S1: `{fail.s1_id}` ↔ Cand: `{fail.cand_id}`]**\n"
                        )
                        f.write(
                            f"- **S1 Name / Address:** `{fail.s1_name}` | `{fail.s1_address}`\n"
                        )
                        f.write(
                            f"- **Cand Name / Address:** `{fail.cand_name}` | "
                            f"`{fail.cand_address}`\n"
                        )
                        score_str = (
                            f"{fail.model_score:.3f}" if fail.model_score is not None else "N/A"
                        )
                        f.write(f"- **Model Score:** `{score_str}`\n")
                        f.write(f"- **Root Cause:** {fail.root_cause_hint}\n\n")
