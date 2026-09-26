"""
Parallax Neural Representation Track — Master CLI Orchestrator
==============================================================
Chains cache generation, pairwise feature construction, and E0-E3 experiment evaluation.
Defaults to --dry-run to prevent accidental local heavy execution.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = EXPERIMENT_DIR / "scripts"


def run_cmd(cmd: list[str], dry_run: bool = False) -> None:
    """Execute command with logging and error checking."""
    cmd_str = " ".join(cmd)
    print(f"\n[RUN] {cmd_str}")
    if dry_run:
        print("  -> Dry-run enabled: skipped execution.")
        return

    t0 = time.time()
    res = subprocess.run(cmd, check=False)
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"ERROR: Command exited with code {res.returncode} in {elapsed:.1f}s.")
        sys.exit(res.returncode)
    print(f"SUCCESS: Completed in {elapsed:.1f}s.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallax Master Orchestrator for Neural Representation Experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scale", choices=["5000", "200000"], default="5000", help="Data split scale to execute")
    parser.add_argument(
        "--stage",
        choices=["all", "indicxlit", "qwen", "features", "eval"],
        default="all",
        help="Pipeline stage to execute",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device for neural models")
    parser.add_argument("--qwen-model-path", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="Model name or local offline directory")
    parser.add_argument("--indicxlit-model-dir", type=str, default=None, help="Directory containing offline IndicXlit weights")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument(
        "--run-experiment",
        action="store_true",
        help="Explicit flag required to run the real pipeline. If not passed, runs in dry-run mode.",
    )
    args = parser.parse_args()

    python_exe = sys.executable
    is_dry_run = not args.run_experiment

    if is_dry_run:
        print("\n" + "=" * 70)
        print("SAFETY GUARD: Running in DRY-RUN mode.")
        print("To actually execute expensive model inference and training on Kaggle,")
        print("pass the flag: --run-experiment")
        print("=" * 70 + "\n")

    print(f"Target Scale: {args.scale}")
    print(f"Target Stage: {args.stage}")
    print(f"Target Device: {args.device}")

    # Stage 1: IndicXlit Cache
    if args.stage in ("all", "indicxlit"):
        print("\n--- STAGE 1: Generating IndicXlit Transliteration Cache ---")
        cmd = [
            python_exe,
            str(SCRIPTS_DIR / "generate_indicxlit_cache.py"),
            "--scale", args.scale,
        ]
        if args.indicxlit_model_dir:
            cmd.extend(["--model-dir", args.indicxlit_model_dir])
        if is_dry_run:
            cmd.append("--dry-run")
        run_cmd(cmd, dry_run=False)

    # Stage 2: Qwen Embedding Cache
    if args.stage in ("all", "qwen"):
        print("\n--- STAGE 2: Generating Qwen3-Embedding Cache ---")
        cmd = [
            python_exe,
            str(SCRIPTS_DIR / "generate_qwen_cache.py"),
            "--scale", args.scale,
            "--device", args.device,
            "--model-path", args.qwen_model_path,
            "--batch-size", str(args.batch_size),
        ]
        if is_dry_run:
            cmd.append("--dry-run")
        run_cmd(cmd, dry_run=False)

    # Stage 3: Pairwise Features Construction
    if args.stage in ("all", "features"):
        print("\n--- STAGE 3: Constructing Pairwise Neural Features ---")
        cmd = [
            python_exe,
            str(SCRIPTS_DIR / "build_pairwise_features.py"),
            "--scale", args.scale,
        ]
        run_cmd(cmd, dry_run=is_dry_run)

    scale_tag = "5k" if args.scale == "5000" else "200k"

    # Stage 4: Frozen 5-Fold Evaluation
    if args.stage in ("all", "eval"):
        print("\n--- STAGE 4: Running Frozen 5-Fold CV Harness (E0 - E3) ---")
        cmd = [
            python_exe,
            str(SCRIPTS_DIR / "run_experiments.py"),
            "--augmented-features", str(EXPERIMENT_DIR / "caches" / f"augmented_features_{scale_tag}.parquet"),
        ]
        run_cmd(cmd, dry_run=is_dry_run)

    print("\n[COMPLETE] Neural representation workflow finished.")


if __name__ == "__main__":
    main()
