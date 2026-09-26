"""
Parallax IndicXlit Representation Cache Generator
=================================================
Precomputes and caches neural transliterations for unique Indic-script entities
using AI4Bharat IndicXlit. Runs ONCE per unique entity, NOT per candidate pair.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

# Add local experiment scripts to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from detect_script import (
    has_indic_script,
    segment_mixed_script,
)


class IndicXlitCacheEngine:
    """Manages multi-lingual IndicXlit engines with per-word mixed-script fallback."""

    def __init__(
        self,
        beam_width: int = 4,
        rescore: bool = False,
        model_dir: str | None = None,
    ) -> None:
        self.beam_width = beam_width
        self.rescore = rescore
        self.model_dir = model_dir
        self.engines: dict[str, Any] = {}
        self._initialized = False

    def _get_engine(self, lang_code: str) -> Any:
        """Lazily initialize and cache XlitEngine per language."""
        if lang_code in self.engines:
            return self.engines[lang_code]

        try:
            from ai4bharat.transliteration import XlitEngine  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "ai4bharat-transliteration is required to run IndicXlit. "
                "Install it on Kaggle using: pip install ai4bharat-transliteration"
            ) from e

        # Initialize engine for Indic -> Roman (English)
        kwargs: dict[str, Any] = {
            "src_script_type": "indic",
            "beam_width": self.beam_width,
            "rescore": self.rescore,
        }
        if self.model_dir:
            kwargs["models_path"] = self.model_dir

        engine = XlitEngine(lang_code, **kwargs)
        self.engines[lang_code] = engine
        return engine

    def transliterate_string(self, text: str | None) -> str:
        """
        Transliterate string with mixed-script awareness.
        Translates Indic words using IndicXlit, keeps Latin/numeric words verbatim.
        """
        if not text or not str(text).strip():
            return ""

        raw_str = str(text).strip()
        if not has_indic_script(raw_str):
            return raw_str

        segments = segment_mixed_script(raw_str)
        out_tokens: list[str] = []

        for item in segments:
            if item.lang_code is not None:
                try:
                    engine = self._get_engine(item.lang_code)
                    # Returns {'<lang>': [best_word, ...]}
                    res = engine.translit_word(item.token, topk=1)
                    if isinstance(res, dict) and item.lang_code in res and res[item.lang_code]:
                        best_translit = res[item.lang_code][0]
                        out_tokens.append(best_translit)
                    else:
                        out_tokens.append(item.token)
                except Exception:
                    # Fallback on raw token if transliteration fails
                    out_tokens.append(item.token)
            else:
                out_tokens.append(item.token)

        return " ".join(out_tokens).strip()


def run_cache_generation(
    sources: list[pd.DataFrame],
    output_path: Path,
    beam_width: int = 4,
    rescore: bool = False,
    model_dir: str | None = None,
    resume: bool = True,
    dry_run: bool = False,
) -> pd.DataFrame:
    """
    Extract unique Indic entities from sources, transliterate, and cache.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Combine records to find unique entities
    combined = pd.concat(sources, ignore_index=True)
    combined = combined.drop_duplicates(subset=["entity_id"])

    # Filter to records with Indic script in business_name or business_address
    name_mask = combined["business_name"].apply(has_indic_script)
    addr_mask = combined["business_address"].apply(has_indic_script)
    indic_entities = combined[name_mask | addr_mask].copy()

    total_indic = len(indic_entities)
    print(f"Total unique records scanned: {len(combined):,}")
    print(f"Total entities with Indic script: {total_indic:,} ({total_indic / max(1, len(combined)) * 100:.2f}%)")

    if dry_run:
        print("[DRY-RUN] Exiting without performing model transliteration.")
        return pd.DataFrame()

    existing_df = pd.DataFrame()
    already_cached_ids: set[str] = set()

    if resume and output_path.exists():
        try:
            existing_df = pd.read_parquet(output_path)
            already_cached_ids = set(existing_df["entity_id"].astype(str))
            print(f"Loaded existing cache with {len(already_cached_ids):,} entities. Resuming...")
        except Exception as e:
            print(f"Warning: Could not read existing cache ({e}). Starting fresh.")

    to_process = indic_entities[~indic_entities["entity_id"].astype(str).isin(already_cached_ids)]
    print(f"Entities to process: {len(to_process):,}")

    if len(to_process) == 0:
        print("All Indic entities are already cached. Nothing to do.")
        return existing_df

    engine = IndicXlitCacheEngine(beam_width=beam_width, rescore=rescore, model_dir=model_dir)

    results: list[dict[str, Any]] = []
    t0 = time.time()

    for row in tqdm(to_process.itertuples(index=False), total=len(to_process), desc="IndicXlit"):
        eid = str(row.entity_id)
        raw_name = str(row.business_name) if pd.notna(row.business_name) else ""
        raw_addr = str(row.business_address) if pd.notna(row.business_address) else ""

        trans_name = engine.transliterate_string(raw_name) if has_indic_script(raw_name) else raw_name
        trans_addr = engine.transliterate_string(raw_addr) if has_indic_script(raw_addr) else raw_addr

        results.append({
            "entity_id": eid,
            "raw_name": raw_name,
            "indicxlit_name": trans_name,
            "raw_address": raw_addr,
            "indicxlit_address": trans_addr,
        })

    elapsed = time.time() - t0
    rate = len(to_process) / max(0.01, elapsed)
    print(f"Transliterated {len(to_process):,} entities in {elapsed:.1f}s ({rate:.1f} entities/sec).")

    new_df = pd.DataFrame(results)
    if not existing_df.empty:
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        final_df = final_df.drop_duplicates(subset=["entity_id"], keep="last")
    else:
        final_df = new_df

    final_df.to_parquet(output_path, index=False)
    print(f"Cache successfully saved to {output_path} ({len(final_df):,} total records).")
    return final_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute IndicXlit transliteration cache for Parallax")
    parser.add_argument("--scale", choices=["5000", "200000"], default="5000", help="Data split scale")
    parser.add_argument("--s1-path", type=Path, default=None, help="Path to Source 1 TSV")
    parser.add_argument("--s2-path", type=Path, default=None, help="Path to Source 2 TSV")
    parser.add_argument("--s3-path", type=Path, default=None, help="Path to Source 3 TSV")
    parser.add_argument("--output-path", type=Path, default=None, help="Output Parquet cache path")
    parser.add_argument("--beam-width", type=int, default=4, help="Beam width for beam search")
    parser.add_argument("--rescore", action="store_true", help="Enable dictionary rescoring")
    parser.add_argument("--model-dir", type=str, default=None, help="Directory containing offline model weights")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite existing cache instead of resuming")
    parser.add_argument("--dry-run", action="store_true", help="Inspect counts only without loading model")
    args = parser.parse_args()

    # Default paths based on scale
    if args.scale == "5000":
        s1_path = args.s1_path or Path("data/medium_split_200k/train_source1.tsv")
        s2_path = args.s2_path or Path("data/medium_split_200k/train_source2.tsv")
        s3_path = args.s3_path or Path("data/medium_split_200k/train_source3.tsv")
        out_path = args.output_path or Path("experiments/neural_text/caches/indicxlit_translit_cache_5k.parquet")
        s1_limit = 5000
    else:
        s1_path = args.s1_path or Path("data/medium_split_200k/train_source1.tsv")
        s2_path = args.s2_path or Path("data/medium_split_200k/train_source2.tsv")
        s3_path = args.s3_path or Path("data/medium_split_200k/train_source3.tsv")
        out_path = args.output_path or Path("experiments/neural_text/caches/indicxlit_translit_cache_200k.parquet")
        s1_limit = None

    print(f"Loading data from: {s1_path}, {s2_path}, {s3_path} (scale={args.scale})")
    s1_df = pd.read_csv(s1_path, sep="\t", nrows=s1_limit)
    s2_df = pd.read_csv(s2_path, sep="\t")
    s3_df = pd.read_csv(s3_path, sep="\t")

    run_cache_generation(
        sources=[s1_df, s2_df, s3_df],
        output_path=out_path,
        beam_width=args.beam_width,
        rescore=args.rescore,
        model_dir=args.model_dir,
        resume=not args.no_resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
