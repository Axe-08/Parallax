"""
Parallax Qwen3-Embedding-0.6B Cache Generator
============================================
Computes and caches L2-normalized dense embeddings for unique entities using
Qwen3-Embedding-0.6B. Runs ONCE per unique entity, NOT per candidate pair.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add local experiment scripts to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def detect_device(requested: str = "auto") -> str:
    """Resolve compute device with auto-detection."""
    if requested in ("cuda", "cpu"):
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class QwenEmbeddingEngine:
    """Manages Qwen3-Embedding-0.6B inference with batching and L2 normalization."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "auto",
        batch_size: int = 64,
        max_length: int = 64,
        embedding_dim: int = 1024,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device = detect_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.model: Any = None
        self._backend = "sentence-transformers"

    def load_model(self) -> None:
        """Load model using sentence-transformers or raw transformers."""
        print(f"Loading Qwen3-Embedding-0.6B from: {self.model_name_or_path} (device={self.device})")
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

            kwargs: dict[str, Any] = {"device": self.device}
            # Load in float16 if running on CUDA to minimize VRAM and maximize throughput
            if self.device == "cuda":
                import torch

                kwargs["model_kwargs"] = {"torch_dtype": torch.float16}

            self.model = SentenceTransformer(self.model_name_or_path, **kwargs)
            self._backend = "sentence-transformers"
            print("Successfully loaded via sentence-transformers.")
        except Exception as e:
            print(f"sentence-transformers load failed ({e}), falling back to transformers AutoModel...")
            self._load_transformers_backend()

    def _load_transformers_backend(self) -> None:
        """Fallback loader using huggingface transformers directly."""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer  # type: ignore[import-not-found]

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path, padding_side="left", trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_name_or_path, torch_dtype=dtype, trust_remote_code=True
            ).to(self.device)
            self.model.eval()
            self._backend = "transformers"
            print("Successfully loaded via transformers AutoModel.")
        except Exception as err:
            raise RuntimeError(
                f"Failed to load Qwen embedding model from '{self.model_name_or_path}'. "
                f"Ensure packages are installed and model path exists. Error: {err}"
            ) from err

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode a list of text strings into L2-normalized float32 vectors."""
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        if self.model is None:
            self.load_model()

        if self._backend == "sentence-transformers":
            # sentence-transformers encode handles batching internally
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            # Matryoshka dimension truncation if requested
            if embeddings.shape[1] > self.embedding_dim:
                embeddings = embeddings[:, : self.embedding_dim]
                # Re-normalize after truncation
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-12)
            return np.asarray(embeddings, dtype=np.float32)

        # Fallback transformers backend
        import torch

        all_vecs: list[np.ndarray] = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Qwen Embedding Batches"):
            batch_texts = texts[i : i + self.batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                out = self.model(**encoded)
                # Last token pooling for causal encoder architectures
                attn = encoded["attention_mask"]
                seq_lens = attn.sum(dim=1) - 1
                hidden = out.last_hidden_state
                pooled = hidden[torch.arange(hidden.shape[0]), seq_lens]

                # MRL truncation
                if self.embedding_dim < pooled.shape[1]:
                    pooled = pooled[:, : self.embedding_dim]

                # L2 normalize
                norm = torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_vecs.append(norm.cpu().to(torch.float32).numpy())

        return np.vstack(all_vecs)


def run_qwen_cache_generation(
    records_df: pd.DataFrame,
    output_path: Path,
    metadata_path: Path,
    model_name_or_path: str = "Qwen/Qwen3-Embedding-0.6B",
    device: str = "auto",
    batch_size: int = 64,
    embedding_dim: int = 1024,
    resume: bool = True,
    dry_run: bool = False,
) -> None:
    """
    Generate and cache embeddings for all unique entities in records_df.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # Deduplicate entities
    unique_entities = records_df.drop_duplicates(subset=["entity_id"]).copy()
    unique_entities["business_name"] = unique_entities["business_name"].fillna("").astype(str)

    total_entities = len(unique_entities)
    print(f"Total unique entities to encode: {total_entities:,}")

    if dry_run:
        print("[DRY-RUN] Exiting without model download or inference.")
        return

    # Check for existing cache
    existing_ids: list[str] = []
    existing_embeddings: np.ndarray | None = None

    if resume and output_path.exists():
        try:
            data = np.load(output_path, allow_pickle=False)
            existing_ids = list(data["entity_ids"].astype(str))
            existing_embeddings = data["embeddings"]
            print(f"Loaded existing cache with {len(existing_ids):,} entities from {output_path}.")
        except Exception as e:
            print(f"Warning: Failed to load existing cache ({e}). Starting fresh.")

    cached_id_set = set(existing_ids)
    to_process = unique_entities[~unique_entities["entity_id"].astype(str).isin(cached_id_set)].copy()
    print(f"Entities remaining to encode: {len(to_process):,}")

    if len(to_process) == 0:
        print("All entities are already cached. Nothing to do.")
        return

    engine = QwenEmbeddingEngine(
        model_name_or_path=model_name_or_path,
        device=device,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
    )

    t0 = time.time()
    texts_to_encode = to_process["business_name"].tolist()
    new_embeddings = engine.encode_texts(texts_to_encode)
    elapsed = time.time() - t0
    rate = len(to_process) / max(0.01, elapsed)

    print(f"Encoded {len(to_process):,} entities in {elapsed:.1f}s ({rate:.1f} entities/sec).")

    new_ids = to_process["entity_id"].astype(str).tolist()

    if existing_embeddings is not None and len(existing_ids) > 0:
        combined_ids = np.array(existing_ids + new_ids, dtype=object)
        combined_embeddings = np.vstack([existing_embeddings, new_embeddings])
    else:
        combined_ids = np.array(new_ids, dtype=object)
        combined_embeddings = new_embeddings

    # Save to compressed .npz archive
    np.savez_compressed(
        output_path,
        entity_ids=combined_ids,
        embeddings=combined_embeddings.astype(np.float16),  # FP16 saves 50% disk
    )
    print(f"Saved {len(combined_ids):,} embeddings to {output_path} (shape={combined_embeddings.shape}).")

    metadata = {
        "model_name_or_path": model_name_or_path,
        "embedding_dim": embedding_dim,
        "device": engine.device,
        "batch_size": batch_size,
        "num_entities": len(combined_ids),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "throughput_entities_per_sec": round(rate, 2),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Qwen3-Embedding-0.6B cache for Parallax")
    parser.add_argument("--scale", choices=["5000", "200000"], default="5000", help="Data split scale")
    parser.add_argument("--s1-path", type=Path, default=None, help="Path to Source 1 TSV")
    parser.add_argument("--s2-path", type=Path, default=None, help="Path to Source 2 TSV")
    parser.add_argument("--s3-path", type=Path, default=None, help="Path to Source 3 TSV")
    parser.add_argument("--candidates-path", type=Path, default=None, help="Optional candidates parquet to filter")
    parser.add_argument("--output-path", type=Path, default=None, help="Output .npz path")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-Embedding-0.6B", help="Model name or local path")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Compute device")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    parser.add_argument("--embedding-dim", type=int, default=1024, help="Embedding dimension")
    parser.add_argument("--no-resume", action="store_true", help="Overwrite existing cache instead of resuming")
    parser.add_argument("--dry-run", action="store_true", help="Inspect counts only without loading model")
    args = parser.parse_args()

    if args.scale == "5000":
        s1_path = args.s1_path or Path("data/medium_split_200k/train_source1.tsv")
        s2_path = args.s2_path or Path("data/medium_split_200k/train_source2.tsv")
        s3_path = args.s3_path or Path("data/medium_split_200k/train_source3.tsv")
        cands_path = args.candidates_path or Path("baseline_artifacts/candidate_pairs_sample.parquet")
        out_path = args.output_path or Path("experiments/neural_text/caches/qwen_name_embeddings_5k.npz")
        meta_path = Path("experiments/neural_text/caches/qwen_metadata_5k.json")
        s1_limit = 5000
    else:
        s1_path = args.s1_path or Path("data/medium_split_200k/train_source1.tsv")
        s2_path = args.s2_path or Path("data/medium_split_200k/train_source2.tsv")
        s3_path = args.s3_path or Path("data/medium_split_200k/train_source3.tsv")
        cands_path = args.candidates_path or Path("data/full_dataset/candidate_pairs.parquet")
        out_path = args.output_path or Path("experiments/neural_text/caches/qwen_name_embeddings_200k.npz")
        meta_path = Path("experiments/neural_text/caches/qwen_metadata_200k.json")
        s1_limit = None

    print(f"Reading records for scale={args.scale}...")
    s1_df = pd.read_csv(s1_path, sep="\t", nrows=s1_limit)
    s2_df = pd.read_csv(s2_path, sep="\t")
    s3_df = pd.read_csv(s3_path, sep="\t")

    all_targets = pd.concat([s2_df, s3_df], ignore_index=True)

    # Filter targets to candidate pool if candidate pairs parquet exists to save compute
    if cands_path.exists():
        print(f"Filtering targets to entities appearing in {cands_path}...")
        cand_pairs = pd.read_parquet(cands_path, columns=["s1_id", "cand_id"])
        cand_ids = set(cand_pairs["cand_id"].astype(str).unique())
        filtered_targets = all_targets[all_targets["entity_id"].astype(str).isin(cand_ids)]
        combined = pd.concat([s1_df, filtered_targets], ignore_index=True)
    else:
        print("Candidate parquet not found; embedding all records in sources.")
        combined = pd.concat([s1_df, all_targets], ignore_index=True)

    run_qwen_cache_generation(
        records_df=combined,
        output_path=out_path,
        metadata_path=meta_path,
        model_name_or_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        resume=not args.no_resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
