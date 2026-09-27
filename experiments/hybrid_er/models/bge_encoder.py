"""
BGE-M3 dense encoder (CLS pooling + L2 normalization, as in BGE-M3's dense head).

Used both as the trainable bi-encoder (training/bge_trainer.py) and for inference
(corpus encoding for retrieval and pairwise similarity features). Inference sorts texts
by length to minimise padding, runs under autocast, and halves the batch on CUDA OOM
instead of crashing.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-m3"


def autocast_dtype(device: str, requested: str = "auto") -> torch.dtype | None:
    """None = no autocast. 'auto' -> bf16 where supported, else fp16 (V100), on CUDA only."""
    if not device.startswith("cuda") or requested == "fp32":
        return None
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    # is_bf16_supported() is True on V100 (sm_70) via slow emulation; require native bf16 (sm_80+).
    major, _ = torch.cuda.get_device_capability(torch.device(device))
    return torch.bfloat16 if major >= 8 else torch.float16


class BGEEncoder(nn.Module):
    def __init__(self, model_name_or_path: str | Path = DEFAULT_MODEL, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.model_name_or_path = str(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModel.from_pretrained(self.model_name_or_path)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    @property
    def dim(self) -> int:
        return int(self.model.config.hidden_size)

    def freeze_word_embeddings(self) -> int:
        """Freezes the (250K-row) word-embedding table; returns the number of frozen params."""
        emb = self.model.get_input_embeddings()
        for p in emb.parameters():
            p.requires_grad_(False)
        return sum(p.numel() for p in emb.parameters())

    def tokenize(self, texts: Sequence[str], max_length: int, device: str | torch.device) -> dict[str, torch.Tensor]:
        batch = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(out.last_hidden_state[:, 0].float(), dim=-1)

    @torch.inference_mode()
    def encode(
        self,
        texts: Sequence[str],
        *,
        device: str,
        batch_size: int = 256,
        max_length: int = 128,
        precision: str = "auto",
        show_progress: bool = True,
        out_dtype: np.dtype = np.float16,
    ) -> np.ndarray:
        """Returns L2-normalized embeddings (len(texts), dim) in the input order."""
        from tqdm import tqdm

        was_training = self.training
        self.eval()
        n = len(texts)
        out = np.empty((n, self.dim), dtype=out_dtype)
        if n == 0:
            return out
        lengths = np.fromiter((len(t) for t in texts), dtype=np.int64, count=n)
        order = np.argsort(-lengths, kind="stable")  # longest first: OOM surfaces immediately
        amp = autocast_dtype(device, precision)
        bs = batch_size
        i = 0
        pbar = tqdm(total=n, desc="encode", unit="txt", disable=not show_progress, mininterval=10)
        while i < n:
            idx = order[i : i + bs]
            try:
                batch = self.tokenize([texts[j] for j in idx], max_length, device)
                with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                    emb = self(**batch)
            except torch.cuda.OutOfMemoryError:
                if bs == 1:
                    raise
                torch.cuda.empty_cache()
                bs = max(1, bs // 2)
                logger.warning("CUDA OOM while encoding; retrying with batch size %d", bs)
                continue
            out[idx] = emb.cpu().numpy().astype(out_dtype)
            i += len(idx)
            pbar.update(len(idx))
        pbar.close()
        if was_training:
            self.train()
        return out

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)


def encode_multi_device(
    model_name_or_path: str | Path,
    texts: Sequence[str],
    devices: Sequence[str],
    chunk_size: int = 8192,
    **encode_kwargs,
) -> np.ndarray:
    """
    Data-parallel inference with one model replica per device. Devices pull fixed-size chunks
    from a shared queue, so a GPU slowed down by other tenants simply takes fewer chunks
    instead of stalling the whole job. Output order == input order.
    """
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    devices = list(devices)
    if len(devices) == 1 or len(texts) <= chunk_size:
        enc = BGEEncoder(model_name_or_path).to(devices[0])
        return enc.encode(texts, device=devices[0], **encode_kwargs)
    # transformers resolves AutoModel lazily; concurrent first imports from threads race.
    from transformers import AutoModel, AutoTokenizer  # noqa: F401

    starts = list(range(0, len(texts), chunk_size))
    results: dict[int, np.ndarray] = {}
    lock = threading.Lock()
    next_chunk = [0]
    kw = dict(encode_kwargs)
    kw["show_progress"] = False

    def work(dev: str) -> dict[str, float]:
        enc = BGEEncoder(model_name_or_path).to(dev)
        n, t0 = 0, time.time()
        try:
            while True:
                with lock:
                    if next_chunk[0] >= len(starts):
                        break
                    ci = next_chunk[0]
                    next_chunk[0] += 1
                s = starts[ci]
                results[ci] = enc.encode(texts[s : s + chunk_size], device=dev, **kw)
                n += len(results[ci])
        finally:
            del enc
            torch.cuda.empty_cache()
        return {"device": dev, "texts": n, "texts_per_s": n / max(time.time() - t0, 1e-9)}

    with ThreadPoolExecutor(max_workers=len(devices)) as ex:
        stats = list(ex.map(work, devices))
    logger.info("Multi-device encode: %s", [{k: (round(v, 1) if isinstance(v, float) else v) for k, v in st.items()} for st in stats])
    return np.concatenate([results[i] for i in range(len(starts))], axis=0)
