"""
Supervised contrastive fine-tuning of BGE-M3 for multi-positive entity resolution.

Training unit: one (anchor S1, positive target) pair. Every GT positive of an S1 becomes its
own example, so an S1 with matches {S2-A, S2-B, S3-C} contributes three examples and never
assumes a single positive.

Loss (per micro-batch, InfoNCE with false-negative masking):

    columns   = [positives of every row] + [hard negatives of every row]
    logits    = cos(anchor_i, column_j) / temperature
    mask      = column_j is another GT positive of anchor_i (and not row i's own positive)
    loss_i    = cross_entropy(logits_i with masked columns removed, target = i)

so a second true match of the same S1 appearing in the batch (as another row's positive, or
as a duplicated column) is never pushed away.

Hard negatives come from the multi-channel miner (training/hard_negatives.py); each example
samples `n_hard` of its anchor's mined negatives. One text view is sampled per micro-batch
(e.g. full / name / address) so one model serves all retrieval views.

Supports gradient accumulation (note: negatives are per micro-batch; accumulation raises the
effective optimizer batch, not the number of negatives), mixed precision (bf16 where
supported, else fp16 + GradScaler), frozen word embeddings, gradient checkpointing, periodic
validation with best-checkpoint selection, and exact resume from the last checkpoint.
"""
from __future__ import annotations

import json
import logging
import math
import random
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from experiments.hybrid_er.models.bge_encoder import BGEEncoder, autocast_dtype

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    model_name: str = "BAAI/bge-m3"
    seed: int = 42
    epochs: float = 1.0
    max_steps: int | None = None
    batch_size: int = 32  # anchors per micro-batch
    grad_accum: int = 1
    n_hard: int = 7
    lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    temperature: float = 0.02
    view_probs: dict[str, float] = field(default_factory=lambda: {"full": 0.5, "name": 0.25, "address": 0.25})
    max_length: dict[str, int] = field(default_factory=lambda: {"full": 128, "name": 48, "address": 96})
    precision: str = "auto"  # auto | fp16 | bf16 | fp32
    freeze_word_embeddings: bool = True
    gradient_checkpointing: bool = True
    eval_every: int = 500
    log_every: int = 50
    eval_ks: tuple[int, ...] = (1, 10, 20, 50)
    select_metric: str = "recall@10"
    encode_batch_size: int = 256


@dataclass
class TrainingData:
    """
    Integer-indexed training material. Entity indices refer to rows of `texts[view]`.

    examples      : (n, 2) int array of (anchor_idx, positive_idx)
    anchor_pos    : anchor_idx -> set of ALL GT positive target indices (for masking)
    anchor_negs   : anchor_idx -> int array of mined hard negatives (hardest first)
    texts         : view -> array of serialized texts, indexed by entity idx
    """

    examples: np.ndarray
    anchor_pos: Mapping[int, set[int]]
    anchor_negs: Mapping[int, np.ndarray]
    texts: Mapping[str, np.ndarray]


@dataclass
class ValidationData:
    """Retrieval validation: val queries vs a corpus containing their positives + distractors."""

    query_idx: np.ndarray
    query_country: np.ndarray
    query_pos: list[set[int]]  # GT positive entity indices per query
    corpus_idx: np.ndarray
    corpus_country: np.ndarray
    view: str = "full"


def masked_infonce(
    q: torch.Tensor,
    d: torch.Tensor,
    anchors: Sequence[int],
    col_targets: Sequence[int],
    anchor_pos: Mapping[int, set[int]],
    temperature: float,
) -> torch.Tensor:
    """
    q: (B, h) anchors; d: (N, h) columns whose first B entries are the rows' own positives.
    Masks every column that is a GT positive of the row's anchor except column i itself.
    """
    b = q.shape[0]
    logits = (q @ d.T) / temperature
    mask = torch.zeros_like(logits, dtype=torch.bool)
    for i, a in enumerate(anchors):
        pos = anchor_pos[a]
        for j, t in enumerate(col_targets):
            if j != i and t in pos:
                mask[i, j] = True
    logits = logits.masked_fill(mask, float("-inf"))
    return F.cross_entropy(logits, torch.arange(b, device=q.device))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BGETrainer:
    def __init__(
        self,
        config: TrainConfig,
        data: TrainingData,
        val: ValidationData | None,
        out_dir: Path | str,
        device: str,
    ) -> None:
        self.cfg = config
        self.data = data
        self.val = val
        self.out_dir = Path(out_dir)
        self.device = device
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[dict[str, Any]] = []
        self.best_metric = -math.inf
        self.best_step: int | None = None
        views = [v for v, p in config.view_probs.items() if p > 0]
        missing = set(views) - set(data.texts)
        if missing:
            raise ValueError(f"TrainingData has no texts for views {missing}")
        self.views = views
        self.view_p = np.array([config.view_probs[v] for v in views], dtype=float)
        self.view_p /= self.view_p.sum()
        for a in {int(x) for x in data.examples[:, 0]}:
            if a not in data.anchor_pos:
                raise ValueError(f"Anchor {a} has no GT positive set.")

    # ------------------------------------------------------------------ setup
    def steps_per_epoch(self) -> int:
        return math.ceil(len(self.data.examples) / (self.cfg.batch_size * self.cfg.grad_accum))

    def total_steps(self) -> int:
        if self.cfg.max_steps:
            return int(self.cfg.max_steps)
        return max(1, int(self.cfg.epochs * self.steps_per_epoch()))

    def _build(self, init_from: str) -> None:
        from transformers import get_linear_schedule_with_warmup

        self.encoder = BGEEncoder(init_from, gradient_checkpointing=self.cfg.gradient_checkpointing).to(self.device)
        self.frozen_params = self.encoder.freeze_word_embeddings() if self.cfg.freeze_word_embeddings else 0
        params = [p for p in self.encoder.parameters() if p.requires_grad]
        self.trainable_params = sum(p.numel() for p in params)
        decay = [p for n, p in self.encoder.named_parameters() if p.requires_grad and p.ndim > 1]
        no_decay = [p for n, p in self.encoder.named_parameters() if p.requires_grad and p.ndim <= 1]
        self.optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=self.cfg.lr,
        )
        total = self.total_steps()
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, int(self.cfg.warmup_ratio * total), total
        )
        self.amp = autocast_dtype(self.device, self.cfg.precision)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp == torch.float16)
        logger.info(
            "Encoder ready: trainable %.1fM params, frozen %.1fM, amp=%s, total_steps=%d",
            self.trainable_params / 1e6, self.frozen_params / 1e6, self.amp, total,
        )

    # ------------------------------------------------------------------ batches
    def _epoch_batches(self, epoch: int) -> list[np.ndarray]:
        rng = np.random.default_rng(self.cfg.seed * 7919 + epoch)
        perm = rng.permutation(len(self.data.examples))
        bs = self.cfg.batch_size
        return [perm[i : i + bs] for i in range(0, len(perm), bs)]

    def _micro_batch(self, rows: np.ndarray, rng: np.random.Generator) -> tuple[str, list[str], list[str], list[int], list[int]]:
        view = str(rng.choice(self.views, p=self.view_p))
        texts = self.data.texts[view]
        ex = self.data.examples[rows]
        anchors = [int(a) for a in ex[:, 0]]
        cols = [int(p) for p in ex[:, 1]]
        for a in anchors:
            negs = self.data.anchor_negs.get(a)
            if negs is None or len(negs) == 0:
                continue
            take = min(self.cfg.n_hard, len(negs))
            cols.extend(int(x) for x in rng.choice(negs, size=take, replace=False))
        return view, [texts[a] for a in anchors], [texts[c] for c in cols], anchors, cols

    # ------------------------------------------------------------------ checkpoints
    def _save(self, name: str, state: dict[str, Any] | None) -> Path:
        tmp = self.ckpt_dir / f".{name}.tmp"
        final = self.ckpt_dir / name
        if tmp.exists():
            shutil.rmtree(tmp)
        self.encoder.save(tmp)
        if state is not None:
            torch.save(state, tmp / "trainer_state.pt")
        (tmp / "train_config.json").write_text(json.dumps(asdict(self.cfg), indent=2, default=str))
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)
        return final

    def _state(self, step: int, epoch: int, batch_in_epoch: int) -> dict[str, Any]:
        return {
            "step": step,
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "best_metric": self.best_metric,
            "best_step": self.best_step,
            "history": self.history,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }

    # ------------------------------------------------------------------ validation
    @torch.no_grad()
    def validate(self, step: int) -> dict[str, float]:
        if self.val is None:
            return {}
        v = self.val
        t = time.time()
        texts = self.data.texts[v.view]
        ml = self.cfg.max_length[v.view]
        qe = self.encoder.encode([texts[i] for i in v.query_idx], device=self.device,
                                 batch_size=self.cfg.encode_batch_size, max_length=ml, show_progress=False)
        ce = self.encoder.encode([texts[i] for i in v.corpus_idx], device=self.device,
                                 batch_size=self.cfg.encode_batch_size, max_length=ml, show_progress=False)
        kmax = max(self.cfg.eval_ks)
        qt = torch.from_numpy(qe).to(self.device, dtype=torch.float32)
        ct = torch.from_numpy(ce).to(self.device, dtype=torch.float32)
        hits = {k: 0 for k in self.cfg.eval_ks}
        n_pos = 0
        rr = 0.0
        countries = np.unique(v.corpus_country)
        for c in countries:
            qm = np.flatnonzero(v.query_country == c)
            cm = np.flatnonzero(v.corpus_country == c)
            if len(qm) == 0 or len(cm) == 0:
                continue
            sims = qt[qm] @ ct[cm].T
            top = torch.topk(sims, k=min(kmax, len(cm)), dim=1).indices.cpu().numpy()
            cidx = v.corpus_idx[cm]
            for row, qi in enumerate(qm):
                pos = v.query_pos[qi]
                ranked = cidx[top[row]]
                n_pos += len(pos)
                for k in self.cfg.eval_ks:
                    hits[k] += len(pos & set(ranked[:k].tolist()))
                first = next((r for r, e in enumerate(ranked) if e in pos), None)
                rr += 0.0 if first is None else 1.0 / (first + 1)
        metrics = {f"recall@{k}": hits[k] / max(n_pos, 1) for k in self.cfg.eval_ks}
        metrics["mrr@{}".format(kmax)] = rr / max(len(v.query_idx), 1)
        metrics["val_seconds"] = round(time.time() - t, 1)
        metrics["step"] = step
        self.encoder.train()
        return metrics

    # ------------------------------------------------------------------ train
    def train(self, resume: bool = True) -> dict[str, Any]:
        _set_seed(self.cfg.seed)
        last = self.ckpt_dir / "last"
        state = None
        if resume and (last / "trainer_state.pt").exists():
            logger.info("Resuming from %s", last)
            self._build(str(last))
            state = torch.load(last / "trainer_state.pt", map_location="cpu", weights_only=False)
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.scaler.load_state_dict(state["scaler"])
            self.best_metric, self.best_step = state["best_metric"], state["best_step"]
            self.history = state["history"]
            random.setstate(state["rng"]["python"])
            np.random.set_state(state["rng"]["numpy"])
            torch.set_rng_state(state["rng"]["torch"])
            if state["rng"]["cuda"] is not None:
                torch.cuda.set_rng_state_all(state["rng"]["cuda"])
            step, epoch, start_batch = state["step"], state["epoch"], state["batch_in_epoch"]
        else:
            self._build(self.cfg.model_name)
            step, epoch, start_batch = 0, 0, 0
            m = self.validate(step)
            if m:
                logger.info("step 0 (pre-training) validation: %s", m)
                self.history.append({"type": "val", **m})
                # The pretrained weights compete for "best": fine-tuning must beat zero-shot.
                self.best_metric, self.best_step = m[self.cfg.select_metric], 0
                self._save("best", None)

        total = self.total_steps()
        accum = self.cfg.grad_accum
        self.encoder.train()
        t0 = time.time()
        seen = 0
        run_loss, run_n = 0.0, 0
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)

        while step < total:
            batches = self._epoch_batches(epoch)
            # Per-(epoch, batch) RNG makes view/negative sampling identical after resume.
            b = start_batch
            while b < len(batches) and step < total:
                self.optimizer.zero_grad(set_to_none=True)
                group = batches[b : b + accum]
                for rows in group:
                    rng = np.random.default_rng((self.cfg.seed, epoch, b))
                    view, q_txt, d_txt, anchors, cols = self._micro_batch(rows, rng)
                    ml = self.cfg.max_length[view]
                    batch = self.encoder.tokenize(q_txt + d_txt, ml, self.device)
                    with torch.autocast("cuda", dtype=self.amp, enabled=self.amp is not None):
                        emb = self.encoder(**batch)
                    q, d = emb[: len(q_txt)], emb[len(q_txt) :]
                    loss = masked_infonce(q, d, anchors, cols, self.data.anchor_pos, self.cfg.temperature)
                    self.scaler.scale(loss / len(group)).backward()
                    run_loss += float(loss.detach())
                    run_n += 1
                    seen += len(rows)
                    b += 1
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.encoder.parameters() if p.requires_grad], self.cfg.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                step += 1

                if step % self.cfg.log_every == 0 or step == total:
                    el = time.time() - t0
                    rec = {
                        "type": "train",
                        "step": step,
                        "epoch": epoch,
                        "loss": run_loss / max(run_n, 1),
                        "lr": self.scheduler.get_last_lr()[0],
                        "examples_per_s": seen / max(el, 1e-9),
                        "peak_vram_gib": round(torch.cuda.max_memory_allocated(self.device) / 2**30, 2)
                        if self.device.startswith("cuda") else None,
                    }
                    self.history.append(rec)
                    logger.info("train %s", rec)
                    run_loss, run_n = 0.0, 0

                if step % self.cfg.eval_every == 0 or step == total:
                    m = self.validate(step)
                    if m:
                        self.history.append({"type": "val", **m})
                        logger.info("val %s", m)
                        if m[self.cfg.select_metric] > self.best_metric:
                            self.best_metric, self.best_step = m[self.cfg.select_metric], step
                            self._save("best", None)
                            logger.info("New best %s=%.4f at step %d", self.cfg.select_metric, self.best_metric, step)
                    self._save("last", self._state(step, epoch, b))
            if b >= len(batches):
                epoch += 1
                start_batch = 0

        if self.best_step is None:  # no validation data: the final weights are the selection
            self.best_step = step
            self._save("best", None)
        summary = {
            "total_steps": total,
            "steps_per_epoch": self.steps_per_epoch(),
            "best_step": self.best_step,
            "best_metric": self.best_metric,
            "select_metric": self.cfg.select_metric,
            "train_seconds": round(time.time() - t0, 1),
            "trainable_params": self.trainable_params,
            "frozen_params": self.frozen_params,
            "peak_vram_gib": round(torch.cuda.max_memory_allocated(self.device) / 2**30, 2)
            if self.device.startswith("cuda") else None,
            "history": self.history,
        }
        (self.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary
