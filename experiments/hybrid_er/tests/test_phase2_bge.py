"""Phase 2 correctness tests: serialization, mining safety, loss masking, splits, dense index, trainer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
for p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.hybrid_er.core.cv import split_s1_groups
from experiments.hybrid_er.core.serialization import RecordSerializer, SerializerConfig
from experiments.hybrid_er.training.hard_negatives import mine_hard_negatives

torch = pytest.importorskip("torch")


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory) -> Path:
    """A 2-layer random XLM-R with a character-level fast tokenizer (no network needed)."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaModel

    d = tmp_path_factory.mktemp("tiny_bge")
    chars = list("abcdefghijklmnopqrstuvwxyz0123456789 |:<>-.,&'") + ["क", "म", "ल"]
    vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
    for c in chars:
        vocab.setdefault(c, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                                   unk_token="<unk>", cls_token="<s>", sep_token="</s>")
    fast.save_pretrained(d)
    cfg = XLMRobertaConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                           intermediate_size=64, max_position_embeddings=160, pad_token_id=1)
    torch.manual_seed(0)
    XLMRobertaModel(cfg).save_pretrained(d)
    return d


def _records(rows):
    return pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])


# ----------------------------------------------------------------------------- serialization
def test_serializer_views_and_explicit_missing():
    df = _records([("1", "Acme  LLC", "", "US"), ("2", "Ｂｅｓｔ Token", "12 Main St", "India")])
    ser = RecordSerializer()
    assert ser.serialize(df, "full") == [
        "name: acme llc | address: <missing> | country: us",
        "name: best token | address: 12 main st | country: india",
    ]
    assert ser.serialize(df, "address") == ["<missing>", "12 main st"]
    assert ser.serialize(df, "name_country")[1] == "name: best token | country: india"
    with pytest.raises(ValueError):
        ser.serialize(df, "nope")


def test_serializer_token_aware_field_truncation(tiny_model_dir):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_model_dir)
    ser = RecordSerializer(tok, SerializerConfig(max_name_tokens=5, max_address_tokens=100))
    df = _records([("1", "abcdefghij", "a very long address", "US")])
    out = ser.serialize(df, "full")[0]
    # name truncated on a token boundary, address untouched, field boundary preserved
    assert out == "name: abcde | address: a very long address | country: us"
    assert ser.truncation_counts["name"] == 1


# ----------------------------------------------------------------------------- mining
def _pool(rows):
    return pd.DataFrame(rows, columns=["s1_id", "source", "cand_id", "blocker", "rank", "score"])


def test_mining_never_emits_gt_and_tracks_provenance():
    gt = {"a": {"S2-1", "S3-9"}, "b": set()}
    pool = _pool([
        ("a", "S2", "S2-1", "bge", 1, 0.9),   # GT positive -> never a negative
        ("a", "S2", "S2-2", "bge", 2, 0.8),
        ("a", "S3", "S3-5", "lex", 1, 0.7),
        ("a", "S2", "S2-2", "lex", 3, 0.5),   # same cand found by two channels
        ("a", "S3", "S3-9", "lex", 2, 0.6),   # GT positive
        ("b", "S2", "S2-7", "lex", 1, 0.4),   # zero-match S1: everything is negative
    ])
    neg, summary = mine_hard_negatives(pool, gt, n_per_s1=10, channel_priority=["bge", "lex"])
    pairs = set(zip(neg["s1_id"], neg["cand_id"]))
    assert ("a", "S2-1") not in pairs and ("a", "S3-9") not in pairs
    assert pairs == {("a", "S2-2"), ("a", "S3-5"), ("b", "S2-7")}
    a = neg[neg["s1_id"] == "a"].set_index("cand_id")
    # interleave: lex rank-1 (S3-5) beats bge rank-2 (S2-2)
    assert a.loc["S3-5", "neg_order"] == 0 and a.loc["S2-2", "neg_order"] == 1
    assert a.loc["S2-2", "found_by"] == "bge|lex" and a.loc["S2-2", "hard_negative_source"] == "bge"
    assert a.loc["S2-2", "best_rank"] == 2
    assert summary.pool_positive_pairs == 2 and summary.negatives_found_by_multiple == 1


def test_mining_caps_per_s1_and_rejects_unknown_s1():
    pool = _pool([("a", "S2", f"S2-{i}", "bge", i, 1 - i / 10) for i in range(1, 8)])
    neg, _ = mine_hard_negatives(pool, {"a": set()}, n_per_s1=3)
    assert neg["cand_id"].tolist() == ["S2-1", "S2-2", "S2-3"]
    with pytest.raises(ValueError, match="missing from the supplied GT"):
        mine_hard_negatives(pool, {}, n_per_s1=3)
    with pytest.raises(ValueError, match="duplicate"):
        mine_hard_negatives(pd.concat([pool, pool.head(1)]), {"a": set()})


# ----------------------------------------------------------------------------- loss
def test_masked_infonce_excludes_other_positives_of_same_anchor():
    from experiments.hybrid_er.training.bge_trainer import masked_infonce

    torch.manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
    d = torch.nn.functional.normalize(torch.randn(5, 8), dim=-1)
    # rows: (anchor 10, pos 100), (anchor 10, pos 101); column 4 is target 102 (3rd positive of 10)
    anchors, cols = [10, 10], [100, 101, 555, 556, 102]
    anchor_pos = {10: {100, 101, 102}}
    tau = 0.1
    got = masked_infonce(q, d, anchors, cols, anchor_pos, tau)
    logits = (q @ d.T) / tau
    exp0 = -(logits[0, 0] - torch.logsumexp(logits[0, [0, 2, 3]], 0))
    exp1 = -(logits[1, 1] - torch.logsumexp(logits[1, [1, 2, 3]], 0))
    assert torch.allclose(got, (exp0 + exp1) / 2, atol=1e-6)


# ----------------------------------------------------------------------------- splits
def test_split_s1_groups_disjoint_excluding_and_order_independent():
    ids = [f"S1-{i}" for i in range(100)]
    tr, va = split_s1_groups(ids, 50, 10, seed=3, exclude=ids[:20])
    tr2, va2 = split_s1_groups(list(reversed(ids)), 50, 10, seed=3, exclude=set(ids[:20]))
    assert (tr, va) == (tr2, va2)
    assert not set(tr) & set(va) and not (set(tr) | set(va)) & set(ids[:20])
    with pytest.raises(ValueError):
        split_s1_groups(ids, 90, 10, seed=3, exclude=ids[:20])


# ----------------------------------------------------------------------------- dense index
def test_dense_index_exact_per_source_topk_and_country_fallback():
    pytest.importorskip("faiss")
    from experiments.hybrid_er.retrieval.dense_index import DenseIndex

    rng = np.random.default_rng(0)
    emb = rng.normal(size=(60, 16)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ids = np.array([f"T{i}" for i in range(60)], dtype=object)
    src = np.array(["S2"] * 30 + ["S3"] * 30)
    cty = np.array((["US"] * 15 + ["IN"] * 15) * 2)
    idx = DenseIndex("flat").build(emb, ids, src, cty)
    q = emb[[0, 20]] + 0.01
    hits = idx.search(q, np.array(["q0", "q1"], dtype=object), np.array(["US", "FR"]), k=3)
    for s in ("S2", "S3"):
        h = hits[(hits["s1_id"] == "q0") & (hits["source"] == s)]
        mask = (src == s) & (cty == "US")
        brute = ids[mask][np.argsort(-(emb[mask] @ q[0]))[:3]]
        assert h["cand_id"].tolist() == brute.tolist() and h["rank"].tolist() == [1, 2, 3]
    # unseen country "FR" searches all partitions of each source instead of being dropped
    assert len(hits[hits["s1_id"] == "q1"]) == 6 and idx.fallback_queries == 1
    assert set(hits["source"]) <= {"S2", "S3"}
    # the torch brute-force backend returns exactly the FAISS Flat results
    t_hits = DenseIndex("torch", device="cpu").build(emb, ids, src, cty).search(
        q, np.array(["q0", "q1"], dtype=object), np.array(["US", "FR"]), k=3)
    pd.testing.assert_frame_equal(t_hits.drop(columns="score"), hits.drop(columns="score"))
    assert np.allclose(t_hits["score"], hits["score"], atol=1e-5)


# ----------------------------------------------------------------------------- trainer
def _tiny_training(tiny_model_dir):
    from experiments.hybrid_er.training.bge_trainer import TrainingData, ValidationData

    names = ["acme", "best token", "kamal", "delta co", "zeta", "omega"]
    texts_full = np.array(
        [f"name: {n} | address: {i} main | country: us" for i, n in enumerate(names)]
        + [f"name: {n} llc | address: {i} main st | country: us" for i, n in enumerate(names)],
        dtype=object,
    )
    texts = {"full": texts_full, "name": texts_full, "address": texts_full}
    # anchors 0..5, their positives 6..11 (anchor 0 has two positives: 6 and 7)
    examples = np.array([(i, i + 6) for i in range(6)] + [(0, 7)], dtype=np.int64)
    anchor_pos = {i: {i + 6} for i in range(6)}
    anchor_pos[0] = {6, 7}
    anchor_negs = {i: np.array([j + 6 for j in range(6) if j != i and j + 6 not in anchor_pos[i]]) for i in range(6)}
    val = ValidationData(query_idx=np.array([2, 3]), query_country=np.array(["US", "US"]),
                         query_pos=[{8}, {9}], corpus_idx=np.arange(6, 12), corpus_country=np.array(["US"] * 6))
    return TrainingData(examples, anchor_pos, anchor_negs, texts), val


def test_trainer_runs_checkpoints_and_resumes(tiny_model_dir, tmp_path):
    from experiments.hybrid_er.training.bge_trainer import BGETrainer, TrainConfig

    data, val = _tiny_training(tiny_model_dir)
    cfg = TrainConfig(model_name=str(tiny_model_dir), max_steps=4, batch_size=4, grad_accum=2, n_hard=2,
                      lr=1e-3, eval_every=2, log_every=1, eval_ks=(1, 3), select_metric="recall@3",
                      gradient_checkpointing=False, freeze_word_embeddings=True, precision="fp32")
    summary = BGETrainer(cfg, data, val, tmp_path / "run", "cpu").train(resume=True)
    ck = tmp_path / "run" / "checkpoints"
    assert (ck / "best" / "model.safetensors").exists() and (ck / "last" / "trainer_state.pt").exists()
    assert summary["total_steps"] == 4 and summary["best_step"] in (0, 2, 4)
    vals = [h for h in summary["history"] if h["type"] == "val"]
    assert [v["step"] for v in vals] == [0, 2, 4]
    assert all(np.isfinite(h["loss"]) for h in summary["history"] if h["type"] == "train")

    # Resume: extend to 6 steps; it must continue from step 4, not restart.
    cfg6 = TrainConfig(**{**cfg.__dict__, "max_steps": 6})
    s6 = BGETrainer(cfg6, data, val, tmp_path / "run", "cpu").train(resume=True)
    train_steps = [h["step"] for h in s6["history"] if h["type"] == "train"]
    assert train_steps == [1, 2, 3, 4, 5, 6]
    state = torch.load(ck / "last" / "trainer_state.pt", weights_only=False)
    assert state["step"] == 6
    assert json.loads((ck / "last" / "train_config.json").read_text())["max_steps"] == 6


def test_encoder_outputs_normalized_in_input_order(tiny_model_dir):
    from experiments.hybrid_er.models.bge_encoder import BGEEncoder

    enc = BGEEncoder(str(tiny_model_dir))
    texts = ["a", "a much longer text here", "mid text"]
    e = enc.encode(texts, device="cpu", batch_size=2, max_length=32, show_progress=False, out_dtype=np.float32)
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-4)
    e1 = enc.encode([texts[1]], device="cpu", max_length=32, show_progress=False, out_dtype=np.float32)
    assert np.allclose(e[1], e1[0], atol=1e-4)


def test_lexical_retrieve_ranks_within_country():
    from experiments.hybrid_er.retrieval.lexical import lexical_retrieve

    s1 = _records([("q1", "Acme Holdings", "12 Main Street", "US"), ("q2", "Acme Holdings", "12 Main Street", "IN")])
    tg = _records([("S2-1", "Acme Holdings LLC", "12 Main St", "US"), ("S3-1", "Acme Holding", "99 Oak Ave", "US"),
                   ("S2-2", "Zeta", "1 Elm", "US"), ("S3-2", "Acme Holdings", "12 Main Street", "XX")])
    tg["source"] = ["S2", "S3", "S2", "S3"]
    hits = lexical_retrieve(s1, tg)
    assert set(hits["s1_id"]) == {"q1"}  # q2's country has no targets; nothing crosses countries
    name = hits[hits["blocker"] == "lex_name"].sort_values("rank")
    assert name["rank"].tolist() == list(range(1, len(name) + 1))
    assert "S3-2" not in set(hits["cand_id"])
    assert name["score"].is_monotonic_decreasing


def test_multi_device_encode_matches_single_device_order(tiny_model_dir):
    from experiments.hybrid_er.models.bge_encoder import BGEEncoder, encode_multi_device

    texts = [f"name: shop {i % 37} | address: {i} main" for i in range(1200)]
    multi = encode_multi_device(str(tiny_model_dir), texts, ["cpu", "cpu", "cpu"], chunk_size=100, max_length=32,
                                show_progress=False, out_dtype=np.float32)
    single = BGEEncoder(str(tiny_model_dir)).encode(texts, device="cpu", max_length=32,
                                                    show_progress=False, out_dtype=np.float32)
    assert multi.shape == single.shape and np.allclose(multi, single, atol=1e-4)
