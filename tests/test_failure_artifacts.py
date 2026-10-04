"""失败模式分类与产物写盘。"""

from __future__ import annotations

import json

import numpy as np

from rlalign.artifacts import (
    load_params,
    read_json,
    round_floats,
    save_params,
    write_json,
    write_jsonl,
)
from rlalign.env import generate_tasks
from rlalign.eval import summarize
from rlalign.failure import BUCKET_LABELS, classify, classify_record, render_table
from rlalign.policy import PolicyConfig, PolicyNet
from rlalign.reward import score_candidate

SMALL = PolicyConfig(d_emb=8, attn_dim=8, hidden=12, value_hidden=8)


def _rec(candidate: str, task=None, truncated: bool = False) -> dict:
    task = task or generate_tasks(1, seed=51, prefix="t")[0]
    return score_candidate(candidate, task, truncated=truncated)


# ------------------------------------------------------------------ 分类
def test_classify_record_ok():
    task = generate_tasks(1, seed=52, prefix="t")[0]
    assert classify_record(_rec(task.target, task)) == "ok"


def test_classify_record_truncated():
    task = generate_tasks(1, seed=53, prefix="t")[0]
    # 截断优先于语法类判定
    assert classify_record(_rec("a++b", task, truncated=True)) == "truncated"


def test_classify_record_unbalanced_paren():
    assert classify_record(_rec("a+b)")) == "unbalanced_paren"
    assert classify_record(_rec("(a+b")) == "unbalanced_paren"


def test_classify_record_syntax():
    assert classify_record(_rec("a++b")) == "syntax"


def test_classify_record_eval_error():
    """能解析但求值除零 -> eval_error 桶。"""
    rec = _rec("a/0")
    assert rec["kind"] == "eval"
    assert classify_record(rec) == "eval_error"


def test_classify_record_semantic():
    # "1" 一定能解析、一定不求值报错，但数值不对
    rec = _rec("1")
    assert rec["kind"] == "mismatch"
    assert classify_record(rec) == "semantic"


def test_classify_counts_are_exhaustive_and_disjoint():
    task = generate_tasks(1, seed=54, prefix="t")[0]
    records = [
        _rec(task.target, task),
        _rec("a++b", task),
        _rec("a+b)", task),
        _rec("a/0", task),
        _rec("1", task),
        _rec("a+b", task, truncated=True),
    ]
    cls = classify(records)
    assert sum(row["count"] for row in cls["table"]) == len(records)
    assert cls["n"] == len(records)
    assert abs(sum(row["share"] for row in cls["table"]) - 1.0) < 1e-12


def test_classify_success_rate_matches_records():
    task = generate_tasks(1, seed=55, prefix="t")[0]
    records = [_rec(task.target, task), _rec("1", task)]
    cls = classify(records)
    assert cls["success_rate"] == 0.5


def test_classify_all_buckets_present_even_when_empty():
    cls = classify([_rec("1")])
    assert {row["bucket"] for row in cls["table"]} == set(BUCKET_LABELS)


def test_classify_reports_by_injected_bug():
    task = generate_tasks(1, seed=56, prefix="t")[0]
    cls = classify([_rec(task.target, task)])
    assert task.bug in cls["by_injected_bug"]
    assert cls["by_injected_bug"][task.bug]["success_rate"] == 1.0


def test_classify_examples_are_recorded():
    cls = classify([_rec("a++b")])
    row = next(r for r in cls["table"] if r["bucket"] == "syntax")
    assert row["count"] == 1
    assert row["examples"][0]["candidate"] == "a++b"


def test_classify_on_empty_records_is_safe():
    cls = classify([])
    assert cls["n"] == 0
    assert cls["success_rate"] == 0.0


def test_render_table_contains_all_buckets():
    cls = classify([_rec("1")])
    text = render_table(cls)
    for bucket in BUCKET_LABELS:
        assert bucket in text
    assert "|" in text


# ------------------------------------------------------------------ 评估汇总
def test_summarize_reports_binomial_stderr():
    task = generate_tasks(1, seed=57, prefix="t")[0]
    records = [_rec(task.target, task)] * 25 + [_rec("1", task)] * 75
    s = summarize(records)
    assert s["n"] == 100
    assert s["success_rate"] == 0.25
    expected = (0.25 * 0.75 / 100) ** 0.5
    assert abs(s["success_rate_stderr"] - expected) < 1e-12


def test_summarize_empty():
    s = summarize([])
    assert s["n"] == 0
    assert s["success_rate"] == 0.0


# ------------------------------------------------------------------ 产物
def test_round_floats_recurses():
    out = round_floats({"a": 1.23456789, "b": [1.0 / 3.0], "c": 5, "d": True})
    assert out["a"] == 1.234568
    assert out["b"][0] == 0.333333
    assert out["c"] == 5
    assert out["d"] is True


def test_round_floats_handles_numpy_types():
    out = round_floats({"x": np.float64(1.5), "y": np.int64(3), "z": np.array([1.0 / 7.0])})
    assert out["x"] == 1.5
    assert out["y"] == 3
    assert isinstance(out["x"], float)
    assert isinstance(out["y"], int)
    assert out["z"] == [0.142857]


def test_write_json_is_byte_identical_across_runs(tmp_path):
    """可复现性的一部分：同样的数据写两次必须逐字节一致。"""
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    payload = {"b": 1.0 / 3.0, "a": [1, 2, 3], "c": {"nested": 2.0 / 7.0}}
    write_json(p1, payload)
    write_json(p2, payload)
    assert p1.read_bytes() == p2.read_bytes()
    assert b"\r\n" not in p1.read_bytes()  # 统一 LF，跨平台字节一致


def test_write_json_sorts_keys():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.json"
        write_json(p, {"z": 1, "a": 2})
        text = p.read_text(encoding="utf-8")
        assert text.index('"a"') < text.index('"z"')


def test_write_jsonl_is_byte_identical(tmp_path):
    rows = [{"i": i, "v": i / 3.0} for i in range(5)]
    p1 = tmp_path / "a.jsonl"
    p2 = tmp_path / "b.jsonl"
    write_jsonl(p1, rows)
    write_jsonl(p2, rows)
    assert p1.read_bytes() == p2.read_bytes()
    assert len(p1.read_text(encoding="utf-8").strip().splitlines()) == 5


def test_read_json_roundtrip(tmp_path):
    p = tmp_path / "r.json"
    write_json(p, {"x": [1.5, 2.5]})
    assert read_json(p) == {"x": [1.5, 2.5]}


def test_save_load_params_roundtrip(tmp_path):
    pol = PolicyNet(SMALL, np.random.default_rng(0))
    p = tmp_path / "ckpt.npz"
    save_params(p, pol)
    other = PolicyNet(SMALL, np.random.default_rng(99))
    load_params(p, other)
    for (_, a, _), (_, b, _) in zip(pol.parameters(), other.parameters()):
        assert np.array_equal(a, b)


def test_load_params_rejects_wrong_structure(tmp_path):
    pol = PolicyNet(SMALL, np.random.default_rng(0))
    p = tmp_path / "ckpt.npz"
    save_params(p, pol)
    bigger = PolicyNet(PolicyConfig(d_emb=16, attn_dim=16, hidden=20), np.random.default_rng(0))
    try:
        load_params(p, bigger)
    except ValueError:
        return
    raise AssertionError("参数结构不一致时应当报错")


def test_results_json_is_valid_json_after_write(tmp_path):
    p = tmp_path / "results.json"
    write_json(p, {"baselines": {"ppo": {"success_rate": 0.24000001}}})
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["baselines"]["ppo"]["success_rate"] == 0.24
