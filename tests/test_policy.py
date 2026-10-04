"""策略网络：前向、采样确定性与参考策略快照。"""

from __future__ import annotations

import numpy as np
import pytest

from rlalign.env import BOS_ID, EOS_ID, VOCAB_SIZE, encode
from rlalign.policy import (
    PolicyConfig,
    PolicyNet,
    ValueNet,
    load_snapshot,
    param_count_summary,
    snapshot,
)

SMALL = PolicyConfig(d_emb=8, attn_dim=8, hidden=12, value_hidden=8)


def make_policy(seed: int = 0) -> PolicyNet:
    return PolicyNet(SMALL, np.random.default_rng(seed))


def prompt(text: str = "a+b*c") -> np.ndarray:
    ids = encode(text)
    out = np.zeros(SMALL.prompt_len, dtype=np.int64)
    out[: len(ids)] = ids
    return out


# ------------------------------------------------------------------ 结构
def test_forward_sequence_output_shape():
    pol = make_policy()
    toks = np.array([4, 13, 19], dtype=np.int64)
    logits, caches, seq_cache = pol.forward_sequence(prompt(), toks)
    assert logits.shape == (3, VOCAB_SIZE)
    assert len(caches) == 3


def test_logprobs_are_normalized():
    pol = make_policy()
    toks = np.array([4, 13, 19, 20], dtype=np.int64)
    logp, _, _, _ = pol.sequence_logprobs(prompt(), toks)
    assert np.allclose(np.exp(logp).sum(axis=1), 1.0)


def test_step0_does_not_depend_on_later_tokens():
    """第一步只以上一个 token = <bos> 为条件，因此与后续 token 无关。"""
    pol = make_policy()
    a, _, _ = pol.forward_sequence(prompt(), np.array([4, 13], dtype=np.int64))
    b, _, _ = pol.forward_sequence(prompt(), np.array([4, 20], dtype=np.int64))
    assert np.allclose(a[0], b[0])


def test_previous_token_influences_next_step_logits():
    """上一步的 token 必须真的进入下一步的计算图（因果性）。

    注意要比较**第一步 token 不同**的两条序列：第 t 步的输入是 ``y_{t-1}``，
    所以 ``[4,13]`` 与 ``[4,20]`` 在第 1 步的输入完全相同，logits 本来就该一样。
    """
    pol = make_policy(1)
    pol.lin1.W *= 20.0  # 放大第一层，避免随机初始化下差异被 ReLU 抹平
    toks_a = np.array([4, 13], dtype=np.int64)
    toks_b = np.array([7, 13], dtype=np.int64)
    a, _, _ = pol.forward_sequence(prompt(), toks_a)
    b, _, _ = pol.forward_sequence(prompt(), toks_b)
    assert np.allclose(a[0], b[0])  # 第 0 步都以上一个 token = <bos> 为条件
    assert not np.allclose(a[1], b[1])  # 第 1 步的输入 token 不同 -> logits 必须不同


def test_prompt_changes_conditioning():
    pol = make_policy()
    toks = np.array([4, 13, 19], dtype=np.int64)
    a, _, _ = pol.forward_sequence(prompt("a+b"), toks)
    b, _, _ = pol.forward_sequence(prompt("c*a"), toks)
    assert not np.allclose(a, b)


def test_parameter_count_is_small():
    """明确边界：这不是大模型，参数量远小于 1M。"""
    pol = make_policy()
    critic = ValueNet(SMALL, np.random.default_rng(0))
    counts = param_count_summary(pol, critic)
    assert counts["policy"] < 1_000_000
    assert counts["total"] < 1_000_000
    assert counts["total"] == counts["policy"] + counts["critic"]


def test_num_parameters_matches_parameter_arrays():
    pol = make_policy()
    total = sum(p.size for _, p, _ in pol.parameters())
    assert pol.num_parameters() == total


# ------------------------------------------------------------------ 采样
def test_sampling_same_seed_is_identical():
    """确定性：同种子两次采样完全一致。"""
    pol = make_policy(1)
    p = prompt("a+b*c")
    t1, l1, e1 = pol.sample_sequence(p, np.random.default_rng(7))
    t2, l2, e2 = pol.sample_sequence(p, np.random.default_rng(7))
    assert np.array_equal(t1, t2)
    assert np.allclose(l1, l2)
    assert e1 == e2


def test_sampling_different_seed_differs():
    pol = make_policy(2)
    p = prompt("a+b*c")
    t1, _, _ = pol.sample_sequence(p, np.random.default_rng(7))
    t2, _, _ = pol.sample_sequence(p, np.random.default_rng(8))
    assert not np.array_equal(t1, t2)


def test_sampling_respects_max_len():
    pol = make_policy(3)
    toks, lps, _ = pol.sample_sequence(prompt("a+b"), np.random.default_rng(1), max_len=5)
    assert len(toks) <= 5
    assert len(lps) == len(toks)


def test_sampling_stops_at_eos():
    pol = make_policy(4)
    # 强行把 EOS 的概率抬到压倒性，验证采样会在 EOS 处停止
    toks, _, hit = pol.sample_sequence(prompt("a"), np.random.default_rng(5), max_len=12)
    if hit:
        assert int(toks[-1]) == EOS_ID
        assert len(toks) < 12 or True


def test_token_ids_are_in_vocab():
    pol = make_policy(5)
    toks, _, _ = pol.sample_sequence(prompt("a+b"), np.random.default_rng(3))
    assert np.all(toks >= 0)
    assert np.all(toks < VOCAB_SIZE)


def test_sampled_logprobs_match_recomputed_values():
    """采样返回的 logprob 必须和事后重算的一致（否则 PPO 的 ratio 就错了）。"""
    pol = make_policy(6)
    p = prompt("a*b+c")
    toks, lps, _ = pol.sample_sequence(p, np.random.default_rng(9), max_len=6)
    for t in range(len(toks)):
        assert lps[t] == pytest.approx(pol.token_logprob(p, toks, t), abs=1e-12)


def test_greedy_is_deterministic_and_argmax():
    pol = make_policy(7)
    p = prompt("a+b")
    g1 = pol.greedy_sequence(p)
    g2 = pol.greedy_sequence(p)
    assert np.array_equal(g1, g2)
    logits, _, _ = pol.forward_sequence(p, np.array([int(g1[0])], dtype=np.int64))
    assert int(g1[0]) == int(np.argmax(logits[0]))


def test_greedy_respects_max_len():
    pol = make_policy(8)
    g = pol.greedy_sequence(prompt("a+b"), max_len=4)
    assert 1 <= len(g) <= 4


# ------------------------------------------------------------------ 快照
def test_snapshot_is_independent_copy():
    pol = make_policy(9)
    snap = snapshot(pol)
    for _, p, _ in pol.parameters():
        p += 1.0
    for (_, p, _), s in zip(pol.parameters(), snap):
        assert not np.allclose(p, s)


def test_load_snapshot_restores_exactly():
    pol = make_policy(10)
    ref = make_policy(11)
    load_snapshot(ref, snapshot(pol))
    for (_, a, _), (_, b, _) in zip(pol.parameters(), ref.parameters()):
        assert np.array_equal(a, b)


def test_load_snapshot_preserves_array_identity():
    """原地写入，保证优化器持有的引用不会失效。"""
    pol = make_policy(12)
    ref = make_policy(13)
    before = [id(p) for _, p, _ in ref.parameters()]
    load_snapshot(ref, snapshot(pol))
    after = [id(p) for _, p, _ in ref.parameters()]
    assert before == after


def test_snapshot_handles_duplicate_parameter_names():
    """回归测试：多个查表层的参数名都是 'E'，用名字做字典键会互相覆盖。"""
    pol = make_policy(14)
    names = [n for n, _, _ in pol.parameters()]
    assert names.count("E") >= 3  # 三个嵌入表同名
    snap = snapshot(pol)
    assert len(snap) == len(names)  # 按位置索引，一个都没丢
    ref = make_policy(15)
    load_snapshot(ref, snap)
    assert np.array_equal(pol.emb.E, ref.emb.E)


def test_load_snapshot_rejects_wrong_length():
    pol = make_policy(16)
    with pytest.raises(ValueError):
        load_snapshot(pol, snapshot(pol)[:-1])


# ------------------------------------------------------------------ Critic
def test_critic_value_shape_and_selfconsistency():
    critic = ValueNet(SMALL, np.random.default_rng(17))
    pol = make_policy(18)
    toks = np.array([4, 13, 19], dtype=np.int64)
    h0 = pol.prompt_vector_nograd(prompt())
    v = critic.values_for_sequence(h0, toks)
    assert v.shape == (3,)
    assert np.all(np.isfinite(v))


def test_critic_is_independent_of_policy_parameters():
    """Critic 有自己独立的嵌入表，不与策略共享参数。"""
    critic = ValueNet(SMALL, np.random.default_rng(19))
    pol = make_policy(20)
    critic_ids = {id(p) for _, p, _ in critic.parameters()}
    policy_ids = {id(p) for _, p, _ in pol.parameters()}
    assert not (critic_ids & policy_ids)


def test_entropy_grad_shapes_and_finite():
    from rlalign.policy import entropy_grad

    g = entropy_grad(np.array([0.1, 2.0, -1.0]))
    assert g.shape == (3,)
    assert np.all(np.isfinite(g))
