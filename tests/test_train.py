"""训练闭环：能真的提升成功率、确定性与基线行为。"""

from __future__ import annotations

import numpy as np
import pytest

from rlalign.env import VOCAB_SIZE, generate_tasks
from rlalign.eval import evaluate_policy, evaluate_random_policy
from rlalign.grpo import GRPOConfig
from rlalign.nn import Adam
from rlalign.policy import PolicyConfig, PolicyNet, ValueNet, load_snapshot, snapshot
from rlalign.ppo import PPOConfig, ppo_iteration
from rlalign.reward import RewardConfig, score_candidate
from rlalign.rollout import collect_rollout, pad_prompts
from rlalign.train import RunConfig, build_tasks, train_sft

RC = RewardConfig()


# ------------------------------------------------------------------ 小规模配置
def tiny_cfg(seed: int = 0, n_train: int = 40) -> RunConfig:
    return RunConfig(
        seed=seed,
        n_train=n_train,
        n_eval=20,
        eval_interval=5,
        sft_epochs=40,
        sft_lr=8e-3,
        d_emb=16,
        hidden=64,
    )


# ------------------------------------------------------------------ rollout
def test_pad_prompts_shape_and_alignment():
    tasks = generate_tasks(3, seed=61, prefix="t")
    ids = pad_prompts(tasks, prompt_len=16)
    assert ids.shape == (3, 16)
    from rlalign.env import PAD_ID, encode

    assert list(ids[0, : len(tasks[0].buggy)]) == encode(tasks[0].buggy)
    assert np.all(ids[0, len(tasks[0].buggy) :] == PAD_ID)


def test_collect_rollout_shapes_and_mask():
    tasks = generate_tasks(4, seed=62, prefix="t")
    pol = PolicyNet(PolicyConfig(d_emb=8, attn_dim=8, hidden=12), np.random.default_rng(0))
    ref = PolicyNet(PolicyConfig(d_emb=8, attn_dim=8, hidden=12), np.random.default_rng(0))
    load_snapshot(ref, snapshot(pol))
    b = collect_rollout(pol, tasks, np.random.default_rng(1), ref_policy=ref, group_size=1)
    assert b.n == 4
    assert b.token_ids.shape == b.mask.shape == b.logp_old.shape
    assert b.n_tokens == int(b.mask.sum())
    assert np.all(b.lengths >= 1)
    assert np.all(b.mask.sum(axis=1) == b.lengths)
    # 填充区域的 mask 必须为 0
    for i in range(b.n):
        assert np.all(b.mask[i, int(b.lengths[i]) :] == 0.0)
    assert len(b.strings) == 4
    assert b.values is None  # 没传 critic
    assert b.rewards is not None


def test_collect_rollout_group_mode():
    tasks = generate_tasks(3, seed=63, prefix="t")
    pol = PolicyNet(PolicyConfig(d_emb=8, attn_dim=8, hidden=12), np.random.default_rng(0))
    b = collect_rollout(pol, tasks, np.random.default_rng(1), group_size=4)
    assert b.n == 12
    R, g = b.group_matrix()
    assert g == 4
    assert R.shape == (3, 4)
    assert list(b.group_ids) == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]


def test_collect_rollout_is_deterministic():
    tasks = generate_tasks(4, seed=64, prefix="t")
    p1 = PolicyNet(PolicyConfig(d_emb=8, attn_dim=8, hidden=12), np.random.default_rng(0))
    p2 = PolicyNet(PolicyConfig(d_emb=8, attn_dim=8, hidden=12), np.random.default_rng(0))
    b1 = collect_rollout(p1, tasks, np.random.default_rng(5), group_size=2)
    b2 = collect_rollout(p2, tasks, np.random.default_rng(5), group_size=2)
    assert np.array_equal(b1.token_ids, b2.token_ids)
    assert np.allclose(b1.logp_old, b2.logp_old)
    assert b1.strings == b2.strings


def test_collect_rollout_with_critic_produces_values():
    tasks = generate_tasks(3, seed=65, prefix="t")
    cfg = PolicyConfig(d_emb=8, attn_dim=8, hidden=12, value_hidden=8)
    pol = PolicyNet(cfg, np.random.default_rng(0))
    crit = ValueNet(cfg, np.random.default_rng(1))
    b = collect_rollout(pol, tasks, np.random.default_rng(2), critic=crit, group_size=1)
    assert b.values is not None
    assert b.values.shape == b.token_ids.shape
    assert np.all(np.isfinite(b.values))


# ------------------------------------------------------------------ 评估
def test_random_policy_baseline_is_poor():
    """随机策略不应该蒙对：这是基线可信度的基本检查。"""
    ev = generate_tasks(60, seed=66, prefix="ev")
    res = evaluate_random_policy(ev, VOCAB_SIZE, seed=1, reward_cfg=RC)
    assert res["success_rate"] == 0.0
    assert res["mean_reward"] < 0.2


def test_random_policy_is_deterministic():
    ev = generate_tasks(20, seed=67, prefix="ev")
    a = evaluate_random_policy(ev, VOCAB_SIZE, seed=3, reward_cfg=RC)
    b = evaluate_random_policy(ev, VOCAB_SIZE, seed=3, reward_cfg=RC)
    assert a["success_rate"] == b["success_rate"]
    assert [r["candidate"] for r in a["records"]] == [r["candidate"] for r in b["records"]]


def test_evaluate_policy_reports_records():
    ev = generate_tasks(10, seed=68, prefix="ev")
    pol = PolicyNet(PolicyConfig(d_emb=8, attn_dim=8, hidden=12), np.random.default_rng(0))
    res = evaluate_policy(pol, ev, seed=7, reward_cfg=RC)
    assert res["n"] == 10
    assert len(res["records"]) == 10
    assert 0.0 <= res["success_rate"] <= 1.0


def test_evaluate_policy_greedy_is_deterministic():
    ev = generate_tasks(10, seed=69, prefix="ev")
    pol = PolicyNet(PolicyConfig(d_emb=8, attn_dim=8, hidden=12), np.random.default_rng(0))
    a = evaluate_policy(pol, ev, seed=7, reward_cfg=RC, greedy=True)
    b = evaluate_policy(pol, ev, seed=8, reward_cfg=RC, greedy=True)
    assert a["success_rate"] == b["success_rate"]


# ------------------------------------------------------------------ SFT 真的能学
def test_sft_reduces_loss():
    cfg = tiny_cfg(seed=0)
    train = generate_tasks(cfg.n_train, seed=cfg.seed, prefix="tr")
    _pol, hist = train_sft(cfg, train, log=lambda *a: None)
    assert hist[-1]["sft_loss"] < hist[0]["sft_loss"]
    assert np.isfinite(hist[-1]["sft_loss"])


def test_sft_improves_success_rate_over_random_policy():
    """核心断言：训练必须真的把成功率从随机水平抬起来（趋势方向正确）。"""
    cfg = RunConfig(seed=0, n_train=40, n_eval=20, sft_epochs=40, sft_lr=8e-3)
    train = generate_tasks(cfg.n_train, seed=cfg.seed, prefix="tr")
    pol, _hist = train_sft(cfg, train, log=lambda *a: None)

    trained = evaluate_policy(pol, train, seed=999, reward_cfg=RC)["success_rate"]
    untrained_pol = PolicyNet(
        PolicyConfig(d_emb=cfg.d_emb, hidden=cfg.hidden), np.random.default_rng(cfg.seed)
    )
    untrained = evaluate_policy(untrained_pol, train, seed=999, reward_cfg=RC)["success_rate"]

    assert untrained < 0.05, f"未训练策略竟然有 {untrained:.3f} 的成功率"
    assert trained > 0.5, f"SFT 训练后训练集成功率只有 {trained:.3f}，没有学到东西"


def test_sft_is_deterministic():
    cfg = tiny_cfg(seed=1, n_train=24)
    train = generate_tasks(cfg.n_train, seed=cfg.seed, prefix="tr")
    p1, h1 = train_sft(cfg, train, log=lambda *a: None)
    p2, h2 = train_sft(cfg, train, log=lambda *a: None)
    assert h1[-1]["sft_loss"] == h2[-1]["sft_loss"]
    for (_, a, _), (_, b, _) in zip(p1.parameters(), p2.parameters()):
        assert np.array_equal(a, b)


def test_sft_improves_process_reward():
    """过程奖励是稠密信号，SFT 之后它应当明显上升。"""
    cfg = tiny_cfg(seed=2, n_train=32)
    train = generate_tasks(cfg.n_train, seed=cfg.seed, prefix="tr")
    pol, _ = train_sft(cfg, train, log=lambda *a: None)
    after = evaluate_policy(pol, train, seed=999, reward_cfg=RC)["mean_process"]
    before = evaluate_policy(
        PolicyNet(PolicyConfig(d_emb=cfg.d_emb, hidden=cfg.hidden), np.random.default_rng(cfg.seed)),
        train,
        seed=999,
        reward_cfg=RC,
    )["mean_process"]
    assert after > before


# ------------------------------------------------------------------ RL 冒烟
def test_ppo_smoke_improves_training_reward():
    """小规模 PPO 冒烟：过程奖励提供稠密信号，训练批平均 reward 应当上升。"""
    cfg = tiny_cfg(seed=3, n_train=40)
    train = generate_tasks(cfg.n_train, seed=cfg.seed, prefix="tr")
    pcfg = PolicyConfig(d_emb=cfg.d_emb, hidden=cfg.hidden)
    pol = PolicyNet(pcfg, np.random.default_rng(cfg.seed))
    crit = ValueNet(pcfg, np.random.default_rng(cfg.seed + 1))
    ref = PolicyNet(pcfg, np.random.default_rng(cfg.seed + 2))
    load_snapshot(ref, snapshot(pol))
    opt = Adam(pol.parameters() + crit.parameters(), lr=1e-3)
    rl_cfg = PPOConfig(iterations=25, prompts_per_iter=16, inner_epochs=2, lr=1e-3)
    rng = np.random.default_rng(cfg.seed + 3)

    first = None
    last = None
    for it in range(rl_cfg.iterations):
        batch = collect_rollout(pol, train[:16], rng, ref_policy=ref, critic=crit, reward_cfg=RC)
        ppo_iteration(pol, crit, batch, rl_cfg, opt)
        if it == 0:
            first = float(np.mean(batch.rewards))
        last = float(np.mean(batch.rewards))
    assert np.isfinite(first) and np.isfinite(last)
    assert last > first, f"PPO 训练后 reward 没有上升: {first:.4f} -> {last:.4f}"


def test_grpo_smoke_runs_and_stays_finite():
    from rlalign.grpo import grpo_iteration

    cfg = tiny_cfg(seed=4, n_train=16)
    train = generate_tasks(cfg.n_train, seed=cfg.seed, prefix="tr")
    pcfg = PolicyConfig(d_emb=cfg.d_emb, hidden=cfg.hidden)
    pol = PolicyNet(pcfg, np.random.default_rng(cfg.seed))
    opt = Adam(pol.parameters(), lr=1e-3)
    rl_cfg = GRPOConfig(iterations=6, prompts_per_iter=4, group_size=4, inner_epochs=1, lr=1e-3)
    rng = np.random.default_rng(cfg.seed + 3)
    for _ in range(rl_cfg.iterations):
        batch = collect_rollout(pol, train[:4], rng, group_size=4, reward_cfg=RC)
        stats = grpo_iteration(pol, batch, rl_cfg, opt)
        assert all(np.isfinite(v) for v in stats.values())
    assert all(np.all(np.isfinite(p)) for _, p, _ in pol.parameters())


def test_build_tasks_split_is_disjoint():
    cfg = RunConfig(n_train=50, n_eval=25, seed=5)
    train, ev = build_tasks(cfg)
    assert len(train) == 50 and len(ev) == 25
    assert not ({t.buggy for t in train} & {t.buggy for t in ev})
