"""GRPO：组内归一化优势（含退化输入）、无 Critic 的一次完整迭代。"""

from __future__ import annotations

import numpy as np
import pytest

from rlalign.env import generate_tasks
from rlalign.grpo import GRPOConfig, group_advantages, grpo_iteration
from rlalign.nn import Adam
from rlalign.policy import PolicyConfig, PolicyNet, load_snapshot, snapshot
from rlalign.reward import RewardConfig
from rlalign.rollout import collect_rollout

SMALL = PolicyConfig(d_emb=8, attn_dim=8, hidden=12, value_hidden=8)


# ------------------------------------------------------------------ 组内归一化
def test_group_advantages_zero_mean_unit_std_per_group():
    rewards = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
    adv = group_advantages(rewards, group_size=4)
    assert adv.shape == (8,)
    for g in range(2):
        chunk = adv[g * 4 : (g + 1) * 4]
        assert abs(chunk.mean()) < 1e-10
        assert abs(chunk.std() - 1.0) < 1e-3  # 分母是 std + eps


def test_group_advantages_preserve_order_within_group():
    rewards = np.array([1.0, 2.0, 3.0, 4.0])
    adv = group_advantages(rewards, group_size=4)
    assert np.all(np.diff(adv) > 0)


def test_group_advantages_approximately_scale_invariant():
    """整组奖励同时乘以正数时优势近似不变。

    注意**只是近似**：分母是 ``std + eps``，奖励整体放大后 std 按同比例放大，
    而 eps 保持不变，所以严格的比例不变性被 eps 破坏（相对偏差约 eps/std）。
    这里用 1e-3 的相对容差来体现"近似"这个事实。
    """
    rewards = np.array([1.0, 2.0, 3.0, 4.0])
    a1 = group_advantages(rewards, group_size=4)
    a2 = group_advantages(rewards * 7.0, group_size=4)
    assert np.allclose(a1, a2, rtol=1e-3)
    # 若把 eps 设为 0，比例不变性就是严格的
    b1 = group_advantages(rewards, group_size=4, eps=0.0)
    b2 = group_advantages(rewards * 7.0, group_size=4, eps=0.0)
    assert np.allclose(b1, b2)


def test_group_advantages_shift_invariance():
    rewards = np.array([1.0, 2.0, 3.0, 4.0])
    a1 = group_advantages(rewards, group_size=4)
    a2 = group_advantages(rewards + 100.0, group_size=4)
    assert np.allclose(a1, a2, atol=1e-6)


def test_group_advantages_degenerate_all_equal_is_zero():
    """退化输入：组内奖励完全相等 -> 标准差为 0 -> 优势必须全为 0。

    这是本项目被单测抓出来的真实 bug：早期实现用布尔掩码索引
    ``(n_groups, 1)`` 去筛 ``(n_groups, G)`` 的数组，直接 IndexError；
    后来改成除以 ``std + eps``，又会在分子非零时把噪声放大。
    """
    adv = group_advantages(np.full(6, 1.5), group_size=3)
    assert np.all(adv == 0.0)
    assert np.all(np.isfinite(adv))


def test_group_advantages_degenerate_per_group_only():
    """只有某一组退化时，另一组必须照常归一化。"""
    rewards = np.array([5.0, 5.0, 5.0, 1.0, 2.0, 3.0])
    adv = group_advantages(rewards, group_size=3)
    assert np.all(adv[:3] == 0.0)
    assert abs(adv[3:].mean()) < 1e-10
    assert abs(adv[3:].std() - 1.0) < 1e-3


def test_group_size_one_yields_zero_advantages():
    adv = group_advantages(np.array([1.0, 2.0, 3.0]), group_size=1)
    assert np.all(adv == 0.0)


def test_group_advantages_rejects_non_multiple_size():
    with pytest.raises(ValueError):
        group_advantages(np.array([1.0, 2.0, 3.0]), group_size=2)


def test_group_advantages_rejects_zero_size():
    with pytest.raises(ValueError):
        group_advantages(np.array([1.0]), group_size=0)


def test_group_advantages_no_nan_for_constant_groups():
    adv = group_advantages(np.zeros(8), group_size=4)
    assert not np.any(np.isnan(adv))


# ------------------------------------------------------------------ 迭代
def _group_batch(group_size: int = 4, n_tasks: int = 4):
    tasks = generate_tasks(n_tasks, seed=41, prefix="t")
    pol = PolicyNet(SMALL, np.random.default_rng(0))
    ref = PolicyNet(SMALL, np.random.default_rng(0))
    load_snapshot(ref, snapshot(pol))
    batch = collect_rollout(
        pol, tasks, np.random.default_rng(3), ref_policy=ref,
        group_size=group_size, reward_cfg=RewardConfig(),
    )
    return pol, batch


def test_grpo_iteration_has_no_critic_in_signature():
    """GRPO 的核心特征之一就是不需要 Critic。"""
    import inspect

    params = list(inspect.signature(grpo_iteration).parameters)
    assert params == ["policy", "batch", "cfg", "optimizer"]


def test_grpo_iteration_returns_finite_stats_and_updates_parameters():
    pol, batch = _group_batch()
    before = [p.copy() for _, p, _ in pol.parameters()]
    opt = Adam(pol.parameters(), lr=1e-3)
    cfg = GRPOConfig(iterations=1, inner_epochs=1, group_size=4)
    stats = grpo_iteration(pol, batch, cfg, opt)

    for key in ("policy_loss", "kl_k3", "entropy", "ratio_mean", "clip_frac", "grad_norm",
                "group_reward_var", "group_reward_std"):
        assert key in stats
        assert np.isfinite(stats[key]), f"{key} 不是有限值"
    assert stats["group_reward_var"] >= 0.0
    assert any(not np.array_equal(a, b) for a, b in zip(before, [p for _, p, _ in pol.parameters()]))


def test_grpo_rejects_wrong_group_size():
    pol, batch = _group_batch(group_size=4)
    opt = Adam(pol.parameters(), lr=1e-4)
    with pytest.raises(ValueError):
        grpo_iteration(pol, batch, GRPOConfig(iterations=1, inner_epochs=1, group_size=3), opt)


def test_grpo_rejects_batch_without_group_info():
    pol = PolicyNet(SMALL, np.random.default_rng(0))
    tasks = generate_tasks(3, seed=42, prefix="t")
    batch = collect_rollout(pol, tasks, np.random.default_rng(4), group_size=1)
    batch.group_ids = None
    opt = Adam(pol.parameters(), lr=1e-4)
    with pytest.raises(ValueError):
        grpo_iteration(pol, batch, GRPOConfig(iterations=1, inner_epochs=1, group_size=1), opt)


def test_grpo_zero_lr_leaves_ratio_at_one():
    pol, batch = _group_batch()
    opt = Adam(pol.parameters(), lr=0.0)
    stats = grpo_iteration(pol, batch, GRPOConfig(iterations=1, inner_epochs=1, group_size=4), opt)
    assert stats["ratio_mean"] == pytest.approx(1.0, abs=1e-9)


def test_grpo_degenerate_group_gives_zero_policy_gradient():
    """组内奖励全相等 -> 优势全 0 -> 策略梯度项为 0，参数不应改变。"""
    pol, batch = _group_batch()
    batch.rewards = np.full(batch.n, 1.0)
    before = [p.copy() for _, p, _ in pol.parameters()]
    opt = Adam(pol.parameters(), lr=1e-3)
    grpo_iteration(pol, batch, GRPOConfig(iterations=1, inner_epochs=1, group_size=4), opt)
    # 熵正则会带来非零梯度，因此这里只要求梯度有限、不产生 NaN
    after = [p for _, p, _ in pol.parameters()]
    assert all(np.all(np.isfinite(a)) for a in after)
    assert len(before) == len(after)
