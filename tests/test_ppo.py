"""PPO：GAE、优势归一化、clipped surrogate 边界、一次完整迭代。"""

from __future__ import annotations

import numpy as np
import pytest

from rlalign.env import generate_tasks
from rlalign.nn import Adam
from rlalign.policy import PolicyConfig, PolicyNet, ValueNet, load_snapshot, snapshot
from rlalign.ppo import (
    PPOConfig,
    _surrogate_dratio,
    compute_gae,
    normalize_advantages,
    ppo_iteration,
    shaped_rewards,
)
from rlalign.reward import RewardConfig
from rlalign.rollout import collect_rollout

SMALL = PolicyConfig(d_emb=8, attn_dim=8, hidden=12, value_hidden=8)


# ------------------------------------------------------------------ GAE
def test_gae_matches_hand_computed_case_1():
    """3 步手算例：γ=1.0, λ=0.95。

    δ_2 = 1 + 0 - 0.7 = 0.3                         -> A_2 = 0.3
    δ_1 = 0 + 0.7 - 0.6 = 0.1                       -> A_1 = 0.1 + 0.95*0.3   = 0.385
    δ_0 = 0 + 0.6 - 0.5 = 0.1                       -> A_0 = 0.1 + 0.95*0.385 = 0.46575
    """
    adv, ret = compute_gae(np.array([0.0, 0.0, 1.0]), np.array([0.5, 0.6, 0.7]), gamma=1.0, lam=0.95)
    assert np.allclose(adv, [0.46575, 0.385, 0.3])
    assert np.allclose(ret, [0.96575, 0.985, 1.0])


def test_gae_matches_hand_computed_case_2():
    """4 步手算例：γ=0.9, λ=0.8，价值全为 0.5。

    δ_3 = 2 + 0     - 0.5 = 1.5    -> A_3 = 1.5
    δ_2 = 0 + 0.45  - 0.5 = -0.05  -> A_2 = -0.05 + 0.72*1.5    = 1.03
    δ_1 = 0 + 0.45  - 0.5 = -0.05  -> A_1 = -0.05 + 0.72*1.03   = 0.6916
    δ_0 = 1 + 0.45  - 0.5 = 0.95   -> A_0 = 0.95  + 0.72*0.6916 = 1.447952
    """
    adv, ret = compute_gae(
        np.array([1.0, 0.0, 0.0, 2.0]), np.full(4, 0.5), gamma=0.9, lam=0.8
    )
    assert np.allclose(adv, [1.447952, 0.6916, 1.03, 1.5])
    assert np.allclose(ret, [1.947952, 1.1916, 1.53, 2.0])


def test_gae_lambda_zero_reduces_to_td_error():
    adv, _ = compute_gae(np.array([1.0, 2.0, 3.0]), np.zeros(3), gamma=1.0, lam=0.0)
    assert np.allclose(adv, [1.0, 2.0, 3.0])


def test_gae_lambda_one_gamma_one_equals_monte_carlo_minus_value():
    """λ=1, γ=1 时 A_t = (Σ_{k≥t} r_k) − V_t。"""
    adv, _ = compute_gae(np.array([1.0, 2.0, 3.0]), np.full(3, 0.5), gamma=1.0, lam=1.0)
    assert np.allclose(adv, [5.5, 4.5, 2.5])


def test_gae_terminal_step_does_not_bootstrap():
    """最后一步没有下一状态价值，V(s_{T}) 必须按 0 处理。"""
    adv, _ = compute_gae(np.array([0.0, 5.0]), np.array([9.0, 9.0]), gamma=1.0, lam=0.0)
    # δ_1 = 5 + 0 - 9 = -4；δ_0 = 0 + 9 - 9 = 0
    assert np.allclose(adv, [0.0, -4.0])


def test_gae_masks_out_padding():
    """EOS 之后的填充步不参与，也不向更早的步传播。"""
    rewards = np.array([1.0, 2.0, 3.0, 99.0])
    values = np.zeros(4)
    mask = np.array([1.0, 1.0, 1.0, 0.0])
    adv, ret = compute_gae(rewards, values, mask, gamma=1.0, lam=0.95)
    assert np.allclose(adv[:3], [5.6075, 4.85, 3.0])
    assert adv[3] == 0.0
    assert ret[3] == 0.0


def test_gae_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_gae(np.zeros(3), np.zeros(4))


# ------------------------------------------------------------------ 优势归一化
def test_normalize_advantages_zero_mean_unit_std():
    adv = np.array([0.1, -0.5, 3.0, 1.2, -2.0, 0.4])
    mask = np.ones(6)
    out = normalize_advantages(adv, mask)
    assert abs(out.mean()) < 1e-10
    # 分母是 std + eps，所以标准化后的标准差是 1 - O(eps/std)，用 1e-6 容差
    assert abs(out.std() - 1.0) < 1e-6


def test_normalize_advantages_ignores_masked_entries():
    adv = np.array([0.1, -0.5, 3.0, 1.2, 999.0])
    mask = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    out = normalize_advantages(adv, mask)
    assert out[4] == 0.0
    assert abs(out[:4].mean()) < 1e-10


def test_normalize_advantages_degenerate_input_is_all_zeros():
    """全相等的优势 -> std=0，必须返回全 0 而不是 NaN。"""
    adv = np.full(5, 0.7)
    out = normalize_advantages(adv, np.ones(5))
    assert np.all(out == 0.0)
    assert np.all(np.isfinite(out))


def test_normalize_advantages_all_masked_is_all_zeros():
    out = normalize_advantages(np.array([1.0, 2.0]), np.zeros(2))
    assert np.all(out == 0.0)


# ------------------------------------------------------------------ clip 边界
def _surrogate_objective(ratio, adv, eps):
    return -np.minimum(ratio * adv, np.clip(ratio, 1 - eps, 1 + eps) * adv)


def test_clip_objective_flat_beyond_upper_bound_for_positive_adv():
    """比值超出 1+ε 后，目标函数不再随比值变化（这就是信任域）。"""
    eps = 0.2
    a = _surrogate_objective(np.array([1.5]), np.array([1.0]), eps)[0]
    b = _surrogate_objective(np.array([3.0]), np.array([1.0]), eps)[0]
    c = _surrogate_objective(np.array([100.0]), np.array([1.0]), eps)[0]
    assert a == pytest.approx(b) == pytest.approx(c)
    assert a == pytest.approx(-1.2)


def test_clip_objective_flat_beyond_lower_bound_for_negative_adv():
    eps = 0.2
    a = _surrogate_objective(np.array([0.5]), np.array([-1.0]), eps)[0]
    b = _surrogate_objective(np.array([0.05]), np.array([-1.0]), eps)[0]
    assert a == pytest.approx(b)
    assert a == pytest.approx(0.8)


def test_surrogate_gradient_is_zero_outside_clip():
    eps = 0.2
    ratio = np.array([1.5, 3.0, 100.0])
    adv = np.array([1.0, 1.0, 1.0])
    assert np.allclose(_surrogate_dratio(ratio, adv, eps), 0.0)

    ratio_lo = np.array([0.5, 0.05])
    adv_neg = np.array([-1.0, -1.0])
    assert np.allclose(_surrogate_dratio(ratio_lo, adv_neg, eps), 0.0)


def test_surrogate_gradient_is_minus_adv_inside_clip():
    eps = 0.2
    ratio = np.array([0.95, 1.0, 1.05])
    adv = np.array([2.0, 2.0, 2.0])
    assert np.allclose(_surrogate_dratio(ratio, adv, eps), -2.0)


def test_clip_objective_rises_inside_region_then_saturates():
    """信任域内目标随 ratio 单调上升；越界后封顶不再变化。"""
    eps = 0.2
    inside = [
        -_surrogate_objective(np.array([r]), np.array([1.0]), eps)[0]
        for r in (0.8, 0.9, 1.0, 1.1, 1.2)
    ]
    assert inside == sorted(inside)

    outside = [
        -_surrogate_objective(np.array([r]), np.array([1.0]), eps)[0]
        for r in (1.2, 5.0, 50.0, 1e6)
    ]
    assert all(v == pytest.approx(outside[0]) for v in outside)


# ------------------------------------------------------------------ 奖励塑形
def _tiny_batch(group_size: int = 1):
    tasks = generate_tasks(4, seed=31, prefix="t")
    pol = PolicyNet(SMALL, np.random.default_rng(0))
    ref = PolicyNet(SMALL, np.random.default_rng(0))
    load_snapshot(ref, snapshot(pol))
    crit = ValueNet(SMALL, np.random.default_rng(1))
    batch = collect_rollout(
        pol, tasks, np.random.default_rng(2), ref_policy=ref, critic=crit,
        group_size=group_size, reward_cfg=RewardConfig(),
    )
    return pol, ref, crit, batch


def test_shaped_rewards_adds_terminal_reward_at_last_valid_step():
    _pol, _ref, _crit, batch = _tiny_batch()
    r, kl = shaped_rewards(batch, kl_coef=0.05)
    for i in range(batch.n):
        last = int(batch.lengths[i]) - 1
        expected = -0.05 * kl[i, last] + batch.rewards[i]
        assert r[i, last] == pytest.approx(expected)
        # 填充位置必须严格为 0
        assert np.all(r[i, int(batch.lengths[i]):] == 0.0)


def test_kl_is_zero_when_policy_equals_reference():
    """策略与参考策略相同时，KL 估计必须约为 0（这是 KL 实现正确的基本检查）。"""
    pol, _ref, _crit, batch = _tiny_batch()
    kl_k1 = (batch.logp_ref - batch.logp_old) * batch.mask
    assert np.allclose(kl_k1, 0.0, atol=1e-12)
    # k3 估计 exp(Δ) − Δ − 1 在 Δ=0 处也为 0
    k3 = np.exp(kl_k1) - kl_k1 - 1.0
    assert np.allclose(k3, 0.0, atol=1e-12)


def test_kl_k3_is_non_negative():
    pol, _ref, _crit, batch = _tiny_batch()
    rng = np.random.default_rng(5)
    lp = batch.logp_old + rng.normal(scale=0.3, size=batch.logp_old.shape) * batch.mask
    d = batch.logp_ref - lp
    k3 = np.exp(d) - d - 1.0
    assert np.all(k3[batch.mask > 0] >= -1e-12)


# ------------------------------------------------------------------ 完整迭代
def test_ppo_iteration_returns_finite_stats_and_updates_parameters():
    pol, _ref, crit, batch = _tiny_batch()
    before = [p.copy() for _, p, _ in pol.parameters()]
    opt = Adam(pol.parameters() + crit.parameters(), lr=1e-3)
    cfg = PPOConfig(iterations=1, inner_epochs=1, prompts_per_iter=4)
    stats = ppo_iteration(pol, crit, batch, cfg, opt)

    for key in ("policy_loss", "value_loss", "entropy", "kl_k3", "kl_k1", "ratio_mean",
                "clip_frac", "grad_norm"):
        assert key in stats
        assert np.isfinite(stats[key]), f"{key} 不是有限值"
    assert stats["ratio_mean"] > 0.0
    assert 0.0 <= stats["clip_frac"] <= 1.0
    assert any(not np.array_equal(a, b) for a, b in zip(before, [p for _, p, _ in pol.parameters()]))


def test_ppo_ratio_is_one_on_first_epoch():
    """第一次更新前 ratio 必须严格为 1（策略还没变）。"""
    pol, _ref, crit, batch = _tiny_batch()
    opt = Adam(pol.parameters() + crit.parameters(), lr=0.0)  # lr=0 -> 参数不动
    cfg = PPOConfig(iterations=1, inner_epochs=1, prompts_per_iter=4)
    stats = ppo_iteration(pol, crit, batch, cfg, opt)
    assert stats["ratio_mean"] == pytest.approx(1.0, abs=1e-9)
    assert stats["clip_frac"] == pytest.approx(0.0, abs=1e-12)
