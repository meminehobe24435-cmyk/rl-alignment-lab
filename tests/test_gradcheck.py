"""数值梯度校验：证明手推的反向传播是对的（本项目最重要的正确性证据）。"""

from __future__ import annotations

import numpy as np

from rlalign.gradcheck import (
    ABS_TOL,
    REL_FLOOR,
    REL_TOL,
    check_policy_gradients,
    check_value_gradients,
    numeric_grad,
    relative_error,
)
from rlalign.policy import PolicyConfig

# 用小网络跑校验，保证测试足够快（结构与真实配置完全一致，只是维度更小）
SMALL = PolicyConfig(d_emb=6, attn_dim=6, hidden=10, value_hidden=8)


def test_numeric_grad_matches_analytic_for_quadratic():
    """对 f(x)=Σx² 做中心差分，解析梯度是 2x。"""
    x = np.array([0.3, -1.2, 2.5])

    def f(v):
        return float(np.sum(v * v))

    g = numeric_grad(f, x, eps=1e-6)
    assert np.allclose(g, 2 * x, atol=1e-6)


def test_numeric_grad_matches_analytic_for_softmax_cross_entropy():
    from rlalign.nn import log_softmax

    z = np.array([0.5, -1.0, 2.0])
    target = 2

    def f(v):
        return -float(log_softmax(v)[target])

    g = numeric_grad(f, z, eps=1e-6)
    p = np.exp(log_softmax(z))
    expected = p.copy()
    expected[target] -= 1.0
    assert np.allclose(g, expected, atol=1e-7)


def test_relative_error_is_zero_for_identical():
    a = np.array([1.0, -2.0, 0.0])
    assert np.all(relative_error(a, a) == 0.0)


def test_relative_error_handles_both_zero():
    assert relative_error(np.array([0.0]), np.array([0.0]))[0] == 0.0


def test_relative_error_detects_discrepancy():
    # 分母是 |a| + |n|，所以 1 与 2 的相对误差是 1/3 而不是 1/2
    assert relative_error(np.array([1.0]), np.array([2.0]))[0] == 1.0 / 3.0
    assert relative_error(np.array([1.0]), np.array([-1.0]))[0] == 1.0


def test_policy_gradients_pass_finite_difference():
    """策略网络（含单头注意力与 tanh 编码器）的全参数梯度校验。"""
    r = check_policy_gradients(SMALL, seed=0, eps=1e-6, n_per_param=6)
    assert r["n_checks"] > 0
    assert r["n_significant"] > 0
    assert r["max_rel_err"] < REL_TOL, f"策略网络梯度校验失败: {r['worst']}"
    assert r["max_abs_err_of_small_grads"] < ABS_TOL
    assert r["passed"] is True


def test_critic_gradients_pass_finite_difference():
    r = check_value_gradients(SMALL, seed=0, eps=1e-6, n_per_param=6)
    assert r["n_checks"] > 0
    assert r["max_rel_err"] < REL_TOL, f"Critic 梯度校验失败: {r['worst']}"
    assert r["max_abs_err_of_small_grads"] < ABS_TOL
    assert r["passed"] is True


def test_policy_gradients_pass_with_different_seeds():
    """换随机初始化再验一次，排除"恰好这一组参数对"的可能。"""
    for seed in (1, 2, 3):
        r = check_policy_gradients(SMALL, seed=seed, eps=1e-6, n_per_param=4)
        assert r["passed"] is True, f"seed={seed} 失败: {r['worst']}"


def test_critic_gradients_pass_with_different_seeds():
    for seed in (1, 2):
        r = check_value_gradients(SMALL, seed=seed, eps=1e-6, n_per_param=4)
        assert r["passed"] is True, f"seed={seed} 失败: {r['worst']}"


def test_gradient_check_reports_per_parameter_records():
    r = check_policy_gradients(SMALL, seed=0, n_per_param=3)
    assert len(r["checks"]) == r["n_checks"]
    for rec in r["checks"][:5]:
        assert set(rec) >= {"param", "index", "analytic", "numeric", "rel_err", "abs_err"}
        assert np.isfinite(rec["analytic"])
        assert np.isfinite(rec["numeric"])


def test_gradient_check_covers_all_parameter_tensors():
    """每个参数张量都必须被抽样到，不能漏掉某一层。"""
    from rlalign.policy import PolicyNet

    pol = PolicyNet(SMALL, np.random.default_rng(0))
    names = {n for n, _, _ in pol.parameters()}
    r = check_policy_gradients(SMALL, seed=0, n_per_param=2)
    checked = {rec["param"] for rec in r["checks"]}
    assert checked == names


def test_gradient_check_does_not_mutate_parameters():
    """有限差分扰动后必须把参数恢复原样。"""
    from rlalign.policy import PolicyNet

    pol = PolicyNet(SMALL, np.random.default_rng(7))
    before = [p.copy() for _, p, _ in pol.parameters()]
    check_policy_gradients(SMALL, seed=7, n_per_param=3)
    # 重新构造同种子的网络，参数应与运行校验时未受污染的一致
    pol2 = PolicyNet(SMALL, np.random.default_rng(7))
    after = [p for _, p, _ in pol2.parameters()]
    assert all(np.array_equal(a, b) for a, b in zip(before, after))


def test_rel_floor_is_above_finite_difference_noise():
    """判据自检：相对误差只在梯度足够大时才可信。"""
    assert REL_FLOOR > 0.0
    assert ABS_TOL > 1e-11  # 必须高于中心差分的绝对噪声 ~1e-10
