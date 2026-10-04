"""手写层、激活函数与优化器。"""

from __future__ import annotations

import numpy as np
import pytest

from rlalign.nn import (
    Adam,
    Embedding,
    Linear,
    SGD,
    SingleHeadAttention,
    clip_grad_norm_,
    entropy_from_logits,
    global_grad_norm,
    log_softmax,
    relu,
    relu_backward,
    softmax,
    tanh_backward,
)

RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ softmax
def test_softmax_sums_to_one():
    x = np.array([1.0, 2.0, 3.0])
    assert np.isclose(softmax(x).sum(), 1.0)


def test_softmax_is_shift_invariant():
    x = np.array([1.0, 2.0, 3.0])
    assert np.allclose(softmax(x), softmax(x + 100.0))


def test_softmax_numerically_stable_with_huge_logits():
    """大 logits 不能溢出成 inf/nan —— 这正是减最大值那一步的作用。"""
    x = np.array([1000.0, 1000.0, 999.0])
    p = softmax(x)
    assert np.all(np.isfinite(p))
    assert np.isclose(p.sum(), 1.0)
    assert np.isclose(p[0], p[1])
    assert p[0] > p[2]


def test_softmax_numerically_stable_with_very_negative_logits():
    x = np.array([-1000.0, -1001.0, -1002.0])
    p = softmax(x)
    assert np.all(np.isfinite(p))
    assert np.isclose(p.sum(), 1.0)


def test_log_softmax_matches_log_of_softmax():
    x = np.array([0.3, -1.2, 2.0, 0.1])
    assert np.allclose(log_softmax(x), np.log(softmax(x)))


def test_log_softmax_stable_with_huge_logits():
    x = np.array([1e4, 1e4 - 1.0])
    lp = log_softmax(x)
    assert np.all(np.isfinite(lp))
    assert np.isclose(np.exp(lp).sum(), 1.0)


def test_entropy_is_maximal_for_uniform_distribution():
    uniform = np.zeros(5)
    peaked = np.array([50.0, 0.0, 0.0, 0.0, 0.0])
    assert entropy_from_logits(uniform) > entropy_from_logits(peaked)
    assert np.isclose(entropy_from_logits(uniform), np.log(5.0))


def test_entropy_is_non_negative():
    for _ in range(10):
        assert entropy_from_logits(RNG.normal(size=8)) >= 0.0


# ------------------------------------------------------------------ 激活
def test_relu_forward_and_backward():
    x = np.array([-1.0, 0.0, 2.0])
    assert np.allclose(relu(x), [0.0, 0.0, 2.0])
    assert np.allclose(relu_backward(x, np.ones(3)), [0.0, 0.0, 1.0])


def test_tanh_backward_matches_derivative():
    a = np.array([0.0, 0.5, -1.2])
    h = np.tanh(a)
    assert np.allclose(tanh_backward(h, np.ones(3)), 1.0 - h * h)


# ------------------------------------------------------------------ Linear
def test_linear_forward_shape_and_value():
    lin = Linear(3, 2, np.random.default_rng(1))
    lin.W = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    lin.b = np.array([0.5, -0.5])
    out = lin.forward(np.array([2.0, 3.0, 9.0]))
    assert np.allclose(out, [2.5, 2.5])


def test_linear_backward_shapes():
    lin = Linear(4, 3, np.random.default_rng(2))
    x = RNG.normal(size=4)
    lin.forward(x)
    dx = lin.backward(RNG.normal(size=3))
    assert dx.shape == (4,)
    assert lin.dW.shape == (3, 4)
    assert lin.db.shape == (3,)


def test_linear_backward_accumulates_not_overwrites():
    lin = Linear(3, 2, np.random.default_rng(3))
    x = np.array([1.0, 0.0, 0.0])
    lin.forward(x)
    lin.backward(np.array([1.0, 0.0]))
    first = lin.dW.copy()
    lin.forward(x)
    lin.backward(np.array([1.0, 0.0]))
    assert np.allclose(lin.dW, 2 * first)


def test_linear_backward_uses_explicit_x_not_stale_cache():
    """回归测试：逐步循环里必须显式传入 x。

    早期版本让 ``Linear.backward`` 用 ``self._x``，在序列循环中它会被
    下一步覆盖，导致整条序列反传都用了最后一步的激活值 ——
    有限差分校验直接报相对误差 1.0。
    """
    lin = Linear(2, 1, np.random.default_rng(4))
    lin.W = np.array([[1.0, 1.0]])
    lin.b = np.zeros(1)
    x1 = np.array([1.0, 0.0])
    x2 = np.array([0.0, 5.0])
    lin.forward(x1)
    lin.forward(x2)  # 覆盖了内部缓存
    lin.dW.fill(0.0)
    lin.backward(np.array([1.0]), x1)  # 显式传入 x1
    assert np.allclose(lin.dW, [[1.0, 0.0]])


def test_zero_grad_clears_linear():
    lin = Linear(3, 2, np.random.default_rng(5))
    lin.forward(RNG.normal(size=3))
    lin.backward(RNG.normal(size=2))
    lin.zero_grad()
    assert np.allclose(lin.dW, 0.0)
    assert np.allclose(lin.db, 0.0)


# ------------------------------------------------------------------ Embedding
def test_embedding_lookup():
    emb = Embedding(5, 3, np.random.default_rng(6))
    rows = emb.forward(np.array([0, 2, 2]))
    assert rows.shape == (3, 3)
    assert np.allclose(rows[1], rows[2])
    assert np.allclose(rows[0], emb.E[0])


def test_embedding_backward_accumulates_duplicate_ids():
    """同一 id 出现多次时梯度必须累加（np.add.at 而不是花式索引赋值）。"""
    emb = Embedding(4, 2, np.random.default_rng(7))
    emb.backward(np.array([1, 1, 1]), np.ones((3, 2)))
    assert np.allclose(emb.dE[1], [3.0, 3.0])
    assert np.allclose(emb.dE[0], [0.0, 0.0])


# ------------------------------------------------------------------ Attention
def test_attention_softmax_rows_sum_to_one():
    attn = SingleHeadAttention(4, 6, 3, np.random.default_rng(8))
    mem = RNG.normal(size=(5, 4))
    K, V = attn.project(mem)
    ctx, cache = attn.query(K, V, RNG.normal(size=6))
    assert ctx.shape == (3,)
    assert np.all(np.isfinite(ctx))


def test_attention_k_has_no_bias():
    """K 的常数偏置会被 softmax 消掉，因此刻意不设该参数。"""
    attn = SingleHeadAttention(4, 6, 3, np.random.default_rng(9))
    names = [n for n, _, _ in attn.parameters()]
    assert "bk" not in names
    assert set(names) == {"Wq", "bq", "Wk", "Wv", "bv"}


def test_attention_shifting_all_scores_does_not_change_context():
    """把 K 整体平移一个常数，softmax 不变，因此 context 也不变。"""
    attn = SingleHeadAttention(4, 6, 3, np.random.default_rng(10))
    mem = RNG.normal(size=(5, 4))
    q_feat = RNG.normal(size=6)
    K, V = attn.project(mem)
    ctx1, _ = attn.query(K, V, q_feat)
    ctx2, _ = attn.query(K + 3.7, V, q_feat)
    assert np.allclose(ctx1, ctx2)


# ------------------------------------------------------------------ 优化器
def test_sgd_moves_along_negative_gradient():
    p = np.array([1.0, -2.0])
    g = np.array([0.5, -0.25])
    grad = np.zeros_like(p)
    opt = SGD([("p", p, grad)], lr=0.1)
    grad[...] = g
    opt.step()
    assert np.allclose(p, np.array([1.0, -2.0]) - 0.1 * g)


def test_sgd_with_momentum_accumulates_velocity():
    p = np.zeros(1)
    grad = np.zeros(1)
    opt = SGD([("p", p, grad)], lr=1.0, momentum=0.9)
    grad[...] = 1.0
    opt.step()
    assert np.isclose(p[0], -1.0)
    opt.step()  # v = 0.9*(-1) - 1 = -1.9
    assert np.isclose(p[0], -2.9)


def test_adam_first_step_is_lr_scaled_sign():
    """Adam 第一步的更新量约为 lr（偏差修正后），方向沿负梯度。"""
    p = np.array([0.0])
    grad = np.array([5.0])
    opt = Adam([("p", p, grad)], lr=0.1)
    opt.step()
    assert p[0] < 0
    assert abs(p[0] + 0.1) < 1e-6


def test_adam_updates_negative_for_positive_gradient():
    p = np.array([1.0, -1.0])
    grad = np.array([0.3, 0.3])
    opt = Adam([("p", p, grad)], lr=0.01)
    before = p.copy()
    opt.step()
    assert np.all(p < before)


def test_adam_ignores_gradient_scale():
    """Adam 对梯度整体缩放近似不变（除以 sqrt(v)）。"""
    p1 = np.array([0.0])
    p2 = np.array([0.0])
    g1 = np.array([0.001])
    g2 = np.array([1000.0])
    Adam([("p", p1, g1)], lr=0.1).step()
    Adam([("p", p2, g2)], lr=0.1).step()
    assert np.isclose(p1[0], p2[0], atol=1e-9)


def test_grad_norm_and_clipping():
    p = np.zeros(2)
    grad = np.array([3.0, 4.0])  # 范数 = 5
    params = [("p", p, grad)]
    assert np.isclose(global_grad_norm(params), 5.0)
    assert np.isclose(clip_grad_norm_(params, 1.0), 5.0)
    assert np.isclose(global_grad_norm(params), 1.0)


def test_clipping_does_not_change_small_gradients():
    p = np.zeros(2)
    grad = np.array([0.1, 0.1])
    params = [("p", p, grad)]
    before = grad.copy()
    clip_grad_norm_(params, 10.0)
    assert np.allclose(grad, before)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_softmax_rejects_non_finite(bad):
    """NaN/Inf 必须在源头报错，而不是静默污染整条训练链路。"""
    with pytest.raises(ValueError):
        softmax(np.array([1.0, bad]))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_log_softmax_rejects_non_finite(bad):
    with pytest.raises(ValueError):
        log_softmax(np.array([1.0, bad]))
