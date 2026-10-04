"""手写神经网络层、激活函数与优化器（前向 + 手推反向传播）。

设计约定
--------
* 每个层对象自己持有参数（numpy 数组）与同形状的梯度累加器；
  ``backward`` **累加**梯度而不是覆盖，因此需要先调用 :meth:`zero_grad`。
* 参数统一通过 ``parameters()`` 暴露为 ``(name, value, grad)`` 三元组，
  优化器只依赖这个接口，新增层不需要改优化器。
* 所有反向传播公式都是手工推导的（见每个方法的 docstring），
  并由 ``tools/check_gradients.py`` 与 ``tests/test_gradcheck.py``
  用中心差分做数值校验。

这里没有 autograd，没有任何计算图引擎：每个 ``backward`` 里的矩阵乘法
就是链式法则展开后的结果。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "Linear",
    "Embedding",
    "SingleHeadAttention",
    "relu",
    "relu_backward",
    "tanh_backward",
    "softmax",
    "log_softmax",
    "entropy_from_logits",
    "logprob_from_logits",
    "SGD",
    "Adam",
    "global_grad_norm",
    "clip_grad_norm_",
    "zero_grads",
]


# --------------------------------------------------------------------------
# 层
# --------------------------------------------------------------------------
class Linear:
    """全连接层 ``y = W x + b``，W 形状 ``(n_out, n_in)``。

    反向：
        dL/dW = dy ⊗ x        （外积）
        dL/db = dy
        dL/dx = Wᵀ dy
    """

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator, gain: float = 1.0):
        bound = gain / np.sqrt(float(n_in))
        self.W = rng.uniform(-bound, bound, size=(n_out, n_in))
        self.b = np.zeros(n_out, dtype=np.float64)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.n_in = n_in
        self.n_out = n_out
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: ``(n_in,)`` -> ``(n_out,)``。"""
        self._x = x
        return self.W @ x + self.b

    def backward(self, dy: np.ndarray, x: np.ndarray | None = None) -> np.ndarray:
        """``dL/dx = Wᵀ dy``，同时累加 ``dL/dW = dy ⊗ x``、``dL/db = dy``。

        参数 ``x`` 必须由调用方在**序列循环**里显式传入：``self._x`` 只保存
        "最近一次 forward"的输入，逐步循环时会被下一步覆盖。
        早期版本依赖 ``self._x``，导致整条序列反传时全部用了最后一步的激活值，
        被有限差分校验直接抓出来（相对误差 = 1.0）。
        """
        if x is None:
            if self._x is None:
                raise RuntimeError("backward 之前必须先调用 forward，或显式传入 x")
            x = self._x
        self.dW += np.outer(dy, x)
        self.db += dy
        return self.W.T @ dy

    def parameters(self):
        return [("W", self.W, self.dW), ("b", self.b, self.db)]

    def zero_grad(self) -> None:
        self.dW.fill(0.0)
        self.db.fill(0.0)


class Embedding:
    """查表嵌入：``forward(ids) -> E[ids]``。

    反向是对被查到的行做 scatter-add（同一 id 出现多次时梯度要累加，
    所以用 ``np.add.at`` 而不是花式索引赋值）。
    """

    def __init__(self, n_vocab: int, dim: int, rng: np.random.Generator, scale: float = 1.0):
        self.E = rng.normal(0.0, scale / np.sqrt(float(dim)), size=(n_vocab, dim))
        self.dE = np.zeros_like(self.E)
        self.n_vocab = n_vocab
        self.dim = dim

    def forward(self, ids) -> np.ndarray:
        ids = np.asarray(ids, dtype=np.int64)
        return self.E[ids]

    def backward(self, ids, dout: np.ndarray) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        np.add.at(self.dE, ids, dout)

    def parameters(self):
        return [("E", self.E, self.dE)]

    def zero_grad(self) -> None:
        self.dE.fill(0.0)


class SingleHeadAttention:
    """单头点积注意力（手推前向 + 反向）。

    前向::

        K = mem W_kᵀ                # (P, d_k)   注意：K 没有偏置项
        V = mem W_vᵀ + b_v          # (P, d_v)
        q = W_q x_q + b_q           # (d_k,)
        s = K q / √d_k              # (P,)
        α = softmax(s)              # (P,)
        ctx = α V                   # (d_v,)

    **为什么 K 没有偏置**：给 K 加一个常数偏置 ``b_k`` 会让
    ``s_i = (mem_i W_kᵀ + b_k)·q/√d_k`` 多出一个与位置 i 无关的常数项，
    而 softmax 对常数平移不变 —— 因此 ``∂L/∂b_k ≡ 0``，是个纯废参数。
    这个结论是数值梯度校验逼出来的：``b_k`` 的解析梯度恒为 0、
    数值梯度只剩浮点噪声，相对误差直接被判 1.0。
    ``b_v`` 不会被 softmax 消掉（它直接进入加权和），所以保留。

    反向（给定 ``dctx``）::

        dV  = αᵀ dctx                       (P, d_v)   外积累加
        dα  = V dctx                        (P,)
        ds  = α ⊙ (dα − ⟨α, dα⟩)            softmax 的雅可比
        dK  = (ds ⊗ q) / √d_k
        dq  = Kᵀ ds / √d_k
        dW_q = dq ⊗ x_q,  db_q = dq
        dW_k = dK ᵀ mem,  dW_v = dV ᵀ mem,  db_v = Σ dV
        dmem = dK W_k + dV W_v

    ``K``/``V`` 在同一序列的每一步都被复用，因此 ``dK``/``dV`` 由调用方
    在整条序列上累加，最后一次性通过 :meth:`backward_memory` 回传到参数。
    这样避免了"层内部缓存被下一步覆盖"的经典错误。
    """

    def __init__(self, d_mem: int, d_q: int, d_k: int, rng: np.random.Generator, gain: float = 1.0):
        self.d_mem = d_mem
        self.d_q = d_q
        self.d_k = d_k
        self.Wq = rng.uniform(-gain / np.sqrt(d_q), gain / np.sqrt(d_q), size=(d_k, d_q))
        self.bq = np.zeros(d_k)
        self.Wk = rng.uniform(-gain / np.sqrt(d_mem), gain / np.sqrt(d_mem), size=(d_k, d_mem))
        self.Wv = rng.uniform(-gain / np.sqrt(d_mem), gain / np.sqrt(d_mem), size=(d_k, d_mem))
        self.bv = np.zeros(d_k)
        self.dWq = np.zeros_like(self.Wq)
        self.dbq = np.zeros_like(self.bq)
        self.dWk = np.zeros_like(self.Wk)
        self.dWv = np.zeros_like(self.Wv)
        self.dbv = np.zeros_like(self.bv)

    def project(self, mem: np.ndarray):
        """把记忆（提示的逐位置嵌入）投影成 K、V。"""
        K = mem @ self.Wk.T
        V = mem @ self.Wv.T + self.bv
        return K, V

    def query(self, K: np.ndarray, V: np.ndarray, q_feat: np.ndarray):
        """给定 K、V 与查询特征，返回 ``(ctx, cache)``。"""
        q = self.Wq @ q_feat + self.bq
        s = (K @ q) / np.sqrt(float(self.d_k))
        a = softmax(s)
        ctx = a @ V
        return ctx, (q_feat, q, a)

    def backward_query(self, dctx: np.ndarray, cache, K: np.ndarray, V: np.ndarray):
        """返回 ``(dq_feat, dK, dV)``；参数梯度不在这里累加。"""
        q_feat, q, a = cache
        dV = np.outer(a, dctx)
        da = V @ dctx
        ds = a * (da - float(a @ da))
        scale = 1.0 / np.sqrt(float(self.d_k))
        dK = np.outer(ds, q) * scale
        dq = (K.T @ ds) * scale
        self.dWq += np.outer(dq, q_feat)
        self.dbq += dq
        return self.Wq.T @ dq, dK, dV

    def backward_memory(self, dK: np.ndarray, dV: np.ndarray, mem: np.ndarray) -> np.ndarray:
        """把整条序列累加好的 ``dK``/``dV`` 回传到参数与记忆表示。"""
        self.dWk += dK.T @ mem
        self.dWv += dV.T @ mem
        self.dbv += dV.sum(axis=0)
        return dK @ self.Wk + dV @ self.Wv

    def parameters(self):
        return [
            ("Wq", self.Wq, self.dWq),
            ("bq", self.bq, self.dbq),
            ("Wk", self.Wk, self.dWk),
            ("Wv", self.Wv, self.dWv),
            ("bv", self.bv, self.dbv),
        ]

    def zero_grad(self) -> None:
        for _, _, g in self.parameters():
            g.fill(0.0)


# --------------------------------------------------------------------------
# 激活函数
# --------------------------------------------------------------------------
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def relu_backward(x: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """ReLU 反向：梯度只在 ``x > 0`` 处通过（x=0 处次梯度取 0）。"""
    return dy * (x > 0.0)


def tanh_backward(h: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """tanh 反向：``∂tanh(a)/∂a = 1 − tanh(a)²``，成立才最省事。"""
    return dy * (1.0 - h * h)


def _check_finite(x: np.ndarray, name: str) -> np.ndarray:
    """拒绝非有限输入。

    NaN/Inf 一旦进入 softmax 就会静默传播：梯度变成 NaN、参数被污染，
    而训练循环只会在几个 epoch 之后表现为"loss 突然变成 nan"，
    极难定位。这里在源头直接报错，把问题挡在第一次出现的地方。
    """
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} 输入包含 NaN/Inf，请检查上游计算")
    return x


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 softmax：先减去每行的最大值再取指数。

    不做这一步时，logits=1000 会让 ``exp`` 溢出成 inf，除法得到 nan。
    """
    x = _check_finite(np.asarray(x, dtype=np.float64), "softmax")
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数值稳定的 log-softmax：``x - max - log(sum(exp(x - max)))``。"""
    x = _check_finite(np.asarray(x, dtype=np.float64), "log_softmax")
    shifted = x - np.max(x, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def logprob_from_logits(logits: np.ndarray, idx: int) -> float:
    """给定 logits 取单个 token 的对数概率。"""
    return float(log_softmax(logits)[idx])


def entropy_from_logits(logits: np.ndarray) -> float:
    """分类分布的熵 ``H = -Σ p log p``，用 log-softmax 计算以保证稳定。

    恒等式：``H = logsumexp(z) - Σ p z``。使用 ``-Σ p log p`` 在 p=0 处
    会出现 ``0 * -inf = nan``，因此这里走 logsumexp 形式。
    """
    logits = np.asarray(logits, dtype=np.float64)
    lp = log_softmax(logits)
    p = np.exp(lp)
    return float(-np.sum(p * lp))


# --------------------------------------------------------------------------
# 优化器
# --------------------------------------------------------------------------
def zero_grads(params) -> None:
    for _, _, g in params:
        g.fill(0.0)


def global_grad_norm(params) -> float:
    total = 0.0
    for _, _, g in params:
        total += float(np.sum(np.square(g)))
    return float(np.sqrt(total))


def clip_grad_norm_(params, max_norm: float) -> float:
    """按全局 L2 范数裁剪梯度（原地），返回裁剪前的范数。"""
    norm = global_grad_norm(params)
    if max_norm > 0.0 and norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for _, _, g in params:
            g *= scale
    return norm


class SGD:
    """带动量的随机梯度下降。

    ``v <- momentum * v - lr * g``;  ``p <- p + v``
    """

    def __init__(self, params, lr: float = 1e-2, momentum: float = 0.0):
        self.params = list(params)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self._v = [np.zeros_like(p) for _, p, _ in self.params]
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1
        for i, (_, p, g) in enumerate(self.params):
            self._v[i] = self.momentum * self._v[i] - self.lr * g
            p += self._v[i]


class Adam:
    """Adam 优化器（Kingma & Ba, 2015）。

    ``m <- β1 m + (1-β1) g``
    ``v <- β2 v + (1-β2) g²``
    ``m̂ = m / (1-β1ᵗ)``, ``v̂ = v / (1-β2ᵗ)``
    ``p <- p - lr * m̂ / (√v̂ + ε)``
    """

    def __init__(self, params, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.params = list(params)
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self._m = [np.zeros_like(p) for _, p, _ in self.params]
        self._v = [np.zeros_like(p) for _, p, _ in self.params]
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1
        b1, b2 = self.beta1, self.beta2
        bc1 = 1.0 - b1 ** self.step_count
        bc2 = 1.0 - b2 ** self.step_count
        for i, (_, p, g) in enumerate(self.params):
            self._m[i] = b1 * self._m[i] + (1.0 - b1) * g
            self._v[i] = b2 * self._v[i] + (1.0 - b2) * np.square(g)
            m_hat = self._m[i] / bc1
            v_hat = self._v[i] / bc2
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
