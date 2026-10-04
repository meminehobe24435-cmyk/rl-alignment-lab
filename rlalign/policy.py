"""小型自回归策略网络：带注意力条件的微型编码器-解码器（手推前向 + 反向传播）。

架构
----
**编码器**（"提示" = 带 bug 的表达式）::

    E       = emb[prompt_ids] + pos_prompt[0..P-1]        # (P, d)  逐位置嵌入
    h0      = tanh(W_enc · flatten(E) + b_enc)            # (d,)    全局条件向量
    K, V    = attention.project(E)                        # (P, d_k), (P, d_v)

**解码器**（逐 token 自回归，单隐层 MLP 条件语言模型）::

    q_t     = W_q [h0 ; pos_gen[t]] + b_q                 # (d_k,)
    α_t     = softmax(K q_t / √d_k)                       # (P,)    对提示的注意力
    ctx_t   = α_t V                                       # (d_v,)
    x_t     = [ h0 ; emb[y_{t-1}] ; pos_gen[t] ; ctx_t ]  # (3d + d_v,)
    a_t     = W1 x_t + b1
    h_t     = relu(a_t)
    z_t     = W2 h_t + b2
    p_t     = softmax(z_t)

为什么需要注意力
----------------
最早的两版都失败了（见 README 踩坑记录）：

1. ``h0 = mean(E)``：信息瓶颈太严重，模型根本读不出提示里哪个字符是错的；
2. ``h0 = tanh(W·flatten(E))``：条件向量信息够了，SFT 能在训练集上刷到 86.7%，
   但**评测集恒为 0** —— 模型只是把 120 条 (提示 -> 目标) 背下来了，
   解码时没有任何机制在"当前位置"回头看提示，学不会"复制 + 局部改写"。

加上对提示的注意力之后，解码器每一步都能直接看到整个带 bug 的表达式，
"把第 4 个字符的 ``*`` 改成 ``+``" 这类操作才真正可学。

反向传播
--------
除注意力部分（见 :class:`rlalign.nn.SingleHeadAttention`）外：

* ``dW2 += dz ⊗ h_t``, ``db2 += dz``, ``dh_t = W2ᵀ dz``
* ``da_t = dh_t ⊙ 1[a_t > 0]``
* ``dW1 += da_t ⊗ x_t``, ``db1 += da_t``, ``dx_t = W1ᵀ da_t``
* ``dx_t`` 切成 4 段：``dh0`` / ``demb`` / ``dpos_gen[t]`` / ``dctx_t``
* ``dctx_t`` 走注意力的反向，得到 ``dq_feat``（回传到 ``pos_gen[t]``）与 ``dK/dV``
* ``dh0`` 在整条序列上求和后过 ``tanh`` 的导数，再回传到 ``W_enc`` 与 ``E``
* ``E`` 同时被"编码器"和"注意力记忆"两条路径使用，两路梯度**相加**后
  一次性 scatter-add 回 token 嵌入表与位置嵌入表

所有步骤都由 ``tests/test_gradcheck.py`` 用中心差分逐参数核对。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import BOS_ID, EOS_ID, MAX_GEN_TOKENS, MAX_PROMPT_TOKENS, VOCAB_SIZE
from .nn import (
    Embedding,
    Linear,
    SingleHeadAttention,
    log_softmax,
    relu,
    relu_backward,
    softmax,
    tanh_backward,
)

__all__ = [
    "PolicyConfig",
    "PolicyNet",
    "ValueNet",
    "snapshot",
    "load_snapshot",
    "entropy_grad",
    "param_count_summary",
]


@dataclass
class PolicyConfig:
    """网络与序列的尺寸配置。"""

    vocab_size: int = VOCAB_SIZE
    prompt_len: int = MAX_PROMPT_TOKENS
    max_gen_len: int = MAX_GEN_TOKENS
    d_emb: int = 32
    attn_dim: int = 32
    hidden: int = 128
    value_hidden: int = 64
    emb_scale: float = 1.0

    @property
    def x_dim(self) -> int:
        """解码器每步的输入维度：``[h0 ; e_prev ; pos_gen ; ctx]``。"""
        return 3 * self.d_emb + self.attn_dim


def entropy_grad(logits: np.ndarray) -> np.ndarray:
    """softmax 分布熵对 logits 的梯度。

    推导：``H = -Σ_j p_j log p_j``，``dp_j/dz_i = p_j(δ_ij - p_i)``，代入化简得
    ``dH/dz_i = -p_i (log p_i + H)``。
    """
    lp = log_softmax(logits)
    p = np.exp(lp)
    H = float(-np.sum(p * lp))
    return -p * (lp + H)


class PolicyNet:
    """条件语言模型策略 ``π_θ(y_t | prompt, y_<t)``。"""

    def __init__(self, cfg: PolicyConfig, rng: np.random.Generator):
        self.cfg = cfg
        V, d = cfg.vocab_size, cfg.d_emb
        self.emb = Embedding(V, d, rng, scale=cfg.emb_scale)
        self.pos_prompt = Embedding(cfg.prompt_len, d, rng, scale=cfg.emb_scale)
        self.pos_gen = Embedding(cfg.max_gen_len, d, rng, scale=cfg.emb_scale)
        self.enc = Linear(cfg.prompt_len * d, d, rng)
        self.attn = SingleHeadAttention(d, 2 * d, cfg.attn_dim, rng)
        self.lin1 = Linear(cfg.x_dim, cfg.hidden, rng)
        self.lin2 = Linear(cfg.hidden, V, rng)
        self._prompt_ids = np.arange(cfg.prompt_len)

    # -- 参数 ------------------------------------------------------------
    def parameters(self):
        return (
            self.emb.parameters()
            + self.pos_prompt.parameters()
            + self.pos_gen.parameters()
            + self.enc.parameters()
            + self.attn.parameters()
            + self.lin1.parameters()
            + self.lin2.parameters()
        )

    def zero_grad(self) -> None:
        for layer in (self.emb, self.pos_prompt, self.pos_gen, self.enc, self.attn, self.lin1, self.lin2):
            layer.zero_grad()

    def num_parameters(self) -> int:
        return int(sum(p.size for _, p, _ in self.parameters()))

    # -- 前向 ------------------------------------------------------------
    def encode_nograd(self, prompt_ids: np.ndarray):
        """不带缓存地把提示编码成 ``(h0, K, V)``（用于采样 / 评估）。"""
        E = self.emb.E[prompt_ids] + self.pos_prompt.E[self._prompt_ids]
        h0 = np.tanh(self.enc.W @ E.reshape(-1) + self.enc.b)
        K, V = self.attn.project(E)
        return h0, K, V

    def prompt_vector_nograd(self, prompt_ids: np.ndarray) -> np.ndarray:
        """只要条件向量 ``h0``（critic 的输入）。"""
        return self.encode_nograd(prompt_ids)[0]

    def _step_nograd(self, h0: np.ndarray, K: np.ndarray, V: np.ndarray, prev: int, t: int) -> np.ndarray:
        """不带缓存的一步解码（采样 / 贪心 / 评估用）。"""
        p_gen = self.pos_gen.E[t]
        q = self.attn.Wq @ np.concatenate([h0, p_gen]) + self.attn.bq
        a = softmax((K @ q) / np.sqrt(float(self.attn.d_k)))
        ctx = a @ V
        x = np.concatenate([h0, self.emb.E[prev], p_gen, ctx])
        return self.lin2.W @ relu(self.lin1.W @ x + self.lin1.b) + self.lin2.b

    def _step_forward(self, h0, K, V, prev: int, t: int):
        """带缓存的一步解码，返回 ``(logits, cache)``。"""
        e_prev = self.emb.forward(np.array([prev], dtype=np.int64))[0]
        p_gen = self.pos_gen.forward(np.array([t], dtype=np.int64))[0]
        ctx, attn_cache = self.attn.query(K, V, np.concatenate([h0, p_gen]))
        x = np.concatenate([h0, e_prev, p_gen, ctx])
        a1 = self.lin1.forward(x)
        h1 = relu(a1)
        logits = self.lin2.forward(h1)
        return logits, (x, a1, h1, int(prev), int(t), attn_cache)

    def forward_sequence(self, prompt_ids: np.ndarray, token_ids: np.ndarray):
        """整条生成序列的前向，返回 ``(logits (T,V), caches, seq_cache)``。"""
        T = len(token_ids)
        d = self.cfg.d_emb
        ids = self._prompt_ids
        E = self.emb.forward(prompt_ids) + self.pos_prompt.forward(ids)
        flat = E.reshape(-1)
        a_enc = self.enc.forward(flat)
        h0 = np.tanh(a_enc)
        K, V = self.attn.project(E)

        logits = np.empty((T, self.cfg.vocab_size), dtype=np.float64)
        caches = []
        for t in range(T):
            prev = BOS_ID if t == 0 else int(token_ids[t - 1])
            lg, cache = self._step_forward(h0, K, V, prev, t)
            logits[t] = lg
            caches.append(cache)
        return logits, caches, (E, flat, a_enc, h0, K, V)

    def backward_sequence(self, dlogits: np.ndarray, caches, seq_cache, prompt_ids: np.ndarray) -> None:
        """整条序列的反向传播：累加所有参数梯度。"""
        E, flat, a_enc, h0, K, V = seq_cache
        d = self.cfg.d_emb
        ad = self.cfg.attn_dim
        T = len(caches)

        dh0_total = np.zeros(d, dtype=np.float64)
        dK_total = np.zeros_like(K)
        dV_total = np.zeros_like(V)

        for t in range(T - 1, -1, -1):
            x, a1, h1, prev, t_idx, attn_cache = caches[t]
            dh1 = self.lin2.backward(dlogits[t], h1)
            da1 = relu_backward(a1, dh1)
            dx = self.lin1.backward(da1, x)

            dh0_total += dx[:d]
            self.emb.backward(np.array([prev], dtype=np.int64), dx[d : 2 * d][None, :])
            d_pos = dx[2 * d : 3 * d]
            d_ctx = dx[3 * d : 3 * d + ad]

            dq_feat, dK_t, dV_t = self.attn.backward_query(d_ctx, attn_cache, K, V)
            dK_total += dK_t
            dV_total += dV_t
            # dq_feat = [对 h0 的部分 ; 对 pos_gen[t] 的部分]
            dh0_total += dq_feat[:d]
            self.pos_gen.backward(np.array([t_idx], dtype=np.int64), (d_pos + dq_feat[d:])[None, :])

        # 注意力记忆 -> E
        dE = self.attn.backward_memory(dK_total, dV_total, E)
        # h0 = tanh(a_enc) -> flat -> E（与注意力路径的梯度相加）
        dflat = self.enc.backward(tanh_backward(h0, dh0_total), flat)
        dE = dE + dflat.reshape(self.cfg.prompt_len, d)

        self.emb.backward(prompt_ids, dE)
        self.pos_prompt.backward(self._prompt_ids, dE)

    # -- 便捷接口 --------------------------------------------------------
    def sequence_logprobs(self, prompt_ids: np.ndarray, token_ids: np.ndarray):
        """返回 ``(logprobs (T,V), logits, caches, seq_cache)``。"""
        logits, caches, seq_cache = self.forward_sequence(prompt_ids, token_ids)
        return log_softmax(logits), logits, caches, seq_cache

    def token_logprob(self, prompt_ids: np.ndarray, token_ids: np.ndarray, t: int) -> float:
        """第 ``t`` 步被采样 token 的对数概率（不建缓存，用于参考策略）。"""
        h0, K, V = self.encode_nograd(prompt_ids)
        prev = BOS_ID if t == 0 else int(token_ids[t - 1])
        logits = self._step_nograd(h0, K, V, prev, t)
        return float(log_softmax(logits)[int(token_ids[t])])

    def greedy_sequence(self, prompt_ids: np.ndarray, max_len: int | None = None) -> np.ndarray:
        """贪心解码（argmax），用于评估。"""
        T = max_len or self.cfg.max_gen_len
        h0, K, V = self.encode_nograd(prompt_ids)
        out: list[int] = []
        for t in range(T):
            prev = BOS_ID if t == 0 else out[-1]
            nxt = int(np.argmax(self._step_nograd(h0, K, V, prev, t)))
            out.append(nxt)
            if nxt == EOS_ID:
                break
        return np.array(out, dtype=np.int64)

    def sample_sequence(
        self,
        prompt_ids: np.ndarray,
        rng: np.random.Generator,
        max_len: int | None = None,
        temperature: float = 1.0,
    ):
        """随机采样一条序列，返回 ``(tokens, logprobs, hit_eos)``。"""
        T = max_len or self.cfg.max_gen_len
        h0, K, V = self.encode_nograd(prompt_ids)
        toks: list[int] = []
        lps: list[float] = []
        hit_eos = False
        for t in range(T):
            prev = BOS_ID if t == 0 else toks[-1]
            logits = self._step_nograd(h0, K, V, prev, t)
            if temperature != 1.0:
                logits = logits / temperature
            logp = log_softmax(logits)
            p = np.exp(logp)
            p = p / p.sum()  # 抵消浮点误差，保证是合法概率
            nxt = int(rng.choice(self.cfg.vocab_size, p=p))
            toks.append(nxt)
            lps.append(float(logp[nxt]))
            if nxt == EOS_ID:
                hit_eos = True
                break
        return np.array(toks, dtype=np.int64), np.array(lps, dtype=np.float64), hit_eos


class ValueNet:
    """Critic：``V_φ(prompt, y_<t) -> R``（PPO 专用，GRPO 不使用）。

    输入是策略的 ``h0``（**detach**，不回传到策略）与该步的上一个 token 嵌入
    （critic 自己的嵌入表，与策略完全独立，避免梯度串流）。
    """

    def __init__(self, cfg: PolicyConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.emb = Embedding(cfg.vocab_size, cfg.d_emb, rng, scale=cfg.emb_scale)
        self.lin1 = Linear(2 * cfg.d_emb, cfg.value_hidden, rng)
        self.lin2 = Linear(cfg.value_hidden, 1, rng)

    def parameters(self):
        return self.emb.parameters() + self.lin1.parameters() + self.lin2.parameters()

    def zero_grad(self) -> None:
        for layer in (self.emb, self.lin1, self.lin2):
            layer.zero_grad()

    def num_parameters(self) -> int:
        return int(sum(p.size for _, p, _ in self.parameters()))

    def step_forward(self, h0: np.ndarray, prev_id: int):
        e_prev = self.emb.forward(np.array([prev_id], dtype=np.int64))[0]
        x = np.concatenate([h0, e_prev])
        a1 = self.lin1.forward(x)
        h1 = relu(a1)
        v = float(self.lin2.forward(h1)[0])
        return v, (x, a1, h1, int(prev_id))

    def step_backward(self, dv: float, cache) -> None:
        x, a1, h1, prev_id = cache
        dout = np.array([dv], dtype=np.float64)
        dh1 = self.lin2.backward(dout, h1)
        da1 = relu_backward(a1, dh1)
        dx = self.lin1.backward(da1, x)
        # 后一半是输入 token 的嵌入 -> 回填 critic 自己的嵌入表；
        # 前一半是 h0（来自策略，已 detach），梯度到此为止。
        self.emb.backward(np.array([prev_id], dtype=np.int64), dx[self.cfg.d_emb :][None, :])

    def values_for_sequence(self, h0: np.ndarray, token_ids: np.ndarray, cache: bool = False):
        """返回整条序列每一步的价值（以及可选的缓存）。"""
        T = len(token_ids)
        values = np.empty(T, dtype=np.float64)
        caches = []
        for t in range(T):
            prev = BOS_ID if t == 0 else int(token_ids[t - 1])
            if cache:
                v, c = self.step_forward(h0, prev)
                caches.append(c)
            else:
                e_prev = self.emb.E[prev]
                x = np.concatenate([h0, e_prev])
                v = float((self.lin2.W @ relu(self.lin1.W @ x + self.lin1.b) + self.lin2.b)[0])
            values[t] = v
        return (values, caches) if cache else values


# --------------------------------------------------------------------------
# 参考策略（冻结副本）
# --------------------------------------------------------------------------
def snapshot(module) -> list[np.ndarray]:
    """深拷贝一份参数快照（用作参考策略 / 冻结副本）。

    这里**按位置**而不是按名字索引：多个查表层的参数名都叫 ``E``，
    早期版本用 ``{name: array}`` 字典存快照时它们会互相覆盖，
    导致参考策略被悄悄污染（单测抓出来的真实 bug）。
    """
    return [np.array(p, copy=True) for _, p, _ in module.parameters()]


def load_snapshot(module, snap: list[np.ndarray]) -> None:
    """把快照写回模块参数（原地），保持数组身份不变以便优化器仍指向同一块内存。"""
    params = [p for _, p, _ in module.parameters()]
    if len(params) != len(snap):
        raise ValueError(f"快照长度 {len(snap)} 与模块参数个数 {len(params)} 不一致")
    for p, s in zip(params, snap):
        if p.shape != s.shape:
            raise ValueError(f"参数形状不匹配: {p.shape} vs {s.shape}")
        p[...] = s


def param_count_summary(policy: PolicyNet, critic: ValueNet | None = None) -> dict[str, int]:
    out = {"policy": policy.num_parameters()}
    if critic is not None:
        out["critic"] = critic.num_parameters()
        out["total"] = out["policy"] + out["critic"]
    else:
        out["total"] = out["policy"]
    return out
