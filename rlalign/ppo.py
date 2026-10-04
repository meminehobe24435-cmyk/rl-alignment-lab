"""PPO：GAE(λ) 优势估计 + clipped surrogate + Critic 价值网络 + 熵正则 + KL 惩罚。

关键实现点
----------
1. **GAE(λ)**（Schulman et al., 2016）
   ``δ_t = r_t + γ V(s_{t+1}) − V(s_t)``
   ``A_t = δ_t + γλ A_{t+1}``
   支持掩码：EOS 之后的填充步不参与，终止步不做 bootstrap。
2. **KL 惩罚**：逐 token 的 k1 估计 ``kl_t = logπ_ref(a_t) − logπ_old(a_t)``，
   以 ``r_t ← r_t − β·kl_t`` 的形式写进奖励（RLHF 的标准做法），
   同时在损失里单独监控 **k3** 估计 ``exp(Δ) − Δ − 1``（Δ 同上，恒非负）。
3. **Clipped surrogate**：逐 token 比值 ``ratio_t = exp(logπ_new − logπ_old)``，
   ``L = −min(ratio·A, clip(ratio, 1−ε, 1+ε)·A)``。
   比值一旦被裁剪且方向"有利"，该 token 的梯度严格为 0 —— 这就是 PPO 的信任域。
4. **优势归一化**：在整批有效 token 上做零均值单位方差归一化。
5. 因为整个 batch 很小，更新采用**全批**（不分 minibatch），多次 inner epoch
   复用同一批 rollout 数据 —— 更利于复现，行为等价于 PPO 的 epoch 循环。

梯度推导（策略部分）
--------------------
令 ``L_t = −min(r_t A_t, clip(r_t) A_t)``，``r_t = exp(logπ_new(a_t) − logπ_old(a_t))``：

``∂L_t/∂r_t = −A_t · 1[未裁剪分支更小]``，``∂r_t/∂logπ_new(a_t) = r_t``，
``∂logπ_new(a)/∂z_i = δ_{ia} − p_i``。
因此 ``∂L/∂z_i = (1/N) · Σ_t m_t · (−A_t·1[·]) · r_t · (δ_{i,a_t} − p_i)``。
熵正则项 ``−c_e·H`` 的梯度用 ``∂H/∂z_i = −p_i(log p_i + H)``。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nn import Adam, clip_grad_norm_, zero_grads
from .policy import PolicyNet, ValueNet, entropy_grad
from .rollout import RolloutBatch

__all__ = ["PPOConfig", "compute_gae", "normalize_advantages", "shaped_rewards", "ppo_iteration"]

# exp 的输入统一裁剪，避免极端比值产生 inf/nan（对 surrogate 无影响：
# 比值越界后梯度本来就已经是 0）。
_EXP_CLIP = 20.0


@dataclass
class PPOConfig:
    """PPO 超参。

    默认值不是拍脑袋写的：``lr`` 从 3e-3 一路降到 3e-4 是调试的直接结果 ——
    3e-3 下策略会在 10 次迭代内被彻底摧毁（KL 涨到 8+ nat/token、
    价值损失涨到 27，训练集成功率 1.00 -> 0.00）。详见 README 踩坑记录。
    """

    iterations: int = 80
    prompts_per_iter: int = 32
    inner_epochs: int = 2
    lr: float = 3e-4
    clip_eps: float = 0.2
    gamma: float = 1.0
    lam: float = 0.95
    kl_coef: float = 0.05
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 1.0
    temperature: float = 1.0
    seed: int = 0


# --------------------------------------------------------------------------
# 优势估计
# --------------------------------------------------------------------------
def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray | None = None,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """广义优势估计（单条序列，1D 输入）。

    返回 ``(advantages, returns)``，``returns = advantages + values``。
    ``mask`` 中为 0 的步被彻底排除（优势与回报置 0，且不向更早的步传播）。
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    T = rewards.shape[0]
    if mask is None:
        mask = np.ones(T, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    if values.shape[0] != T or mask.shape[0] != T:
        raise ValueError("rewards/values/mask 长度必须一致")

    adv = np.zeros(T, dtype=np.float64)
    running = 0.0
    for t in range(T - 1, -1, -1):
        if mask[t] <= 0.0:
            adv[t] = 0.0
            running = 0.0
            continue
        if t + 1 < T and mask[t + 1] > 0.0:
            next_v = values[t + 1]
            next_adv = running
        else:
            next_v = 0.0  # 终止步：不做 bootstrap
            next_adv = 0.0
        delta = rewards[t] + gamma * next_v - values[t]
        running = delta + gamma * lam * next_adv
        adv[t] = running

    returns = np.where(mask > 0.0, adv + values, 0.0)
    return adv, returns


def normalize_advantages(adv: np.ndarray, mask: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """在掩码有效元素上做零均值单位方差归一化（退化输入时返回全 0，不产生 NaN）。"""
    valid = mask > 0.0
    if not np.any(valid):
        return np.zeros_like(adv)
    vals = adv[valid]
    std = float(vals.std())
    out = np.zeros_like(adv)
    if std < eps:
        return out
    out[valid] = (vals - float(vals.mean())) / (std + eps)
    return out


def _surrogate_dratio(ratio: np.ndarray, adv: np.ndarray, clip_eps: float) -> np.ndarray:
    """``d(−min(ratio·A, clip(ratio)·A)) / dratio``。

    未裁剪分支更小（即落在信任域内）时导数为 ``−A``，否则严格为 0。
    """
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    return np.where(unclipped <= clipped, -adv, 0.0)


def shaped_rewards(batch: RolloutBatch, kl_coef: float) -> tuple[np.ndarray, np.ndarray]:
    """把 KL 惩罚并入奖励，并把序列级总奖励加到终止步上。

    返回 ``(塑形后的奖励 (B,T), 逐 token k1 KL (B,T))``。
    """
    kl = (batch.logp_ref - batch.logp_old) * batch.mask
    r = -kl_coef * kl
    for i in range(batch.n):
        last = int(batch.lengths[i]) - 1
        r[i, last] += batch.rewards[i]
    return r * batch.mask, kl


# --------------------------------------------------------------------------
# 一次迭代
# --------------------------------------------------------------------------
def ppo_iteration(
    policy: PolicyNet,
    critic: ValueNet,
    batch: RolloutBatch,
    cfg: PPOConfig,
    optimizer: Adam,
) -> dict:
    """在已采集的 batch 上做 ``cfg.inner_epochs`` 次全批 PPO 更新，返回诊断指标。"""
    r_shaped, kl_old = shaped_rewards(batch, cfg.kl_coef)

    adv = np.zeros_like(r_shaped)
    ret = np.zeros_like(r_shaped)
    for i in range(batch.n):
        n = int(batch.lengths[i])
        a_i, r_i = compute_gae(
            r_shaped[i, :n], batch.values[i, :n], batch.mask[i, :n], cfg.gamma, cfg.lam
        )
        adv[i, :n] = a_i
        ret[i, :n] = r_i
    adv_norm = normalize_advantages(adv, batch.mask)

    total_tokens = max(batch.n_tokens, 1)
    params = policy.parameters() + critic.parameters()
    eye = np.eye(policy.cfg.vocab_size, dtype=np.float64)
    epoch_stats: list[dict] = []

    for _epoch in range(cfg.inner_epochs):
        zero_grads(policy.parameters())
        zero_grads(critic.parameters())
        pg_sum = vf_sum = ent_sum = kl3_sum = clip_num = ratio_sum = 0.0

        for i in range(batch.n):
            n = int(batch.lengths[i])
            toks = batch.token_ids[i, :n]
            pids = batch.prompt_ids[batch.group_ids[i]]
            logp, logits, caches, enc_cache = policy.sequence_logprobs(pids, toks)
            # critic 的输入 h0 必须 detach：用不带缓存的路径计算，
            # 这样价值网络的梯度不会串回策略。
            h0_det = policy.prompt_vector_nograd(pids)
            p = np.exp(logp)
            m = batch.mask[i, :n]

            logp_take = logp[np.arange(n), toks]
            delta_lp = np.clip(logp_take - batch.logp_old[i, :n], -_EXP_CLIP, _EXP_CLIP)
            ratio = np.exp(delta_lp)
            clipped = np.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
            pg_t = -np.minimum(ratio * adv_norm[i, :n], clipped * adv_norm[i, :n])
            pg_sum += float(np.sum(pg_t * m))
            ratio_sum += float(np.sum(ratio * m))
            clip_num += float(np.sum((np.abs(ratio - 1.0) > cfg.clip_eps) * m))

            # k3 无偏 KL 估计（仅监控，不进入损失）
            d_ref = np.clip(batch.logp_ref[i, :n] - logp_take, -_EXP_CLIP, _EXP_CLIP)
            kl3_sum += float(np.sum((np.exp(d_ref) - d_ref - 1.0) * m))

            # --- 价值损失（普通 MSE，0.5·(V−R)²）---
            v, vcaches = critic.values_for_sequence(h0_det, toks, cache=True)
            dv = (v - ret[i, :n]) * m
            vf_sum += float(0.5 * np.sum(np.square(dv)))

            # --- 熵 ---
            dH = np.array([entropy_grad(logits[t]) for t in range(n)])
            H_t = -np.sum(p * logp, axis=1)
            ent_sum += float(np.sum(H_t * m))

            # --- 组装策略梯度 ---
            d_ratio = _surrogate_dratio(ratio, adv_norm[i, :n], cfg.clip_eps)
            dlogp = d_ratio * ratio * m / total_tokens  # ∂L/∂logπ_new(a_t)
            dlogits = (eye[toks] - p) * dlogp[:, None]
            dlogits += (-cfg.ent_coef / total_tokens) * dH * m[:, None]
            policy.backward_sequence(dlogits, caches, enc_cache, pids)

            # --- 组装价值梯度 ---
            for t in range(n):
                critic.step_backward(cfg.vf_coef * dv[t] / total_tokens, vcaches[t])

        grad_norm = clip_grad_norm_(params, cfg.max_grad_norm)
        optimizer.step()
        epoch_stats.append(
            {
                "policy_loss": pg_sum / total_tokens,
                "value_loss": vf_sum / total_tokens,
                "entropy": ent_sum / total_tokens,
                "kl_k3": kl3_sum / total_tokens,
                "ratio_mean": ratio_sum / total_tokens,
                "clip_frac": clip_num / total_tokens,
                "grad_norm": grad_norm,
            }
        )

    out = {k: float(np.mean([e[k] for e in epoch_stats])) for k in epoch_stats[0]}
    out["kl_k1"] = float(np.sum(kl_old) / total_tokens)
    out["grad_norm"] = float(epoch_stats[-1]["grad_norm"])
    return out
