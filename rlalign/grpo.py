"""GRPO：组内归一化优势 + **不需要 Critic** 的 PPO 式裁剪目标 + 直接 KL 项。

与 PPO 的三个本质差别
---------------------
1. **不需要价值网络**：同一个提示采样一组（G 条）轨迹，
   组内对奖励做 ``(R_i − mean(R)) / (std(R) + ε)`` 归一化，
   得到标量优势后**广播到该条的每一个 token**。
   省掉了 critic，也就省掉了价值网络的拟合误差与额外显存/算力。
2. **优势是序列级的**：没有 GAE，不做逐 token 的信用分配。
   在"只在句末给一次可验证奖励"的任务上这反而更合适。
3. **KL 直接进损失**：用 k3 无偏估计 ``exp(Δ) − Δ − 1``（``Δ = logπ_ref − logπ``）
   以 ``β`` 加权加到损失上，而不是像 PPO 那样先塑形奖励再算 GAE。

退化处理
--------
组内奖励完全相同时 ``std = 0``、分子也恰好为 0。这里显式判断
``std < eps → 优势置 0``，确保不会出现 ``0/0 = NaN`` 或者梯度爆炸
（这是本项目被单测抓出来的真实 bug 之一）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nn import Adam, clip_grad_norm_, zero_grads
from .policy import PolicyNet, entropy_grad
from .rollout import RolloutBatch

__all__ = ["GRPOConfig", "group_advantages", "grpo_iteration"]

_EXP_CLIP = 20.0


@dataclass
class GRPOConfig:
    """GRPO 超参。

    ``group_size = 8``：组越大，组内归一化的均值/方差估计越稳，
    但每条提示的采样开销线性增长。8 是在 CPU 上"信号质量 x 时间预算"
    的折中（实测 6 和 8 的最终成功率差异在噪声范围内）。
    """

    iterations: int = 80
    prompts_per_iter: int = 12
    group_size: int = 8
    inner_epochs: int = 2
    lr: float = 3e-4
    clip_eps: float = 0.2
    kl_coef: float = 0.05
    ent_coef: float = 0.02
    max_grad_norm: float = 1.0
    temperature: float = 1.0
    eps_std: float = 1e-4
    seed: int = 0


def group_advantages(
    rewards: np.ndarray,
    group_size: int,
    eps: float = 1e-4,
) -> np.ndarray:
    """组内归一化优势。

    ``A = (R − mean_group(R)) / (std_group(R) + ε)``，并按 ``group_size`` 展开回原顺序。
    组内奖励完全相同（``std < eps``）时返回全 0，不产生 NaN。
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    if group_size <= 0:
        raise ValueError("group_size 必须为正")
    if rewards.size % group_size != 0:
        raise ValueError("奖励数量必须是 group_size 的整数倍")
    n_groups = rewards.size // group_size
    R = rewards.reshape(n_groups, group_size)
    mean = R.mean(axis=1, keepdims=True)
    std = R.std(axis=1, keepdims=True)
    centered = R - mean
    # std 为 0（组内奖励完全相同）时把缩放系数直接置 0，
    # 避免 0/0 产生 NaN，也避免给完全相同的样本编造出虚假的学习信号。
    # 注意：std 的形状是 (n_groups, 1)，布尔掩码不能直接索引 (n_groups, G) 的数组，
    # 必须用 np.where 广播（这是单测抓出来的真实 bug）。
    scale = np.where(std >= eps, 1.0 / (std + eps), 0.0)
    return (centered * scale).reshape(-1)


def grpo_iteration(
    policy: PolicyNet,
    batch: RolloutBatch,
    cfg: GRPOConfig,
    optimizer: Adam,
) -> dict:
    """在已采集的分组 batch 上做 ``cfg.inner_epochs`` 次全批 GRPO 更新。"""
    if batch.group_ids is None or batch.rewards is None:
        raise ValueError("GRPO 需要带组信息的 batch")
    n_groups = int(batch.group_ids.max()) + 1
    counts = np.bincount(batch.group_ids, minlength=n_groups)
    if not np.all(counts == cfg.group_size):
        raise ValueError(
            f"每组大小必须等于 group_size={cfg.group_size}，实际为 {sorted(set(counts.tolist()))}"
        )

    adv_seq = group_advantages(batch.rewards, cfg.group_size, cfg.eps_std)
    # group_advantages 的输入输出都是"按轨迹顺序"排列的（reshape 后展平回原顺序），
    # 所以这里直接按轨迹下标广播到每个 token，不需要再用 group_ids 去 gather。
    adv_tok = adv_seq[:, None] * batch.mask

    # 组内奖励方差（诊断用：为 0 说明这一组没提供任何学习信号）
    R = batch.rewards.reshape(n_groups, cfg.group_size)
    group_var = float(np.mean(np.var(R, axis=1)))
    group_std_mean = float(np.mean(np.std(R, axis=1)))

    total_tokens = max(batch.n_tokens, 1)
    params = policy.parameters()
    eye = np.eye(policy.cfg.vocab_size, dtype=np.float64)
    epoch_stats: list[dict] = []

    for _epoch in range(cfg.inner_epochs):
        zero_grads(policy.parameters())
        pg_sum = kl_sum = ent_sum = clip_num = ratio_sum = 0.0

        for i in range(batch.n):
            n = int(batch.lengths[i])
            toks = batch.token_ids[i, :n]
            pids = batch.prompt_ids[batch.group_ids[i]]
            logp, logits, caches, enc_cache = policy.sequence_logprobs(pids, toks)
            p = np.exp(logp)
            m = batch.mask[i, :n]

            logp_take = logp[np.arange(n), toks]
            delta_lp = np.clip(logp_take - batch.logp_old[i, :n], -_EXP_CLIP, _EXP_CLIP)
            ratio = np.exp(delta_lp)
            clipped = np.clip(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)
            a_take = adv_tok[i, :n]
            pg_t = -np.minimum(ratio * a_take, clipped * a_take)
            pg_sum += float(np.sum(pg_t * m))
            ratio_sum += float(np.sum(ratio * m))
            clip_num += float(np.sum((np.abs(ratio - 1.0) > cfg.clip_eps) * m))

            # k3 KL 估计，直接进入损失
            d_ref = np.clip(batch.logp_ref[i, :n] - logp_take, -_EXP_CLIP, _EXP_CLIP)
            k3 = np.exp(d_ref) - d_ref - 1.0
            kl_sum += float(np.sum(k3 * m))

            dH = np.array([entropy_grad(logits[t]) for t in range(n)])
            H_t = -np.sum(p * logp, axis=1)
            ent_sum += float(np.sum(H_t * m))

            dlogits = np.zeros_like(logits)
            # 1) 裁剪后的策略梯度
            unclipped = ratio * a_take
            take_unclipped = unclipped <= clipped * a_take
            d_ratio = np.where(take_unclipped, -a_take, 0.0)
            dlogp = d_ratio * ratio * m / total_tokens
            dlogits += (eye[toks] - p) * dlogp[:, None]
            # 2) KL 项：∂(exp(Δ)−Δ−1)/∂logπ_new = 1 − exp(Δ)
            d_kl = (1.0 - np.exp(d_ref)) * m * (cfg.kl_coef / total_tokens)
            dlogits += (eye[toks] - p) * d_kl[:, None]
            # 3) 熵正则
            dlogits += (-cfg.ent_coef / total_tokens) * dH * m[:, None]

            policy.backward_sequence(dlogits, caches, enc_cache, pids)

        grad_norm = clip_grad_norm_(params, cfg.max_grad_norm)
        optimizer.step()
        epoch_stats.append(
            {
                "policy_loss": pg_sum / total_tokens,
                "kl_k3_loss": kl_sum / total_tokens,
                "entropy": ent_sum / total_tokens,
                "ratio_mean": ratio_sum / total_tokens,
                "clip_frac": clip_num / total_tokens,
                "grad_norm": grad_norm,
            }
        )

    out = {k: float(np.mean([e[k] for e in epoch_stats])) for k in epoch_stats[0]}
    out["kl_k3"] = out.pop("kl_k3_loss")
    out["group_reward_var"] = group_var
    out["group_reward_std"] = group_std_mean
    out["grad_norm"] = float(epoch_stats[-1]["grad_norm"])
    return out
