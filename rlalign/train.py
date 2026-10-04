"""训练闭环：随机策略基线、行为克隆（SFT）基线、PPO、GRPO。

四个方法共用同一套任务划分、同一套奖励、同一套评测集与评测种子，
差别只在"怎么更新参数"：

======================  ============================================
方法                     做法
======================  ============================================
random                  从词表均匀采样（完全不做任何学习）
sft (行为克隆)           用少量专家样例 (buggy -> target) 做监督学习
ppo                     GAE + clipped surrogate + Critic + 熵 + KL
grpo                    组内归一化优势 + 无 Critic + 直接 KL
======================  ============================================

``init_from="sft"`` 可以把 RL 从 SFT 的权重出发再训（对应真实 RLHF/RLVR 的
"SFT 之后再做 RL"流程），这是额外的一组诊断，不属于四个基线本身。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .env import EOS_ID, PAD_ID, Task, encode, train_eval_split
from .eval import evaluate_policy, evaluate_random_policy
from .grpo import GRPOConfig, grpo_iteration
from .nn import Adam, log_softmax, zero_grads
from .policy import PolicyConfig, PolicyNet, ValueNet, load_snapshot, snapshot
from .ppo import PPOConfig, ppo_iteration
from .reward import RewardConfig
from .rollout import collect_rollout

__all__ = ["RunConfig", "build_tasks", "policy_cfg", "train_sft", "run_rl", "run_random_baseline"]


@dataclass
class RunConfig:
    """全局实验配置（算法超参在 PPOConfig / GRPOConfig 里）。"""

    seed: int = 0
    n_train: int = 1000
    n_eval: int = 200
    eval_interval: int = 20
    eval_seed: int = 12345
    outcome_weight: float = 1.0
    process_weight: float = 0.3
    d_emb: int = 32
    hidden: int = 128
    value_hidden: int = 64
    # SFT
    sft_epochs: int = 30
    sft_lr: float = 6e-3
    sft_batch: int = 24
    # RL 的初始化方式：scratch（随机初始化）或 sft（从 SFT 权重出发）
    init_from: str = "scratch"


def build_tasks(cfg: RunConfig) -> tuple[list[Task], list[Task]]:
    """构造训练/评测划分（两者互不相交）。"""
    return train_eval_split(cfg.n_train, cfg.n_eval, seed=cfg.seed)


def policy_cfg(cfg: RunConfig) -> PolicyConfig:
    return PolicyConfig(d_emb=cfg.d_emb, hidden=cfg.hidden, value_hidden=cfg.value_hidden)


def reward_cfg(cfg: RunConfig) -> RewardConfig:
    return RewardConfig(outcome_weight=cfg.outcome_weight, process_weight=cfg.process_weight)


def _prompt_ids(policy: PolicyNet, text: str) -> np.ndarray:
    ids = encode(text)
    if len(ids) > policy.cfg.prompt_len:
        raise ValueError(f"提示过长: {text!r}")
    out = np.full(policy.cfg.prompt_len, PAD_ID, dtype=np.int64)
    out[: len(ids)] = ids
    return out


def make_policy(cfg: RunConfig, seed_offset: int = 0, init: PolicyNet | None = None) -> PolicyNet:
    """新建策略网络；``init`` 非空时把它的权重拷过来。"""
    policy = PolicyNet(policy_cfg(cfg), np.random.default_rng(cfg.seed + seed_offset))
    if init is not None:
        load_snapshot(policy, snapshot(init))
    return policy


# --------------------------------------------------------------------------
# 行为克隆 / SFT
# --------------------------------------------------------------------------
def train_sft(
    cfg: RunConfig,
    train_tasks: list[Task],
    policy: PolicyNet | None = None,
    log=print,
) -> tuple[PolicyNet, list[dict]]:
    """用专家样例做监督学习（教师强制 + 交叉熵）。

    每个任务给出一条专家轨迹：``buggy -> target``（尾部追加 ``<eos>``）。
    损失是逐 token 交叉熵（按整批 token 数归一化），Adam 优化。
    这就是"如果不做 RL，只做 SFT 会怎样"的对照基线。
    """
    policy = policy or make_policy(cfg)
    rng = np.random.default_rng(cfg.seed + 77)
    opt = Adam(policy.parameters(), lr=cfg.sft_lr)
    eye = np.eye(policy.cfg.vocab_size, dtype=np.float64)

    prompts = [_prompt_ids(policy, t.buggy) for t in train_tasks]
    targets = [np.array(encode(t.target) + [EOS_ID], dtype=np.int64) for t in train_tasks]
    if any(len(t) > policy.cfg.max_gen_len for t in targets):
        raise ValueError("专家序列超过 max_gen_len")

    history: list[dict] = []
    n = len(train_tasks)
    for epoch in range(1, cfg.sft_epochs + 1):
        order = rng.permutation(n)
        epoch_loss = 0.0
        nb = 0
        for start in range(0, n, cfg.sft_batch):
            batch_idx = order[start : start + cfg.sft_batch]
            total_tokens = max(sum(len(targets[i]) for i in batch_idx), 1)
            zero_grads(policy.parameters())
            for i in batch_idx:
                toks = targets[i]
                logits, caches, enc_cache = policy.forward_sequence(prompts[i], toks)
                logp = log_softmax(logits)
                p = np.exp(logp)
                T = len(toks)
                # loss = -(1/N) Σ_t log π(a_t)；∂loss/∂z_t = -(1/N)(onehot(a_t) − p_t)
                dlogits = (eye[toks] - p) * (-1.0 / total_tokens)
                policy.backward_sequence(dlogits, caches, enc_cache, prompts[i])
                epoch_loss += float(-np.sum(logp[np.arange(T), toks])) / total_tokens
            opt.step()
            nb += 1
        history.append({"epoch": epoch, "sft_loss": epoch_loss / max(nb, 1)})
        if log and (epoch % 10 == 0 or epoch == 1):
            log(f"    [sft] epoch {epoch:2d}  loss={history[-1]['sft_loss']:.4f}")
    return policy, history


# --------------------------------------------------------------------------
# 强化学习主循环（PPO / GRPO 共用骨架）
# --------------------------------------------------------------------------
def run_rl(
    algo: str,
    cfg: RunConfig,
    rl_cfg,
    train_tasks: list[Task],
    eval_tasks: list[Task],
    init: PolicyNet | None = None,
    log=print,
) -> dict:
    """跑一次 PPO 或 GRPO，返回策略、逐迭代指标与最终评测结果。"""
    if algo not in ("ppo", "grpo"):
        raise ValueError(f"未知算法: {algo}")
    rc = reward_cfg(cfg)
    policy = make_policy(cfg, seed_offset=0, init=init)
    critic = ValueNet(policy.cfg, np.random.default_rng(cfg.seed + 1)) if algo == "ppo" else None

    # 参考策略 = 初始策略的冻结副本（KL 惩罚的锚点）
    ref = make_policy(cfg, seed_offset=2)
    load_snapshot(ref, snapshot(policy))

    sample_rng = np.random.default_rng(cfg.seed + 3)
    order_rng = np.random.default_rng(cfg.seed + 4)
    params = policy.parameters() + (critic.parameters() if critic is not None else [])
    opt = Adam(params, lr=rl_cfg.lr)

    group_size = getattr(rl_cfg, "group_size", 1)
    n_prompts = rl_cfg.prompts_per_iter
    metrics: list[dict] = []
    t_start = time.time()

    for it in range(1, rl_cfg.iterations + 1):
        idx = order_rng.permutation(len(train_tasks))[:n_prompts]
        batch_tasks = [train_tasks[int(i)] for i in idx]
        batch = collect_rollout(
            policy,
            batch_tasks,
            sample_rng,
            ref_policy=ref,
            critic=critic,
            group_size=group_size,
            reward_cfg=rc,
            temperature=rl_cfg.temperature,
        )
        if algo == "ppo":
            stats = ppo_iteration(policy, critic, batch, rl_cfg, opt)
        else:
            stats = grpo_iteration(policy, batch, rl_cfg, opt)

        row: dict = {
            "iter": it,
            "train_success_rate": float(np.mean(batch.outcomes >= 1.0)),
            "train_mean_reward": float(np.mean(batch.rewards)),
        }
        row.update(stats)
        if group_size > 1:
            R = batch.rewards.reshape(-1, group_size)
            row["group_reward_var"] = float(np.mean(np.var(R, axis=1)))
        if it % cfg.eval_interval == 0 or it == rl_cfg.iterations:
            ev = evaluate_policy(policy, eval_tasks, seed=cfg.eval_seed, reward_cfg=rc)
            row["eval_success_rate"] = ev["success_rate"]
            row["eval_mean_reward"] = ev["mean_reward"]
            if log:
                log(
                    f"    [{algo}] iter {it:3d}  train_succ={row['train_success_rate']:.3f} "
                    f"reward={row['train_mean_reward']:.3f} kl={stats['kl_k3']:.4f} "
                    f"H={stats['entropy']:.3f} eval_succ={ev['success_rate']:.3f} "
                    f"({time.time() - t_start:.1f}s)"
                )
        metrics.append(row)

    final_sampled = evaluate_policy(policy, eval_tasks, seed=cfg.eval_seed, reward_cfg=rc)
    final_greedy = evaluate_policy(policy, eval_tasks, seed=cfg.eval_seed, reward_cfg=rc, greedy=True)
    return {
        "algorithm": algo,
        "policy": policy,
        "critic": critic,
        "reference": ref,
        "metrics": metrics,
        "final_sampled": final_sampled,
        "final_greedy": final_greedy,
        "elapsed_sec": time.time() - t_start,
        "init_from": cfg.init_from,
        "group_size": group_size,
    }


def run_random_baseline(
    cfg: RunConfig,
    eval_tasks: list[Task],
    vocab_size: int,
    max_gen_len: int = 14,
) -> dict:
    """均匀随机策略基线（不做任何学习，也不需要网络）。"""
    ev = evaluate_random_policy(
        eval_tasks,
        vocab_size=vocab_size,
        seed=cfg.eval_seed,
        reward_cfg=reward_cfg(cfg),
        max_gen_len=max_gen_len,
    )
    return {"algorithm": "random", "final_sampled": ev, "final_greedy": ev, "metrics": []}
