"""评估：在固定的评测集上测量成功率与平均奖励。

评测集与训练集**完全不相交**（见 :func:`rlalign.env.train_eval_split`），
并且只在评估时被使用。所有方法（随机 / SFT / PPO / GRPO）都在这同一套
评测集、同一个采样种子下评估，口径一致。

同时报告两种解码方式：
* ``success_rate``：**随机采样**（temperature=1）—— 反映策略的真实行为分布；
* ``greedy_success_rate``：贪心解码 —— 反映策略的众数行为。
两者都在表里给出，避免"挑一个好看的"。
"""

from __future__ import annotations

import numpy as np

from .env import EOS_ID, PAD_ID, Task, decode_tokens, encode
from .policy import PolicyNet
from .reward import RewardConfig, score_candidate

__all__ = ["summarize", "evaluate_policy", "evaluate_random_policy"]


def summarize(records: list[dict]) -> dict:
    """把逐样本打分记录汇总成指标。

    成功率是二项比例，这里一并给出**二项标准误** ``√(p(1−p)/n)``：
    评测集只有几百个任务，1 个任务就是 1/n 的差距，
    不报误差范围的话"谁赢谁输"很容易被噪声骗到。
    """
    n = len(records)
    if n == 0:
        return {
            "n": 0,
            "success_rate": 0.0,
            "success_rate_stderr": 0.0,
            "mean_reward": 0.0,
            "mean_outcome": 0.0,
            "mean_process": 0.0,
            "truncated_rate": 0.0,
        }
    p = float(np.mean([r["success"] for r in records]))
    return {
        "n": n,
        "success_rate": p,
        "success_rate_stderr": float(np.sqrt(max(p * (1.0 - p), 0.0) / n)),
        "mean_reward": float(np.mean([r["reward"] for r in records])),
        "mean_outcome": float(np.mean([r["outcome"] for r in records])),
        "mean_process": float(np.mean([r["process"] for r in records])),
        "truncated_rate": float(np.mean([r["truncated"] for r in records])),
    }


def evaluate_policy(
    policy: PolicyNet,
    tasks: list[Task],
    seed: int = 12345,
    reward_cfg: RewardConfig | None = None,
    greedy: bool = False,
    temperature: float = 1.0,
    max_gen_len: int | None = None,
) -> dict:
    """在评测集上评估一个策略，返回指标与逐样本记录。"""
    reward_cfg = reward_cfg or RewardConfig()
    rng = np.random.default_rng(seed)
    T = max_gen_len or policy.cfg.max_gen_len
    records: list[dict] = []

    for task in tasks:
        ids = encode(task.buggy)
        pids = np.full(policy.cfg.prompt_len, PAD_ID, dtype=np.int64)
        pids[: len(ids)] = ids
        if greedy:
            toks = policy.greedy_sequence(pids, max_len=T)
        else:
            toks, _lps, _eos = policy.sample_sequence(pids, rng, max_len=T, temperature=temperature)
        text, _ = decode_tokens(toks)
        hit_eos = len(toks) > 0 and int(toks[-1]) == EOS_ID
        records.append(score_candidate(text, task, reward_cfg, truncated=not hit_eos))

    out = summarize(records)
    out["records"] = records
    return out


def evaluate_random_policy(
    tasks: list[Task],
    vocab_size: int,
    seed: int = 12345,
    reward_cfg: RewardConfig | None = None,
    max_gen_len: int = 14,
) -> dict:
    """均匀随机策略基线：从词表中（除 PAD/BOS）均匀采样，直到 EOS 或长度上限。"""
    reward_cfg = reward_cfg or RewardConfig()
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    allowed = np.array([i for i in range(vocab_size) if i not in (PAD_ID, 1)], dtype=np.int64)

    for task in tasks:
        toks: list[int] = []
        hit_eos = False
        for _ in range(max_gen_len):
            nxt = int(allowed[rng.integers(len(allowed))])
            toks.append(nxt)
            if nxt == EOS_ID:
                hit_eos = True
                break
        text, _ = decode_tokens(toks)
        records.append(score_candidate(text, task, reward_cfg, truncated=not hit_eos))

    out = summarize(records)
    out["records"] = records
    return out
