"""轨迹采样（rollout）：从策略里采样表达式，并用可验证奖励打分。

支持的两种采集模式
------------------
* ``group_size = 1``：每个提示采一条轨迹 —— 用于 PPO。
* ``group_size > 1``：每个提示采一组（G 条）轨迹 —— 用于 GRPO 的组内归一化。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .env import MAX_GEN_TOKENS, MAX_PROMPT_TOKENS, PAD_ID, EOS_ID, Task, decode_tokens, encode
from .policy import PolicyNet, ValueNet, snapshot
from .reward import RewardConfig, score_candidate

__all__ = ["RolloutBatch", "collect_rollout", "pad_prompts"]


@dataclass
class RolloutBatch:
    """一次采集得到的整批数据（含填充与掩码）。"""

    prompt_ids: np.ndarray  # (B, P)
    token_ids: np.ndarray  # (B, T) 生成的动作序列，EOS 之后填 PAD
    mask: np.ndarray  # (B, T) float，1 表示该步是真实 token
    lengths: np.ndarray  # (B,) 实际步数（含 EOS 那一步）
    hit_eos: np.ndarray  # (B,) bool
    logp_old: np.ndarray  # (B, T) 采样时的策略对数概率
    logp_ref: np.ndarray  # (B, T) 参考策略对数概率（冻结快照）
    strings: list[str] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)
    buggies: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    bugs: list[str] = field(default_factory=list)
    kinds: list[str] = field(default_factory=list)
    values: np.ndarray | None = None  # (B, T) critic 价值，PPO 才有
    rewards: np.ndarray | None = None  # (B,) 总奖励
    outcomes: np.ndarray | None = None  # (B,) 结果奖励
    processes: np.ndarray | None = None  # (B,) 过程奖励
    group_ids: np.ndarray | None = None  # (B,) 每条轨迹所属的组（提示）下标

    @property
    def n(self) -> int:
        return int(self.token_ids.shape[0])

    @property
    def n_tokens(self) -> int:
        return int(self.mask.sum())

    def group_matrix(self) -> tuple[np.ndarray, int]:
        """把 ``rewards`` 重排成 ``(n_groups, group_size)``，用于组内归一化。"""
        if self.group_ids is None or self.rewards is None:
            raise ValueError("该 batch 没有组信息")
        gids = self.group_ids
        n_groups = int(gids.max()) + 1
        counts = np.bincount(gids, minlength=n_groups)
        g = int(counts[0])
        if not np.all(counts == g):
            raise ValueError("每个组的大小必须一致")
        return self.rewards.reshape(n_groups, g), g


def pad_prompts(tasks: list[Task], prompt_len: int = MAX_PROMPT_TOKENS) -> np.ndarray:
    """把任务的带 bug 表达式编码成定长提示（右侧补 PAD，左侧对齐）。"""
    out = np.full((len(tasks), prompt_len), PAD_ID, dtype=np.int64)
    for i, task in enumerate(tasks):
        ids = encode(task.buggy)
        if len(ids) > prompt_len:
            raise ValueError(f"提示过长: {task.buggy!r}")
        out[i, : len(ids)] = ids
    return out


def collect_rollout(
    policy: PolicyNet,
    tasks: list[Task],
    rng: np.random.Generator,
    ref_policy: PolicyNet | None = None,
    critic: ValueNet | None = None,
    group_size: int = 1,
    reward_cfg: RewardConfig | None = None,
    temperature: float = 1.0,
    max_gen_len: int | None = None,
) -> RolloutBatch:
    """采样一批轨迹并打分。

    每个提示采样 ``group_size`` 条轨迹；``group_ids`` 记录归属，供 GRPO 使用。
    """
    reward_cfg = reward_cfg or RewardConfig()
    T_max = max_gen_len or policy.cfg.max_gen_len

    prompt_ids = pad_prompts(tasks, policy.cfg.prompt_len)
    records: list[dict] = []
    seqs: list[np.ndarray] = []
    lps: list[np.ndarray] = []
    refs: list[np.ndarray] = []
    group_ids: list[int] = []

    for gi, task in enumerate(tasks):
        pids = prompt_ids[gi]
        for _ in range(group_size):
            toks, logps, hit_eos = policy.sample_sequence(
                pids, rng, max_len=T_max, temperature=temperature
            )
            if ref_policy is not None:
                ref_lp = np.array(
                    [ref_policy.token_logprob(pids, toks, t) for t in range(len(toks))],
                    dtype=np.float64,
                )
            else:
                ref_lp = logps
            text, _ = decode_tokens(toks)
            truncated = not hit_eos
            rec = score_candidate(text, task, reward_cfg, truncated=truncated)
            records.append(rec)
            seqs.append(toks)
            lps.append(logps)
            refs.append(ref_lp)
            group_ids.append(gi)

    B = len(seqs)
    T = int(max(len(s) for s in seqs))

    token_ids = np.full((B, T), PAD_ID, dtype=np.int64)
    mask = np.zeros((B, T), dtype=np.float64)
    logp_old = np.zeros((B, T), dtype=np.float64)
    logp_ref = np.zeros((B, T), dtype=np.float64)
    lengths = np.zeros(B, dtype=np.int64)
    hit_eos_arr = np.zeros(B, dtype=bool)
    for i, s in enumerate(seqs):
        n = len(s)
        token_ids[i, :n] = s
        mask[i, :n] = 1.0
        logp_old[i, :n] = lps[i]
        logp_ref[i, :n] = refs[i]
        lengths[i] = n
        hit_eos_arr[i] = records[i]["truncated"] is False

    values = None
    if critic is not None:
        values = np.zeros((B, T), dtype=np.float64)
        for i in range(B):
            h0 = policy.prompt_vector_nograd(prompt_ids[group_ids[i]])
            v = critic.values_for_sequence(h0, token_ids[i, : lengths[i]])
            values[i, : lengths[i]] = v

    return RolloutBatch(
        prompt_ids=prompt_ids,
        token_ids=token_ids,
        mask=mask,
        lengths=lengths,
        hit_eos=hit_eos_arr,
        logp_old=logp_old,
        logp_ref=logp_ref,
        strings=[r["candidate"] for r in records],
        task_ids=[r["task_id"] for r in records],
        buggies=[r["buggy"] for r in records],
        targets=[r["target"] for r in records],
        bugs=[r["bug"] for r in records],
        kinds=[r["kind"] for r in records],
        values=values,
        rewards=np.array([r["reward"] for r in records], dtype=np.float64),
        outcomes=np.array([r["outcome"] for r in records], dtype=np.float64),
        processes=np.array([r["process"] for r in records], dtype=np.float64),
        group_ids=np.array(group_ids, dtype=np.int64),
    )
