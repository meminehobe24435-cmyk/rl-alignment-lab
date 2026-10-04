"""奖励函数：可验证结果奖励 + 过程奖励。

奖励设计
--------
**结果奖励（可验证 / verifiable）**
    ``outcome = 通过的测试用例数 / 总用例数``，取值 ``[0, 1]``。
    由 :func:`rlalign.env.run_tests` 真实执行表达式得出，是精确有理数比较，
    没有任何学习出来的 reward model，也没有人类偏好标注。

**过程奖励（稠密信号）**
    由三项组成，取值 ``[0, 1]``：

    ====================  ======  ==========================================
    分量                  权重    含义
    ====================  ======  ==========================================
    语法合法              0.5     能被词法/语法分析器接受
    求值不报错            0.2     所有变量取值下都能算出来（无除零 / 未绑定）
    与目标的编辑相似度    0.3     ``1 - Lev(候选, 目标) / max(len)``
    ====================  ======  ==========================================

    过程奖励的作用是给"从零开始"的策略一个稠密的起步信号：
    纯结果奖励在随机初始化的策略下几乎恒为 0（梯度全是 0），
    RL 完全无法起步。代价是它会把策略往"抄提示/抄目标"的方向拉，
    可能限制最终成功率上限 —— README 里给出了消融实测与讨论。

总奖励：``reward = outcome_weight * outcome + process_weight * process``。
"""

from __future__ import annotations

from dataclasses import dataclass

from .env import ExprError, Task, parens_balanced, run_tests

__all__ = ["RewardConfig", "levenshtein", "outcome_reward", "process_reward", "score_candidate"]


@dataclass
class RewardConfig:
    """奖励权重。"""

    outcome_weight: float = 1.0
    process_weight: float = 0.3
    # 过程奖励内部权重
    w_syntax: float = 0.5
    w_evaluable: float = 0.2
    w_edit: float = 0.3


def levenshtein(a: str, b: str) -> int:
    """标准编辑距离（可插入/删除/替换），O(len(a)·len(b)) 动态规划。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def outcome_reward(candidate: str, task: Task) -> float:
    """可验证结果奖励：通过测试用例的比例。"""
    if not candidate:
        return 0.0
    return run_tests(candidate, task).frac


def process_reward(candidate: str, task: Task, cfg: RewardConfig | None = None) -> tuple[float, dict]:
    """过程奖励，返回 ``(值, 分量字典)``。"""
    cfg = cfg or RewardConfig()
    parts = {"syntax": 0.0, "evaluable": 0.0, "edit": 0.0}

    # 1) 语法合法
    res = run_tests(candidate, task)
    parse_ok = res.kind not in ("lex", "syntax") and bool(candidate)
    parts["syntax"] = 1.0 if parse_ok else 0.0

    # 2) 全部取值下可求值
    if parse_ok and res.kind != "eval":
        parts["evaluable"] = 1.0

    # 3) 与目标的编辑相似度
    denom = max(len(candidate), len(task.target), 1)
    parts["edit"] = 1.0 - levenshtein(candidate, task.target) / denom

    value = (
        cfg.w_syntax * parts["syntax"]
        + cfg.w_evaluable * parts["evaluable"]
        + cfg.w_edit * parts["edit"]
    )
    return float(value), parts


def score_candidate(
    candidate: str,
    task: Task,
    cfg: RewardConfig | None = None,
    truncated: bool = False,
) -> dict:
    """完整的奖励打分，返回所有分量与失败分类所需的元信息。"""
    cfg = cfg or RewardConfig()
    res = run_tests(candidate, task)
    outcome = res.frac if candidate else 0.0
    process, parts = process_reward(candidate, task, cfg)
    total = cfg.outcome_weight * outcome + cfg.process_weight * process
    return {
        "task_id": task.task_id,
        "candidate": candidate,
        "target": task.target,
        "buggy": task.buggy,
        "bug": task.bug,
        "outcome": float(outcome),
        "process": float(process),
        "reward": float(total),
        "passed": res.passed,
        "total_tests": res.total,
        "kind": res.kind,
        "truncated": bool(truncated),
        "syntax_ok": bool(parts["syntax"]),
        "parens_balanced": bool(parens_balanced(candidate)) if candidate else False,
        "success": bool(outcome >= 1.0),
    }
