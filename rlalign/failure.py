"""失败案例分析：把评测失败的样本归类，给出占比与典型例子。

分类口径（互斥，按优先级判定）
------------------------------
=========================  ================================================
桶                          判定条件
=========================  ================================================
``ok``                     全部测试用例通过
``truncated``              在 ``max_gen_len`` 内没有输出 ``<eos>``（超长截断）
``unbalanced_paren``       左右括号数量/嵌套不配平
``syntax``                 词法或语法错误（含一元负号、多余运算符、缺操作数）
``eval_error``             能解析但求值报错（典型是除零）
``semantic``               能跑通但数值不对（部分用例通过）
=========================  ================================================

括号不配平被单独拎出来，是因为它同时属于"语法错误"，
但在真实代码修复里是可定位、可自动修复的一类，值得独立统计。
"""

from __future__ import annotations

from collections import Counter, OrderedDict

__all__ = ["BUCKET_LABELS", "classify_record", "classify", "render_table"]

BUCKET_LABELS = OrderedDict(
    [
        ("ok", "修复成功（全部测试通过）"),
        ("truncated", "超长截断（未在预算内输出 <eos>）"),
        ("unbalanced_paren", "括号不配平"),
        ("syntax", "语法/词法错误"),
        ("eval_error", "求值错误（除零等）"),
        ("semantic", "语义错误（结果不对）"),
    ]
)


def classify_record(rec: dict) -> str:
    """把单条评测记录归到唯一一个桶。"""
    if rec.get("success"):
        return "ok"
    if rec.get("truncated"):
        return "truncated"
    if not rec.get("parens_balanced", True):
        return "unbalanced_paren"
    if rec.get("kind") in ("lex", "syntax"):
        return "syntax"
    if rec.get("kind") == "eval":
        return "eval_error"
    return "semantic"


def classify(records: list[dict], max_examples: int = 3) -> dict:
    """整体分类统计：占比表 + 每桶典型例子 + 按注入 bug 类型的成功率。"""
    n = len(records)
    buckets: dict[str, list[dict]] = {k: [] for k in BUCKET_LABELS}
    for rec in records:
        buckets[classify_record(rec)].append(rec)

    table = []
    for key, label in BUCKET_LABELS.items():
        items = buckets[key]
        table.append(
            {
                "bucket": key,
                "label": label,
                "count": len(items),
                "share": (len(items) / n) if n else 0.0,
                "examples": [
                    {
                        "task_id": r["task_id"],
                        "buggy": r["buggy"],
                        "target": r["target"],
                        "candidate": r["candidate"],
                        "passed": r["passed"],
                        "total_tests": r["total_tests"],
                        "bug": r["bug"],
                    }
                    for r in items[:max_examples]
                ],
            }
        )

    # 失败样本（非 ok）的平均通过用例数，衡量"差多远"
    failed = [r for r in records if not r.get("success")]
    partial = [r["passed"] / max(r["total_tests"], 1) for r in failed]

    # 按注入 bug 类型统计成功率：哪类 bug 最难过
    by_bug: dict[str, dict] = {}
    bug_counter = Counter(r["bug"] for r in records)
    for bug in sorted(bug_counter):
        sub = [r for r in records if r["bug"] == bug]
        by_bug[bug] = {
            "n": len(sub),
            "success_rate": (sum(1 for r in sub if r.get("success")) / len(sub)) if sub else 0.0,
        }

    return {
        "n": n,
        "success_rate": (sum(1 for r in records if r.get("success")) / n) if n else 0.0,
        "table": table,
        "failed_mean_test_pass_ratio": (sum(partial) / len(partial)) if partial else 0.0,
        "by_injected_bug": by_bug,
    }


def render_table(cls: dict) -> str:
    """渲染成 Markdown 表格（直接可贴进 README）。"""
    lines = [
        "| 失败模式 | 含义 | 数量 | 占比 |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in cls["table"]:
        lines.append(
            f"| `{row['bucket']}` | {row['label']} | {row['count']} | {row['share'] * 100:.1f}% |"
        )
    lines.append("")
    lines.append("典型例子：")
    lines.append("")
    lines.append("| 桶 | 任务 | 带 bug 输入 | 目标 | 模型输出 | 通过用例 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in cls["table"]:
        for ex in row["examples"][:2]:
            cand = ex["candidate"] if ex["candidate"] else "（空）"
            lines.append(
                f"| {row['bucket']} | {ex['task_id']} | `{ex['buggy']}` | `{ex['target']}` | "
                f"`{cand}` | {ex['passed']}/{ex['total_tests']} |"
            )
    return "\n".join(lines)
