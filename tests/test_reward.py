"""奖励函数：可验证结果奖励 + 过程奖励。"""

from __future__ import annotations

import pytest

from rlalign.env import generate_tasks
from rlalign.reward import RewardConfig, levenshtein, outcome_reward, process_reward, score_candidate

TASK = generate_tasks(1, seed=21, prefix="t")[0]


# ------------------------------------------------------------------ 编辑距离
def test_levenshtein_identical():
    assert levenshtein("a+b", "a+b") == 0


def test_levenshtein_empty_strings():
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "") == 3
    assert levenshtein("", "abc") == 3


def test_levenshtein_single_substitution():
    assert levenshtein("a+b", "a-b") == 1


def test_levenshtein_single_insertion_and_deletion():
    assert levenshtein("ab", "abc") == 1
    assert levenshtein("abc", "ab") == 1


def test_levenshtein_is_symmetric():
    assert levenshtein("a*b+c", "a+b*c") == levenshtein("a+b*c", "a*b+c")


def test_levenshtein_known_value():
    assert levenshtein("kitten", "sitting") == 3


# ------------------------------------------------------------------ 结果奖励
def test_outcome_reward_is_one_for_correct_expression():
    """全对 -> 1.0"""
    assert outcome_reward(TASK.target, TASK) == 1.0


def test_outcome_reward_is_zero_for_unparseable():
    """全错 -> 0"""
    assert outcome_reward("a++b", TASK) == 0.0
    assert outcome_reward("$", TASK) == 0.0
    assert outcome_reward("", TASK) == 0.0


def test_outcome_reward_of_buggy_input_is_below_one():
    # 任务的构造保证带 bug 表达式至少挂一个用例
    assert outcome_reward(TASK.buggy, TASK) < 1.0


def test_outcome_reward_is_a_proportion():
    """部分对 -> 比例正确（通过数 / 总数）。

    手工构造任务：目标 ``a``，变量 a 依次取 1/1/2/3，期望值就是 1/1/2/3。
    于是 ``1`` 命中前两组（0.5）、``2`` 只命中第三组（0.25）、``9`` 全错（0.0）。
    """
    from rlalign.env import Task

    task = Task(
        task_id="t-partial",
        buggy="9",
        target="a",
        bug="swap_digit",
        bindings=((1, 1, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1)),
        expected=((1, 1), (1, 1), (2, 1), (3, 1)),
    )
    assert outcome_reward("a", task) == pytest.approx(1.0)
    assert outcome_reward("1", task) == pytest.approx(0.5)
    assert outcome_reward("2", task) == pytest.approx(0.25)
    assert outcome_reward("9", task) == pytest.approx(0.0)
    assert outcome_reward("a++", task) == pytest.approx(0.0)  # 语法错误 -> 0


def test_outcome_reward_partial_over_generated_tasks():
    from rlalign.env import run_tests

    for task in generate_tasks(120, seed=22, prefix="t"):
        res = run_tests(task.buggy, task)
        # 无论是否出现部分通过，奖励口径必须严格等于 通过数/总数
        assert outcome_reward(task.buggy, task) == pytest.approx(res.passed / res.total)


def test_outcome_reward_bounds_over_many_tasks():
    for task in generate_tasks(40, seed=24, prefix="t"):
        for cand in (task.target, task.buggy, "1", "", "a++b"):
            assert 0.0 <= outcome_reward(cand, task) <= 1.0


def test_outcome_reward_matches_run_tests():
    from rlalign.env import run_tests

    for task in generate_tasks(20, seed=23, prefix="t"):
        res = run_tests(task.buggy, task)
        assert outcome_reward(task.buggy, task) == pytest.approx(res.frac)


# ------------------------------------------------------------------ 过程奖励
def test_process_reward_in_unit_interval():
    for cand in [TASK.target, TASK.buggy, "a++b", "", "1"]:
        val, parts = process_reward(cand, TASK)
        assert 0.0 <= val <= 1.0
        assert set(parts) == {"syntax", "evaluable", "edit"}


def test_process_reward_is_maximal_for_target():
    val, parts = process_reward(TASK.target, TASK)
    assert val == pytest.approx(1.0)
    assert parts["syntax"] == 1.0
    assert parts["evaluable"] == 1.0
    assert parts["edit"] == 1.0


def test_process_reward_zero_for_unparseable():
    val, parts = process_reward("a++b", TASK)
    assert parts["syntax"] == 0.0
    assert parts["evaluable"] == 0.0
    assert val < 1.0


def test_process_reward_edit_similarity_decreases_with_distance():
    # 与目标只差 1 个字符的候选，其编辑相似度必须高于完全无关的候选
    last = TASK.target[-1]
    near = TASK.target[:-1] + ("0" if last != "0" else "1")
    assert levenshtein(near, TASK.target) == 1
    _v_target, parts_target = process_reward(TASK.target, TASK)
    _v_near, parts_near = process_reward(near, TASK)
    _v_far, parts_far = process_reward("", TASK)
    assert parts_target["edit"] == pytest.approx(1.0)
    assert parts_near["edit"] > parts_far["edit"]
    assert parts_far["edit"] == pytest.approx(0.0)


def test_process_reward_disabled_by_zero_weights():
    cfg = RewardConfig(w_syntax=0.0, w_evaluable=0.0, w_edit=0.0)
    val, _ = process_reward(TASK.target, TASK, cfg)
    assert val == 0.0


# ------------------------------------------------------------------ 总奖励
def test_total_reward_combines_both_parts():
    cfg = RewardConfig(outcome_weight=1.0, process_weight=0.3)
    rec = score_candidate(TASK.target, TASK, cfg)
    assert rec["outcome"] == 1.0
    assert rec["reward"] == pytest.approx(1.0 + 0.3 * rec["process"])


def test_outcome_only_config_removes_process_contribution():
    cfg = RewardConfig(outcome_weight=1.0, process_weight=0.0)
    rec = score_candidate(TASK.buggy, TASK, cfg)
    assert rec["reward"] == pytest.approx(rec["outcome"])


def test_score_candidate_reports_metadata():
    rec = score_candidate(TASK.target, TASK)
    for key in (
        "task_id",
        "candidate",
        "target",
        "buggy",
        "bug",
        "outcome",
        "process",
        "reward",
        "passed",
        "total_tests",
        "kind",
        "truncated",
        "syntax_ok",
        "parens_balanced",
        "success",
    ):
        assert key in rec
    assert rec["success"] is True


def test_score_candidate_marks_truncation():
    rec = score_candidate(TASK.target, TASK, truncated=True)
    assert rec["truncated"] is True
    assert rec["success"] is True  # 截断标记不影响可验证结果


def test_score_candidate_flags_unbalanced_parens():
    rec = score_candidate("a+b)", TASK)
    assert rec["parens_balanced"] is False
    assert rec["success"] is False
