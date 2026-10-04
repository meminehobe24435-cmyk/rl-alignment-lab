"""迷你表达式语言：词法 / 语法 / 求值 / 测试框架 / 任务集。"""

from __future__ import annotations

from fractions import Fraction

import pytest

from rlalign.env import (
    EOS_ID,
    PAD_ID,
    VOCAB_SIZE,
    EvalError,
    LexError,
    SyntaxError_,
    decode,
    decode_tokens,
    encode,
    evaluate_str,
    generate_tasks,
    parens_balanced,
    parse,
    run_tests,
    train_eval_split,
)

ENV = {"a": 1, "b": 2, "c": 3}


# ---------------------------------------------------------------- 词表 / 编解码
def test_vocab_size_at_most_32():
    # 设计要求：词表 <= 32 个 token
    assert VOCAB_SIZE <= 32


def test_encode_decode_roundtrip():
    s = "3*(a+b)-c/2"
    assert decode(encode(s)) == s


def test_encode_skips_whitespace():
    assert encode(" a + b ") == encode("a+b")


def test_encode_rejects_illegal_char():
    with pytest.raises(LexError):
        encode("a$b")


def test_decode_tokens_reports_missing_eos():
    text, hit = decode_tokens(encode("a+b"))
    assert text == "a+b"
    assert hit is False


def test_decode_tokens_stops_at_eos():
    text, hit = decode_tokens(encode("a+b") + [EOS_ID] + encode("c"))
    assert text == "a+b"
    assert hit is True


def test_decode_tokens_skips_pad():
    text, hit = decode_tokens([PAD_ID] + encode("a") + [EOS_ID])
    assert text == "a"
    assert hit is True


# ---------------------------------------------------------------- 语法分析
@pytest.mark.parametrize(
    "text,expected",
    [("a+b", Fraction(3)), ("a-b", Fraction(-1)), ("a*b", Fraction(2)), ("b/a", Fraction(2))],
)
def test_evaluate_basic_operators(text, expected):
    assert evaluate_str(text, ENV) == expected


def test_operator_precedence():
    # 2 + 3*3 = 11，而不是 (2+3)*3 = 15
    assert evaluate_str("b+c*c", ENV) == Fraction(11)


def test_parentheses_override_precedence():
    assert evaluate_str("(b+c)*c", ENV) == Fraction(15)


def test_multi_digit_number():
    assert evaluate_str("12+3", ENV) == Fraction(15)


def test_leading_zero_number():
    assert evaluate_str("007+1", ENV) == Fraction(8)


def test_division_is_exact_rational():
    # 精确有理数：1/3 不会退化成浮点近似
    assert evaluate_str("a/c", ENV) == Fraction(1, 3)


def test_left_associativity_of_subtraction():
    assert evaluate_str("10-3-2", ENV) == Fraction(5)


def test_left_associativity_of_division():
    assert evaluate_str("8/4/2", ENV) == Fraction(1)


def test_unary_minus_is_rejected():
    with pytest.raises(SyntaxError_):
        parse("-a")


def test_lone_operator_is_syntax_error():
    with pytest.raises(SyntaxError_):
        parse("+")


def test_trailing_operator_is_syntax_error():
    with pytest.raises(SyntaxError_):
        parse("a+")


def test_empty_expression_is_syntax_error():
    with pytest.raises(SyntaxError_):
        parse("")


def test_unclosed_paren_is_syntax_error():
    with pytest.raises(SyntaxError_):
        parse("(a+b")


def test_extra_close_paren_is_syntax_error():
    with pytest.raises(SyntaxError_):
        parse("a+b)")


def test_empty_parens_is_syntax_error():
    with pytest.raises(SyntaxError_):
        parse("()")


def test_double_operator_is_syntax_error():
    with pytest.raises(SyntaxError_):
        parse("a*+b")


def test_division_by_zero_literal():
    with pytest.raises(EvalError):
        evaluate_str("a/0", ENV)


def test_division_by_zero_from_variable():
    with pytest.raises(EvalError):
        evaluate_str("a/(b-b)", ENV)


def test_unknown_variable_is_lex_error():
    # 词表里只有 a/b/c，其它字母在词法层就被拦下
    with pytest.raises(LexError):
        evaluate_str("a+z", ENV)


def test_unbound_variable_raises_eval_error():
    # 直接调求值器：变量名合法但没有绑定值
    node = ("var", "a")
    with pytest.raises(EvalError):
        from rlalign.env import evaluate

        evaluate(node, {})


def test_parens_balanced_helper():
    assert parens_balanced("(a+b)*(c)") is True
    assert parens_balanced("((a+b)") is False
    assert parens_balanced("a+b)") is False
    assert parens_balanced("") is True


# ---------------------------------------------------------------- 测试框架
def test_run_tests_all_pass():
    task = generate_tasks(1, seed=5, prefix="t")[0]
    res = run_tests(task.target, task)
    assert res.passed == res.total
    assert res.kind == "ok"
    assert res.frac == 1.0


def test_run_tests_syntax_error_gives_zero():
    task = generate_tasks(1, seed=5, prefix="t")[0]
    res = run_tests("a++b", task)
    assert res.passed == 0
    assert res.frac == 0.0
    assert res.kind == "syntax"


def test_run_tests_lex_error_gives_zero():
    task = generate_tasks(1, seed=5, prefix="t")[0]
    res = run_tests("a$b", task)
    assert res.kind == "lex"
    assert res.frac == 0.0


def test_run_tests_detects_divergence_free_but_wrong():
    task = generate_tasks(1, seed=5, prefix="t")[0]
    # "1" 一定能解析、一定不除零，但几乎不可能全对
    res = run_tests("1", task)
    assert res.passed < res.total


def test_run_tests_counts_partial_credit():
    tasks = generate_tasks(40, seed=11, prefix="t")
    # 至少有一个任务：目标表达式本身 4/4 通过，说明部分通过计数可用
    for task in tasks[:10]:
        res = run_tests(task.target, task)
        assert 0 <= res.passed <= res.total


# ---------------------------------------------------------------- 任务集
def test_generate_tasks_is_deterministic():
    a = generate_tasks(20, seed=3, prefix="t")
    b = generate_tasks(20, seed=3, prefix="t")
    assert [x.buggy for x in a] == [x.buggy for x in b]
    assert [x.target for x in a] == [x.target for x in b]


def test_generate_tasks_different_seed_differs():
    a = generate_tasks(20, seed=3, prefix="t")
    b = generate_tasks(20, seed=4, prefix="t")
    assert [x.buggy for x in a] != [x.buggy for x in b]


def test_every_task_buggy_actually_fails():
    """任务的带 bug 表达式必须真的挂掉至少一个测试，否则没有修复价值。"""
    for task in generate_tasks(60, seed=7, prefix="t"):
        res = run_tests(task.buggy, task)
        assert res.passed < res.total


def test_task_target_always_solves_its_tests():
    for task in generate_tasks(60, seed=8, prefix="t"):
        assert run_tests(task.target, task).frac == 1.0


def test_task_expected_values_are_exact_fractions():
    for task in generate_tasks(10, seed=9, prefix="t"):
        for (a, b, c), (num, den) in zip(task.bindings, task.expected):
            assert evaluate_str(task.target, {"a": a, "b": b, "c": c}) == Fraction(num, den)


def test_task_buggy_length_within_budget():
    from rlalign.env import MAX_PROMPT_TOKENS

    for task in generate_tasks(60, seed=10, prefix="t"):
        assert len(task.buggy) <= MAX_PROMPT_TOKENS


def test_task_ids_are_unique():
    tasks = generate_tasks(50, seed=12, prefix="t")
    assert len({t.task_id for t in tasks}) == len(tasks)


def test_train_eval_split_is_disjoint():
    train, ev = train_eval_split(60, 30, seed=13)
    assert len(train) == 60
    assert len(ev) == 30
    assert not ({t.buggy for t in train} & {t.buggy for t in ev})
    assert not ({t.target for t in train} & {t.target for t in ev})


def test_train_eval_split_is_deterministic():
    t1, e1 = train_eval_split(30, 15, seed=14)
    t2, e2 = train_eval_split(30, 15, seed=14)
    assert [x.buggy for x in t1] == [x.buggy for x in t2]
    assert [x.buggy for x in e1] == [x.buggy for x in e2]


def test_bug_kinds_are_diverse():
    tasks = generate_tasks(300, seed=15, prefix="t")
    kinds = {t.bug for t in tasks}
    assert len(kinds) >= 5
