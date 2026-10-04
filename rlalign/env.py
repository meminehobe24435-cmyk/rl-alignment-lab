"""迷你表达式语言 + 词法/语法/求值器 + 可验证测试框架 + 任务集生成。

语言定义（刻意做得很小，词表只有 22 个 token）
------------------------------------------------
::

    expr   := term (('+' | '-') term)*
    term   := factor (('*' | '/') factor)*
    factor := number | var | '(' expr ')'
    number := digit+            # 前导 0 允许，"007" 求值为 7
    var    := 'a' | 'b' | 'c'

刻意**不支持**一元负号（``-3``、``a*-b`` 都是语法错误），
因为它在真实代码里也是最常见的低级错误之一，正好作为失败模式。

求值使用 :class:`fractions.Fraction` 做**精确有理数运算**，
所以 "可验证奖励" 是完全精确的：没有浮点容差，不需要 ``np.isclose``。
候选表达式必须对每一组变量取值都得到与目标**精确相等**的有理数。

任务
----
一个任务 = 一段带 bug 的表达式 + 若干组（变量取值, 期望值）。
期望值由目标表达式在该组变量取值下精确求值得到。
修复成功与否**不看字符串是否等于目标**，只看测试是否全部通过 ——
这就是"可验证奖励"（RLVR）的核心：奖励来自执行，不来自人打分或字符串匹配。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

__all__ = [
    "VOCAB",
    "VOCAB_SIZE",
    "PAD_ID",
    "BOS_ID",
    "EOS_ID",
    "CHAR_TO_ID",
    "ID_TO_CHAR",
    "MAX_PROMPT_TOKENS",
    "MAX_GEN_TOKENS",
    "ExprError",
    "LexError",
    "SyntaxError_",
    "EvalError",
    "encode",
    "decode",
    "decode_tokens",
    "parse",
    "evaluate",
    "evaluate_str",
    "parens_balanced",
    "Task",
    "RunResult",
    "run_tests",
    "generate_tasks",
    "train_eval_split",
    "task_to_json",
]

# --------------------------------------------------------------------------
# 词表
# --------------------------------------------------------------------------
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
CHARS = "0123456789+-*/()abc"
VOCAB: tuple[str, ...] = ("<pad>", "<bos>", "<eos>") + tuple(CHARS)
VOCAB_SIZE = len(VOCAB)  # 22
CHAR_TO_ID = {ch: i + 3 for i, ch in enumerate(CHARS)}
ID_TO_CHAR = {i + 3: ch for i, ch in enumerate(CHARS)}

# 目标表达式最多 12 个字符 token，所以 14 步足够生成"表达式 + <eos>"
MAX_GEN_TOKENS = 14
# 带 bug 的提示最长会多出 2 个 token，留出余量
MAX_PROMPT_TOKENS = 16


# --------------------------------------------------------------------------
# 异常
# --------------------------------------------------------------------------
class ExprError(Exception):
    """表达式相关错误的基类。"""


class LexError(ExprError):
    """出现词表之外的字符。"""


class SyntaxError_(ExprError):
    """语法错误（含括号不配平、缺操作数、一元负号等）。"""


class EvalError(ExprError):
    """求值错误（除零、未绑定变量）。"""


# --------------------------------------------------------------------------
# 编解码
# --------------------------------------------------------------------------
def encode(text: str) -> list[int]:
    """字符串 -> token id 序列（跳过空白字符）。"""
    out: list[int] = []
    for ch in text:
        if ch.isspace():
            continue
        tid = CHAR_TO_ID.get(ch)
        if tid is None:
            raise LexError(f"非法字符: {ch!r}")
        out.append(tid)
    return out


def decode(ids: Iterable[int]) -> str:
    """token id 序列 -> 字符串（跳过特殊 token）。"""
    out = []
    for i in ids:
        i = int(i)
        if i in (PAD_ID, BOS_ID, EOS_ID):
            continue
        ch = ID_TO_CHAR.get(i)
        if ch is None:
            raise LexError(f"非法 token id: {i}")
        out.append(ch)
    return "".join(out)


def decode_tokens(ids: Iterable[int]) -> tuple[str, bool]:
    """解码并报告是否在序列中遇到了 ``<eos>``。

    返回值第二项是 ``hit_eos``：为 ``False`` 说明在 ``MAX_GEN_TOKENS`` 内
    没有生成结束符（"超长截断"失败模式）。
    """
    chars: list[str] = []
    hit_eos = False
    for i in ids:
        i = int(i)
        if i == EOS_ID:
            hit_eos = True
            break
        if i in (PAD_ID, BOS_ID):
            continue
        ch = ID_TO_CHAR.get(i)
        if ch is None:
            raise LexError(f"非法 token id: {i}")
        chars.append(ch)
    return "".join(chars), hit_eos


# --------------------------------------------------------------------------
# 词法 / 语法分析
# --------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    """把字符串切成单字符 token，非法字符直接报错。"""
    toks: list[str] = []
    for ch in text:
        if ch.isspace():
            continue
        if ch not in CHAR_TO_ID:
            raise LexError(f"非法字符: {ch!r}")
        toks.append(ch)
    return toks


class _Parser:
    """递归下降解析器：``expr -> term -> factor``。

    解析结果是一棵嵌套元组的 AST：
    ``("num", 7)`` / ``("var", "a")`` / ``("bin", op, left, right)``
    """

    def __init__(self, toks: Sequence[str]):
        self.toks = list(toks)
        self.i = 0

    def _peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _next(self) -> str:
        if self.i >= len(self.toks):
            raise SyntaxError_("表达式意外结束")
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def parse(self):
        if not self.toks:
            raise SyntaxError_("空表达式")
        node = self._expr()
        if self.i != len(self.toks):
            nxt = self.toks[self.i]
            if nxt == ")":
                raise SyntaxError_("多余的右括号")
            raise SyntaxError_(f"位置 {self.i} 处出现多余字符 {nxt!r}")
        return node

    def _expr(self):
        node = self._term()
        while self._peek() in ("+", "-"):
            op = self._next()
            rhs = self._term()
            node = ("bin", op, node, rhs)
        return node

    def _term(self):
        node = self._factor()
        while self._peek() in ("*", "/"):
            op = self._next()
            rhs = self._factor()
            node = ("bin", op, node, rhs)
        return node

    def _factor(self):
        tok = self._peek()
        if tok is None:
            raise SyntaxError_("表达式意外结束，缺少操作数")
        if tok == "(":
            self._next()
            node = self._expr()
            if self._peek() != ")":
                raise SyntaxError_("左括号未闭合")
            self._next()
            return node
        if tok == ")":
            raise SyntaxError_("右括号没有匹配的左括号")
        if tok.isdigit():
            digits = []
            while self._peek() is not None and self._peek().isdigit():
                digits.append(self._next())
            return ("num", int("".join(digits)))
        if tok in ("a", "b", "c"):
            self._next()
            return ("var", tok)
        # 走到这里只可能是运算符或右括号出现在操作数位置
        raise SyntaxError_(f"期望操作数，实际得到 {tok!r}（不支持一元负号）")


def parse(text: str):
    """解析表达式字符串，返回 AST；失败抛 :class:`LexError` / :class:`SyntaxError_`。"""
    return _Parser(_tokenize(text)).parse()


def parens_balanced(text: str) -> bool:
    """仅检查括号配平（不看其它语法）。"""
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


# --------------------------------------------------------------------------
# 求值（精确有理数）
# --------------------------------------------------------------------------
def evaluate(node, env: dict[str, int]) -> Fraction:
    """在给定变量取值下对 AST 精确求值。"""
    kind = node[0]
    if kind == "num":
        return Fraction(node[1])
    if kind == "var":
        name = node[1]
        if name not in env:
            raise EvalError(f"变量 {name!r} 未绑定")
        return Fraction(env[name])
    _, op, left, right = node
    a = evaluate(left, env)
    b = evaluate(right, env)
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if b == 0:
        raise EvalError("除零")
    return a / b


def evaluate_str(text: str, env: dict[str, int]) -> Fraction:
    """解析并求值；解析失败抛 :class:`LexError` / :class:`SyntaxError_`。"""
    return evaluate(parse(text), env)


# --------------------------------------------------------------------------
# 任务与可验证测试框架
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Task:
    """一个代码修复任务。

    Attributes
    ----------
    task_id: 稳定 id（``tr-000`` / ``ev-000``）
    buggy:   带 bug 的表达式（作为模型的条件/提示）
    target:  参考的正确表达式（仅用于过程奖励与失败分析，不作为判定标准）
    bug:     注入的 bug 类型（用于失败分析）
    bindings: 若干组 ``(a, b, c)`` 变量取值
    expected: 每组取值下目标表达式的精确值（分子, 分母）
    """

    task_id: str
    buggy: str
    target: str
    bug: str
    bindings: tuple[tuple[int, int, int], ...]
    expected: tuple[tuple[int, int], ...]

    def envs(self) -> list[dict[str, int]]:
        return [{"a": a, "b": b, "c": c} for a, b, c in self.bindings]


@dataclass(frozen=True)
class RunResult:
    """一次候选表达式在任务上的测试结果。"""

    passed: int
    total: int
    kind: str  # ok | lex | syntax | eval | mismatch
    error: str = ""

    @property
    def frac(self) -> float:
        return self.passed / self.total if self.total else 0.0


def run_tests(candidate: str, task: Task) -> RunResult:
    """运行任务的全部测试用例，返回通过数/总数与失败类型。

    这是"可验证奖励"的唯一裁判：奖励完全由执行结果决定。
    """
    total = len(task.bindings)
    try:
        node = parse(candidate)
    except LexError as exc:
        return RunResult(0, total, "lex", str(exc))
    except SyntaxError_ as exc:
        return RunResult(0, total, "syntax", str(exc))

    passed = 0
    for (a, b, c), (num, den) in zip(task.bindings, task.expected):
        try:
            got = evaluate(node, {"a": a, "b": b, "c": c})
        except EvalError as exc:
            return RunResult(passed, total, "eval", str(exc))
        if got == Fraction(num, den):
            passed += 1
        else:
            # 语义错误：能跑通但结果不对，继续统计后续用例
            pass
    kind = "ok" if passed == total else "mismatch"
    return RunResult(passed, total, kind)


# --------------------------------------------------------------------------
# 任务集生成
# --------------------------------------------------------------------------
def _render(node, parent_prec: int = 0) -> str:
    """AST -> 带最小括号的表达式字符串。"""
    if node[0] == "num":
        return str(node[1])
    if node[0] == "var":
        return node[1]
    _, op, left, right = node
    prec = 1 if op in "+-" else 2
    ls = _render(left, prec)
    # '+' 与 '*' 满足结合律（精确有理数下），右子节点同级不必加括号；
    # '-' 与 '/' 必须加，否则语义会变。
    rp = prec if op in "+*" else prec + 1
    rs = _render(right, rp)
    s = f"{ls}{op}{rs}"
    return f"({s})" if prec < parent_prec else s


def _random_ast(rng: random.Random, n_leaves: int):
    """随机生成一棵恰好有 ``n_leaves`` 个叶子的表达式树。"""
    if n_leaves == 1:
        if rng.random() < 0.55:
            return ("var", rng.choice("abc"))
        return ("num", rng.randint(1, 9))  # 不使用字面量 0，避免目标本身除零
    left_leaves = rng.randint(1, n_leaves - 1)
    right_leaves = n_leaves - left_leaves
    op = rng.choice("+-*/")
    return ("bin", op, _random_ast(rng, left_leaves), _random_ast(rng, right_leaves))


def _mutate(target: str, rng: random.Random) -> tuple[str, str]:
    """对目标表达式注入一个 bug，返回 ``(buggy, bug_kind)``。"""
    chars = list(target)
    ops = [i for i, ch in enumerate(chars) if ch in "+-*/"]
    vars_ = [i for i, ch in enumerate(chars) if ch in "abc"]
    digits = [i for i, ch in enumerate(chars) if ch.isdigit()]
    opens = [i for i, ch in enumerate(chars) if ch == "("]

    kinds = ["swap_op", "swap_var", "swap_digit", "insert_op", "delete_char"]
    if opens:
        kinds.append("drop_paren")
    if ops:
        kinds.append("dup_op")

    kind = rng.choice(kinds)
    if kind == "swap_op" and ops:
        i = rng.choice(ops)
        old = chars[i]
        choices = [c for c in "+-*/" if c != old]
        chars[i] = rng.choice(choices)
    elif kind == "swap_var" and vars_:
        i = rng.choice(vars_)
        old = chars[i]
        choices = [c for c in "abc" if c != old]
        chars[i] = rng.choice(choices)
    elif kind == "swap_digit" and digits:
        i = rng.choice(digits)
        old = chars[i]
        choices = [c for c in "1234567890" if c != old]
        chars[i] = rng.choice(choices)
    elif kind == "drop_paren" and opens:
        # 删除一个左括号（必然造成括号不配平）
        del chars[rng.choice(opens)]
    elif kind == "insert_op":
        i = rng.randint(0, len(chars))
        chars.insert(i, rng.choice("+-*/"))
    elif kind == "dup_op" and ops:
        i = rng.choice(ops)
        chars.insert(i, chars[i])
    else:
        if len(chars) > 1:
            del chars[rng.randrange(len(chars))]

    return "".join(chars), kind


def _sample_bindings(
    target: str, rng: random.Random, n_tests: int, lo: int = 1, hi: int = 6
) -> tuple[tuple[tuple[int, int, int], ...], tuple[tuple[int, int], ...]] | None:
    """采样变量取值并计算期望值；若目标在这些取值下除零则整体重采。"""
    tries = 0
    while tries < 200:
        tries += 1
        bindings = []
        for _ in range(n_tests):
            bindings.append((rng.randint(lo, hi), rng.randint(lo, hi), rng.randint(lo, hi)))
        try:
            values = []
            for a, b, c in bindings:
                v = evaluate_str(target, {"a": a, "b": b, "c": c})
                values.append((v.numerator, v.denominator))
        except ExprError:
            continue
        # 要求取值组合具有区分度，降低"碰巧通过"的概率
        if len({v for v in values}) < 2:
            continue
        return tuple(bindings), tuple(values)
    return None


def generate_tasks(
    n: int,
    seed: int,
    prefix: str = "tr",
    n_tests: int = 4,
    max_target_tokens: int = 12,
) -> list[Task]:
    """确定性地生成 ``n`` 个任务。

    生成流程：随机 AST -> 渲染成字符串 -> 采样变量取值算期望值 -> 注入一个 bug。
    要求：(1) 带 bug 的表达式确实至少挂掉一个测试；(2) 长度在预算内。
    """
    rng = random.Random(seed)
    tasks: list[Task] = []
    guard = 0
    while len(tasks) < n:
        guard += 1
        if guard > n * 400:
            raise RuntimeError("任务生成失败：约束太紧，无法生成足够的任务")

        n_leaves = rng.choice([3, 4, 4, 5])
        target = _render(_random_ast(rng, n_leaves))
        if len(target) > max_target_tokens:
            continue
        try:
            parse(target)
        except ExprError:
            continue
        got = _sample_bindings(target, rng, n_tests)
        if got is None:
            continue
        bindings, expected = got

        buggy = None
        bug_kind = ""
        for _ in range(12):
            cand, kind = _mutate(target, rng)
            if not cand or cand == target:
                continue
            probe = Task("probe", cand, target, kind, bindings, expected)
            res = run_tests(cand, probe)
            if res.passed < res.total:  # bug 真的会挂测试
                buggy, bug_kind = cand, kind
                break
        if buggy is None:
            continue
        if len(buggy) > MAX_PROMPT_TOKENS:
            continue

        tasks.append(
            Task(
                task_id=f"{prefix}-{len(tasks):03d}",
                buggy=buggy,
                target=target,
                bug=bug_kind,
                bindings=bindings,
                expected=expected,
            )
        )
    return tasks


def train_eval_split(
    n_train: int, n_eval: int, seed: int
) -> tuple[list[Task], list[Task]]:
    """构造**互不相交**的训练集与评测集。

    两者用不同的 RNG 种子生成，并且额外做一次字符串去重，
    确保评测集不会出现在训练集里（评测集只在评估时使用）。
    """
    train = generate_tasks(n_train, seed=seed, prefix="tr")
    seen_buggy = {t.buggy for t in train}
    seen_target = {t.target for t in train}
    eval_tasks: list[Task] = []
    bump = 0
    while len(eval_tasks) < n_eval:
        chunk = generate_tasks(n_eval * 2, seed=seed + 10007 + bump, prefix="ev")
        for t in chunk:
            if len(eval_tasks) >= n_eval:
                break
            if t.buggy in seen_buggy or t.target in seen_target:
                continue
            seen_buggy.add(t.buggy)
            seen_target.add(t.target)
            eval_tasks.append(
                Task(
                    task_id=f"ev-{len(eval_tasks):03d}",
                    buggy=t.buggy,
                    target=t.target,
                    bug=t.bug,
                    bindings=t.bindings,
                    expected=t.expected,
                )
            )
        bump += 1
        if bump > 50:
            raise RuntimeError("评测集生成失败")
    return train, eval_tasks


def task_to_json(task: Task) -> dict:
    """序列化为可 JSON 落盘的字典（期望值保持精确分子/分母）。"""
    return {
        "task_id": task.task_id,
        "buggy": task.buggy,
        "target": task.target,
        "bug": task.bug,
        "bindings": [list(b) for b in task.bindings],
        "expected": [list(e) for e in task.expected],
    }
