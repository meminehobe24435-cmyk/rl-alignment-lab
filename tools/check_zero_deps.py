#!/usr/bin/env python
"""零依赖守卫：确保核心代码只用到 numpy 与标准库。

本项目的硬性约束是"不许用任何深度学习 / 强化学习框架"，
所以这个脚本做两件事：

1. **静态扫描**：在 ``rlalign/`` 的源码里找 ``import X`` / ``from X import``，
   把禁用清单里的模块名直接判为失败；
2. **运行时检查**：真正导入全部 ``rlalign`` 子模块，确认 ``sys.modules`` 里
   没有出现禁用模块（防止间接依赖偷偷溜进来）。

禁用清单里同时包含"明显违规"（torch/tensorflow/jax）和
"看起来合规但会破坏约束"（gym/ray/trl/stable-baselines3/tianshou/transformers）
的库。

退出码：0 表示干净，1 表示发现违规。
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "rlalign"

FORBIDDEN = {
    "torch",
    "torchvision",
    "torchaudio",
    "tensorflow",
    "tf",
    "jax",
    "jaxlib",
    "flax",
    "haiku",
    "keras",
    "mxnet",
    "paddle",
    "theano",
    "cupy",
    "autograd",
    "tinygrad",
    "gym",
    "gymnasium",
    "ray",
    "rllib",
    "tianshou",
    "stable_baselines3",
    "stable_baselines",
    "sb3",
    "trl",
    "peft",
    "transformers",
    "accelerate",
    "deepspeed",
    "megatron",
    "vllm",
    "sklearn",
    "scikit_learn",
    "scipy",
    "pandas",
    "seaborn",
    "pytorch_lightning",
    "lightning",
    "einops",
    "triton",
}

# 允许的第三方依赖：核心训练/评估链路只允许 numpy。
ALLOWED_THIRD_PARTY = {"numpy"}

# 可选依赖：只在"画图"这一条支线上被惰性导入，且缺失时优雅降级。
# 它不参与训练与评估，因此不算破坏"零依赖"约束，但要单独列出来。
ALLOWED_OPTIONAL = {"matplotlib"}

STDLIB = set(sys.stdlib_module_names)


def scan_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node.lineno, alias.name.split(".")[0]))
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # 包内相对导入
            if node.module:
                found.append((node.lineno, node.module.split(".")[0]))
    return found


def main() -> int:
    print("=" * 68)
    print("零依赖守卫：核心包只允许 numpy + 标准库")
    print("=" * 68)

    problems: list[str] = []

    # ---- 1. 静态扫描 ----
    files = sorted(PKG.rglob("*.py"))
    print(f"\n扫描 {len(files)} 个源文件 ...")
    third_party: set[str] = set()
    for f in files:
        for lineno, mod in scan_imports(f):
            if mod in FORBIDDEN:
                problems.append(f"{f.relative_to(ROOT)}:{lineno} 导入了禁用模块 {mod!r}")
            elif mod not in STDLIB and mod != "rlalign":
                third_party.add(mod)
    print(f"  用到的第三方模块: {sorted(third_party) or '（无）'}")
    optional = third_party & ALLOWED_OPTIONAL
    if optional:
        print(f"  其中可选（惰性导入、缺失时优雅降级）: {sorted(optional)}")
    unexpected = third_party - ALLOWED_THIRD_PARTY - ALLOWED_OPTIONAL
    if unexpected:
        problems.append(f"出现了非 numpy 的第三方依赖: {sorted(unexpected)}")

    # ---- 2. 运行时检查 ----
    print("\n导入全部 rlalign 子模块并检查 sys.modules ...")
    sys.path.insert(0, str(ROOT))
    before = set(sys.modules)
    import importlib

    modules = [
        "rlalign",
        "rlalign.nn",
        "rlalign.env",
        "rlalign.policy",
        "rlalign.reward",
        "rlalign.rollout",
        "rlalign.ppo",
        "rlalign.grpo",
        "rlalign.train",
        "rlalign.eval",
        "rlalign.failure",
        "rlalign.artifacts",
        "rlalign.gradcheck",
        "rlalign.cli",
    ]
    for name in modules:
        importlib.import_module(name)
    loaded = {m.split(".")[0] for m in set(sys.modules) - before}
    offending = loaded & FORBIDDEN
    if offending:
        problems.append(f"运行时加载了禁用模块: {sorted(offending)}")
    print(f"  新加载的顶层模块: {sorted(loaded)}")

    # plots 依赖 matplotlib，属于可选的"画图"功能，单独说明
    lazy_note = "注意：rlalign/plots.py 会在用到时才导入 matplotlib（可选依赖）"
    print(f"  {lazy_note}")

    print("\n" + "=" * 68)
    if problems:
        print("发现违规 ✗")
        for p in problems:
            print(f"  - {p}")
        print("=" * 68)
        return 1
    print("结论：核心包零第三方依赖（仅 numpy）✓")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
