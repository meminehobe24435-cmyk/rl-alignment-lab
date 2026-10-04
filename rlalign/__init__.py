"""rl-alignment-lab：零依赖（仅 numpy）手写 PPO / GRPO 的强化学习对齐实验场。

本包刻意不使用 torch / jax / tensorflow / gym / trl 等任何框架：
所有神经网络层的前向与反向传播都是手推并用 numpy 实现的，
并在 tests/test_gradcheck.py 中用数值梯度（有限差分）逐参数校验。
"""

from __future__ import annotations

# --- 数值确定性前置设置 ---------------------------------------------------
# 必须在 numpy 被导入之前设置，否则 BLAS 线程数会随环境漂移，
# 进而让 float64 的归约顺序变化、破坏"两次运行逐字节一致"的保证。
import os as _os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    _os.environ.setdefault(_var, "1")

__all__ = ["__version__"]

__version__ = "0.1.0"
