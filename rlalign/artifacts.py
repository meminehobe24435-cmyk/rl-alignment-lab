"""产物读写：确定性 JSON/JSONL 落盘 + 参数检查点。

**为什么要把浮点数四舍五入到 6 位小数**
--------------------------------------
"同一命令连跑两次，指标文件逐字节一致" 需要跨进程也跨 BLAS 实现稳定。
虽然已经在 ``rlalign/__init__.py`` 里把线程数钉成 1（保证 float64 归约顺序一致），
但不同 BLAS 版本（本地 OpenBLAS、CI 上的参考实现）仍可能在最后几位有差异。
把上报的浮点数统一 round 到 6 位小数，让"逐字节一致"成为一个可验证的工程承诺，
而不是依赖底层的偶然巧合。四舍五入只影响**上报**，不影响训练本身。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

__all__ = ["round_floats", "write_json", "write_jsonl", "read_json", "save_params", "load_params"]

NDIGITS = 6


def round_floats(obj, nd: int = NDIGITS):
    """递归地把浮点数（含 numpy 标量）四舍五入，并转成原生 Python 类型。"""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), nd)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): round_floats(v, nd) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, nd) for v in obj]
    if isinstance(obj, np.ndarray):
        return round_floats(obj.tolist(), nd)
    return obj


def write_json(path: str | Path, obj) -> None:
    """确定性 JSON：排序键、UTF-8、LF 换行、浮点 round 到 6 位。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(round_floats(obj), indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    """确定性 JSONL：每行一个对象，键排序、浮点 round。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(round_floats(r), sort_keys=True, ensure_ascii=False) for r in rows
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_params(path: str | Path, module) -> None:
    """把模块参数按位置存成 npz（用于之后单独跑 eval / failure）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"p{i}": p for i, (_, p, _) in enumerate(module.parameters())}
    np.savez(path, **arrays)


def load_params(path: str | Path, module) -> None:
    """按位置把 npz 写回模块参数（原地赋值，保持数组身份）。"""
    with np.load(path) as data:
        params = [p for _, p, _ in module.parameters()]
        if len(params) != len(data.files):
            raise ValueError(f"检查点参数个数 {len(data.files)} 与模块 {len(params)} 不一致")
        for i, p in enumerate(params):
            p[...] = data[f"p{i}"]
