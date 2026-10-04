#!/usr/bin/env python
"""可复现性检查：同一命令连跑两次，指标文件必须**逐字节一致**。

用法::

    py -3.12 tools/check_reproducibility.py
    py -3.12 tools/check_reproducibility.py --iterations 5 --n-train 60 --n-eval 40

做法
----
把同一条 ``rl train`` 命令在**两个不同的输出目录**里各跑一遍，
然后逐字节比较所有指标文件（``metrics_*.jsonl``、``summary_*.json``、
``failure_*.json``、``records_*.json``）。

为什么能做到逐字节一致
----------------------
1. 所有随机性都来自 ``np.random.default_rng(seed)``，种子写死；
2. ``rlalign/__init__.py`` 在导入 numpy **之前**把 BLAS/OMP 线程数钉成 1，
   避免多线程归约顺序不同导致 float64 末位漂移；
3. 上报的浮点数统一 round 到 6 位小数，并且 JSON 用排序键 + LF 换行写出。

退出码：0 表示全部一致，1 表示存在差异（CI 会因此失败）。
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 参与比较的产物：只比"指标/结果"类文件。
# 参数检查点（.npz）是 zip 格式，内部带时间戳，本身不可能逐字节一致，
# 且它不是"指标文件"，因此不在比较范围内。
PATTERNS = ("metrics_*.jsonl", "summary_*.json", "failure_*.json", "records_*.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(out_dir: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for pattern in PATTERNS:
        for p in sorted(out_dir.glob(pattern)):
            found[p.name] = sha256(p)
    return found


def run_once(out_dir: Path, args: argparse.Namespace, algo: str) -> int:
    cmd = [
        sys.executable,
        "-m",
        "rlalign.cli",
        "train",
        "--algo",
        algo,
        "--out",
        str(out_dir),
        "--seed",
        str(args.seed),
        "--n-train",
        str(args.n_train),
        "--n-eval",
        str(args.n_eval),
        "--iterations",
        str(args.iterations),
        "--eval-interval",
        str(args.eval_interval),
        "--prompts-per-iter",
        str(args.prompts_per_iter),
        "--grpo-prompts-per-iter",
        str(args.grpo_prompts_per_iter),
        "--group-size",
        str(args.group_size),
        "--inner-epochs",
        str(args.inner_epochs),
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description="检查两次运行是否逐字节一致")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=60)
    ap.add_argument("--n-eval", type=int, default=40)
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--eval-interval", type=int, default=2)
    ap.add_argument("--prompts-per-iter", type=int, default=12)
    ap.add_argument("--grpo-prompts-per-iter", type=int, default=6)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--inner-epochs", type=int, default=2)
    ap.add_argument("--algorithms", type=str, default="ppo,grpo")
    ap.add_argument("--keep", action="store_true", help="保留临时目录以便排查")
    args = ap.parse_args()

    algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    print("=" * 68)
    print("可复现性检查：同一条命令跑两次，指标文件必须逐字节一致")
    print(f"  算法={algos} 种子={args.seed} 训练集={args.n_train} 评测集={args.n_eval} "
          f"迭代={args.iterations}")
    print("=" * 68)

    tmp = Path(tempfile.mkdtemp(prefix="rlalign-repro-"))
    ok = True
    all_files: set[str] = set()
    try:
        for algo in algos:
            print(f"\n[{algo}] 第 1 次运行 ...")
            d1 = tmp / f"{algo}_run1"
            rc1 = run_once(d1, args, algo)
            print(f"[{algo}] 第 2 次运行 ...")
            d2 = tmp / f"{algo}_run2"
            rc2 = run_once(d2, args, algo)

            if rc1 != 0 or rc2 != 0:
                print(f"  ✗ 运行失败（退出码 {rc1} / {rc2}）")
                ok = False
                continue

            h1 = collect(d1)
            h2 = collect(d2)
            if not h1:
                print("  ✗ 没有找到任何指标文件，检查命令是否真的产出了结果")
                ok = False
                continue
            if set(h1) != set(h2):
                print(f"  ✗ 两次运行产出的文件名不同: {sorted(set(h1) ^ set(h2))}")
                ok = False
                continue

            all_files |= set(h1)
            bad = [name for name in sorted(h1) if h1[name] != h2[name]]
            if bad:
                ok = False
                for name in bad:
                    print(f"  ✗ {name} 内容不一致")
                    a = (d1 / name).read_text(encoding="utf-8").splitlines()
                    b = (d2 / name).read_text(encoding="utf-8").splitlines()
                    for i, (x, y) in enumerate(zip(a, b)):
                        if x != y:
                            print(f"      第 {i + 1} 行不同:")
                            print(f"        run1: {x[:160]}")
                            print(f"        run2: {y[:160]}")
                            break
                    if len(a) != len(b):
                        print(f"      行数不同: {len(a)} vs {len(b)}")
            else:
                for name in sorted(h1):
                    print(f"  ✓ {name}  sha256={h1[name][:16]}...")

        print("\n" + "=" * 68)
        if ok:
            print(f"结论：全部逐字节一致 ✓（比较了 {len(all_files)} 个文件：{sorted(all_files)}）")
        else:
            print("结论：存在不一致 ✗")
        print("=" * 68)
    finally:
        if args.keep:
            print(f"临时目录保留在: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
