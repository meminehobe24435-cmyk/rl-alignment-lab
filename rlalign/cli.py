"""命令行入口：``rl <子命令>``。

子命令
------
* ``rl all``        完整闭环：任务集 -> 四条基线 + 两组消融 -> 评测 -> 失败分析 -> 曲线 -> 报告
* ``rl train``      只跑一个算法（ppo / grpo / sft / random）
* ``rl baseline``   只跑随机策略与行为克隆基线
* ``rl eval``       载入检查点在评测集上评估
* ``rl failure``    载入评测记录做失败模式分类
* ``rl report``     由已有指标文件重建结果汇总（results.json / results.md）
* ``rl gradcheck``  有限差分数值梯度校验（打印相对误差）

所有子命令都接受 ``--out`` 指定产物目录，默认 ``reports/``。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .artifacts import load_params, read_json, round_floats, save_params, write_json, write_jsonl
from .env import VOCAB_SIZE, task_to_json
from .eval import evaluate_policy
from .failure import classify, render_table
from .grpo import GRPOConfig
from .policy import PolicyConfig, PolicyNet, ValueNet, param_count_summary
from .ppo import PPOConfig
from .train import RunConfig, build_tasks, policy_cfg, reward_cfg, run_random_baseline, run_rl, train_sft

DEFAULT_OUT = "reports"


# --------------------------------------------------------------------------
# 配置装配
# --------------------------------------------------------------------------
def _run_config(args) -> RunConfig:
    return RunConfig(
        seed=args.seed,
        n_train=args.n_train,
        n_eval=args.n_eval,
        eval_interval=args.eval_interval,
        eval_seed=args.eval_seed,
        outcome_weight=args.outcome_weight,
        process_weight=args.process_weight,
        d_emb=args.d_emb,
        hidden=args.hidden,
        init_from=getattr(args, "init_from", "scratch"),
    )


def _ppo_config(args, iterations: int | None = None) -> PPOConfig:
    return PPOConfig(
        iterations=iterations or args.iterations,
        prompts_per_iter=args.prompts_per_iter,
        inner_epochs=args.inner_epochs,
        lr=args.lr,
        clip_eps=args.clip_eps,
        gamma=args.gamma,
        lam=args.lam,
        kl_coef=args.kl_coef,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        seed=args.seed,
    )


def _grpo_config(args, iterations: int | None = None) -> GRPOConfig:
    return GRPOConfig(
        iterations=iterations or args.iterations,
        prompts_per_iter=args.grpo_prompts_per_iter,
        group_size=args.group_size,
        inner_epochs=args.inner_epochs,
        lr=args.lr,
        clip_eps=args.clip_eps,
        kl_coef=args.kl_coef,
        ent_coef=args.ent_coef,
        eps_std=args.eps_std,
        seed=args.seed,
    )


def _summary_block(ev: dict, ev_greedy: dict | None = None) -> dict:
    out = {
        "n_eval": ev["n"],
        "success_rate": ev["success_rate"],
        "success_rate_stderr": ev["success_rate_stderr"],
        "mean_reward": ev["mean_reward"],
        "mean_outcome": ev["mean_outcome"],
        "mean_process": ev["mean_process"],
        "truncated_rate": ev["truncated_rate"],
    }
    if ev_greedy is not None:
        out["greedy_success_rate"] = ev_greedy["success_rate"]
    return out


def _cfg_dict(obj) -> dict:
    """dataclass -> dict（去掉 seed 之外的私有项）。"""
    from dataclasses import asdict

    return asdict(obj)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def cmd_all(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfg = _run_config(args)
    rc = reward_cfg(cfg)

    print(f"=== rl-alignment-lab v{__version__} 完整闭环 ===")
    print(f"种子={cfg.seed}  训练集={cfg.n_train}  评测集={cfg.n_eval}  "
          f"结果奖励权重={cfg.outcome_weight}  过程奖励权重={cfg.process_weight}")

    train_tasks, eval_tasks = build_tasks(cfg)
    write_json(out / "tasks_train.json", [task_to_json(t) for t in train_tasks])
    write_json(out / "tasks_eval.json", [task_to_json(t) for t in eval_tasks])
    bug_dist: dict[str, int] = {}
    for t in train_tasks:
        bug_dist[t.bug] = bug_dist.get(t.bug, 0) + 1
    print(f"任务就绪：训练 {len(train_tasks)} / 评测 {len(eval_tasks)}（互不相交）")
    print(f"  训练集注入 bug 分布: {dict(sorted(bug_dist.items()))}")

    pcfg = policy_cfg(cfg)
    probe = PolicyNet(pcfg, np.random.default_rng(0))
    probe_critic = ValueNet(pcfg, np.random.default_rng(0))
    pc = param_count_summary(probe, probe_critic)
    print(f"策略网络参数量={pc['policy']:,}  Critic={pc['critic']:,}  合计={pc['total']:,}")

    results: dict = {
        "meta": {
            "version": __version__,
            "seed": cfg.seed,
            "n_train": cfg.n_train,
            "n_eval": cfg.n_eval,
            "eval_seed": cfg.eval_seed,
            "outcome_weight": cfg.outcome_weight,
            "process_weight": cfg.process_weight,
            "vocab_size": VOCAB_SIZE,
            "param_count": pc,
            "train_bug_distribution": dict(sorted(bug_dist.items())),
        },
        "baselines": {},
        "configs": {"ppo": _cfg_dict(_ppo_config(args)), "grpo": _cfg_dict(_grpo_config(args))},
        "curves": {},
        "failure": {},
    }

    # ---- 1. 随机策略 ----
    print("\n[1/6] 随机策略基线")
    rnd = run_random_baseline(cfg, eval_tasks, VOCAB_SIZE, max_gen_len=pcfg.max_gen_len)
    results["baselines"]["random"] = _summary_block(rnd["final_sampled"])
    results["failure"]["random"] = classify(rnd["final_sampled"]["records"])
    write_json(out / "failure_random.json", results["failure"]["random"])
    print(f"    成功率={rnd['final_sampled']['success_rate']:.3f}  "
          f"平均 reward={rnd['final_sampled']['mean_reward']:.4f}")

    # ---- 2. 行为克隆 / SFT ----
    print("\n[2/6] 行为克隆（SFT）基线")
    sft_policy, sft_hist = train_sft(cfg, train_tasks, log=print)
    sft_ev = evaluate_policy(sft_policy, eval_tasks, seed=cfg.eval_seed, reward_cfg=rc)
    sft_ev_g = evaluate_policy(sft_policy, eval_tasks, seed=cfg.eval_seed, reward_cfg=rc, greedy=True)
    results["baselines"]["sft"] = _summary_block(sft_ev, sft_ev_g)
    results["sft_history"] = sft_hist
    results["curves"]["sft"] = [{"iter": h["epoch"], "sft_loss": h["sft_loss"]} for h in sft_hist]
    results["failure"]["sft"] = classify(sft_ev["records"])
    save_params(out / "checkpoint_sft.npz", sft_policy)
    write_json(out / "records_sft.json", sft_ev["records"])
    write_json(out / "failure_sft.json", results["failure"]["sft"])
    print(f"    成功率(采样)={sft_ev['success_rate']:.3f}  成功率(贪心)={sft_ev_g['success_rate']:.3f}  "
          f"平均 reward={sft_ev['mean_reward']:.4f}")

    # ---- 3 & 4. PPO / GRPO 从头训练 ----
    for step, (name, algo) in enumerate([("ppo", "ppo"), ("grpo", "grpo")], start=3):
        print(f"\n[{step}/6] {name.upper()} 从头训练")
        rl_cfg = _ppo_config(args) if algo == "ppo" else _grpo_config(args)
        run = run_rl(algo, cfg, rl_cfg, train_tasks, eval_tasks, init=None, log=print)
        results["baselines"][name] = _summary_block(run["final_sampled"], run["final_greedy"])
        results["curves"][name] = run["metrics"]
        results["failure"][name] = classify(run["final_sampled"]["records"])
        results["baselines"][name]["elapsed_sec"] = run["elapsed_sec"]
        write_jsonl(out / f"metrics_{name}.jsonl", run["metrics"])
        write_json(out / f"failure_{name}.json", results["failure"][name])
        write_json(out / f"records_{name}.json", run["final_sampled"]["records"])
        save_params(out / f"checkpoint_{name}.npz", run["policy"])
        print(f"    {name.upper()} 完成，用时 {run['elapsed_sec']:.1f}s")

    # ---- 5. 从 SFT 权重出发做 RL（额外诊断，不属于四个基线）----
    print("\n[5/6] RL 从 SFT 权重出发（额外诊断）")
    for name, algo in [("ppo_sft_init", "ppo"), ("grpo_sft_init", "grpo")]:
        rl_cfg = _ppo_config(args) if algo == "ppo" else _grpo_config(args)
        run = run_rl(algo, cfg, rl_cfg, train_tasks, eval_tasks, init=sft_policy, log=print)
        results["baselines"][name] = _summary_block(run["final_sampled"], run["final_greedy"])
        results["curves"][name] = run["metrics"]
        results["failure"][name] = classify(run["final_sampled"]["records"])
        results["baselines"][name]["elapsed_sec"] = run["elapsed_sec"]
        write_jsonl(out / f"metrics_{name}.jsonl", run["metrics"])
        write_json(out / f"failure_{name}.json", results["failure"][name])
        save_params(out / f"checkpoint_{name}.npz", run["policy"])

    # ---- 6. 消融：纯结果奖励 vs 结果+过程奖励 ----
    print("\n[6/6] 消融：过程奖励的作用")
    ab_cfg = RunConfig(**{**cfg.__dict__, "process_weight": 0.0})
    run_ab = run_rl("ppo", ab_cfg, _ppo_config(args), train_tasks, eval_tasks, init=None, log=print)
    results["baselines"]["ppo_outcome_only"] = _summary_block(
        run_ab["final_sampled"], run_ab["final_greedy"]
    )
    results["curves"]["ppo_outcome_only"] = run_ab["metrics"]
    results["failure"]["ppo_outcome_only"] = classify(run_ab["final_sampled"]["records"])
    write_jsonl(out / "metrics_ppo_outcome_only.jsonl", run_ab["metrics"])
    write_json(out / "failure_ppo_outcome_only.json", results["failure"]["ppo_outcome_only"])

    results["meta"]["elapsed_sec"] = time.time() - t0
    _write_report(out, results)

    # ---- 曲线（matplotlib 可选）----
    try:
        from .plots import plot_comparison, plot_curves

        figdir = out / "figures"
        plot_curves(results["curves"], figdir / "training_curves.png")
        plot_comparison(results["baselines"], figdir / "success_rate_comparison.png")
        print(f"\n曲线已保存到 {figdir}")
    except RuntimeError as exc:
        print(f"\n跳过绘图：{exc}")

    print(f"\n全部完成，总用时 {results['meta']['elapsed_sec']:.1f}s")
    print(f"结果汇总: {out / 'results.json'}  /  {out / 'results.md'}")
    _print_summary_table(results["baselines"])
    return 0


def _print_summary_table(baselines: dict) -> None:
    print("\n| 方法 | 成功率(采样) | ±1SE | 成功率(贪心) | 平均 reward |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for name, b in baselines.items():
        g = b.get("greedy_success_rate")
        gs = f"{g * 100:.1f}%" if g is not None else "—"
        se = b.get("success_rate_stderr", 0.0)
        print(
            f"| {name} | {b['success_rate'] * 100:.1f}% | ±{se * 100:.1f}% | {gs} | "
            f"{b['mean_reward']:.4f} |"
        )


def _write_report(out: Path, results: dict) -> None:
    write_json(out / "results.json", results)
    lines = ["# rl-alignment-lab 运行结果", ""]
    meta = results["meta"]
    lines += [
        f"- 种子: `{meta['seed']}`，训练集 `{meta['n_train']}`，评测集 `{meta['n_eval']}`",
        f"- 策略网络参数量: `{meta['param_count']['policy']}`，"
        f"Critic: `{meta['param_count']['critic']}`",
        f"- 总用时: `{meta.get('elapsed_sec', 0):.1f}s`",
        "",
        "## 基线对比（同一评测集、同一评测种子）",
        "",
        "| 方法 | 成功率(采样) | ±1SE | 成功率(贪心) | 平均 reward | 结果奖励 | 过程奖励 | 截断率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, b in results["baselines"].items():
        g = b.get("greedy_success_rate")
        gs = f"{g * 100:.1f}%" if g is not None else "—"
        se = b.get("success_rate_stderr", 0.0)
        lines.append(
            f"| {name} | {b['success_rate'] * 100:.1f}% | ±{se * 100:.1f}% | {gs} | "
            f"{b['mean_reward']:.4f} | {b['mean_outcome']:.4f} | {b['mean_process']:.4f} | "
            f"{b['truncated_rate'] * 100:.1f}% |"
        )
    lines.append("")
    lines.append("## 失败模式分类")
    lines.append("")
    for algo, cls in results["failure"].items():
        lines.append(f"### {algo}（成功率 {cls['success_rate'] * 100:.1f}%）")
        lines.append("")
        lines.append("| 失败模式 | 数量 | 占比 |")
        lines.append("| --- | ---: | ---: |")
        for row in cls["table"]:
            lines.append(f"| `{row['bucket']}` | {row['count']} | {row['share'] * 100:.1f}% |")
        lines.append("")
    lines.append("## 超参")
    lines.append("")
    for algo, c in results["configs"].items():
        lines.append(f"### {algo}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(round_floats(c), indent=2, sort_keys=True, ensure_ascii=False))
        lines.append("```")
        lines.append("")
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------
# 其它子命令
# --------------------------------------------------------------------------
def cmd_train(args) -> int:
    out = Path(args.out)
    cfg = _run_config(args)
    train_tasks, eval_tasks = build_tasks(cfg)
    rc = reward_cfg(cfg)
    algo = args.algo
    if algo == "random":
        run = run_random_baseline(cfg, eval_tasks, VOCAB_SIZE)
        summary = _summary_block(run["final_sampled"])
        records = run["final_sampled"]["records"]
    elif algo == "sft":
        policy, _hist = train_sft(cfg, train_tasks, log=print)
        ev = evaluate_policy(policy, eval_tasks, seed=cfg.eval_seed, reward_cfg=rc)
        ev_g = evaluate_policy(policy, eval_tasks, seed=cfg.eval_seed, reward_cfg=rc, greedy=True)
        summary = _summary_block(ev, ev_g)
        records = ev["records"]
        save_params(out / "checkpoint_sft.npz", policy)
    else:
        rl_cfg = _ppo_config(args) if algo == "ppo" else _grpo_config(args)
        run = run_rl(algo, cfg, rl_cfg, train_tasks, eval_tasks, init=None, log=print)
        summary = _summary_block(run["final_sampled"], run["final_greedy"])
        records = run["final_sampled"]["records"]
        write_jsonl(out / f"metrics_{algo}.jsonl", run["metrics"])
        save_params(out / f"checkpoint_{algo}.npz", run["policy"])
    write_json(out / f"summary_{algo}.json", summary)
    write_json(out / f"records_{algo}.json", records)
    write_json(out / f"failure_{algo}.json", classify(records))
    print(json.dumps(round_floats(summary), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def cmd_baseline(args) -> int:
    out = Path(args.out)
    cfg = _run_config(args)
    train_tasks, eval_tasks = build_tasks(cfg)
    rc = reward_cfg(cfg)
    results = {}
    print("== 随机策略 ==")
    rnd = run_random_baseline(cfg, eval_tasks, VOCAB_SIZE)
    results["random"] = _summary_block(rnd["final_sampled"])
    write_json(out / "failure_random.json", classify(rnd["final_sampled"]["records"]))
    print("== 行为克隆 SFT ==")
    policy, hist = train_sft(cfg, train_tasks, log=print)
    ev = evaluate_policy(policy, eval_tasks, seed=cfg.eval_seed, reward_cfg=rc)
    ev_g = evaluate_policy(policy, eval_tasks, seed=cfg.eval_seed, reward_cfg=rc, greedy=True)
    results["sft"] = _summary_block(ev, ev_g)
    results["sft_history"] = hist
    save_params(out / "checkpoint_sft.npz", policy)
    write_json(out / "failure_sft.json", classify(ev["records"]))
    write_json(out / "baselines.json", results)
    print(json.dumps(round_floats(results), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def cmd_eval(args) -> int:
    out = Path(args.out)
    cfg = _run_config(args)
    _tr, eval_tasks = build_tasks(cfg)
    rc = reward_cfg(cfg)
    policy = PolicyNet(policy_cfg(cfg), np.random.default_rng(cfg.seed))
    load_params(out / f"checkpoint_{args.algo}.npz", policy)
    ev = evaluate_policy(policy, eval_tasks, seed=cfg.eval_seed, reward_cfg=rc)
    ev_g = evaluate_policy(policy, eval_tasks, seed=cfg.eval_seed, reward_cfg=rc, greedy=True)
    summary = _summary_block(ev, ev_g)
    write_json(out / f"eval_{args.algo}.json", summary)
    write_json(out / f"records_{args.algo}.json", ev["records"])
    print(json.dumps(round_floats(summary), indent=2, sort_keys=True, ensure_ascii=False))
    return 0


def cmd_failure(args) -> int:
    out = Path(args.out)
    path = out / f"records_{args.algo}.json"
    if not path.exists():
        print(f"缺少评测记录文件: {path}（先跑 rl all 或 rl eval）", file=sys.stderr)
        return 2
    records = read_json(path)
    cls = classify(records)
    write_json(out / f"failure_{args.algo}.json", cls)
    print(render_table(cls))
    print(f"\n失败样本的平均用例通过比例: {cls['failed_mean_test_pass_ratio']:.3f}")
    print("\n按注入 bug 类型：")
    for bug, info in cls["by_injected_bug"].items():
        print(f"  {bug:14s} n={info['n']:3d} 成功率={info['success_rate'] * 100:.1f}%")
    return 0


def cmd_report(args) -> int:
    out = Path(args.out)
    results_path = out / "results.json"
    if not results_path.exists():
        print(f"缺少 {results_path}（先跑 rl all）", file=sys.stderr)
        return 2
    results = read_json(results_path)
    _write_report(out, results)
    print(f"已重建 {out / 'results.md'}")
    return 0


def cmd_gradcheck(args) -> int:
    from .gradcheck import ABS_TOL, REL_TOL, check_policy_gradients, check_value_gradients

    pcfg = PolicyConfig(d_emb=args.d_emb, hidden=args.hidden)
    print("有限差分数值梯度校验（中心差分，eps=1e-6）")
    results = []
    for name, fn in (("策略网络", check_policy_gradients), ("Critic", check_value_gradients)):
        r = fn(pcfg, seed=args.seed, n_per_param=args.n_checks)
        results.append(r)
        print(
            f"  {name}: 最大相对误差 = {r['max_rel_err']:.3e}"
            f"（梯度≥1e-4 的分量 {r['n_significant']} 个 / 小梯度分量 {r['n_small']} 个）"
        )
        print(
            f"    最差相对误差: {r['worst']['param']}[{r['worst']['index']}] "
            f"解析={r['worst']['analytic']:.10f} 数值={r['worst']['numeric']:.10f}"
        )
        print(f"    小梯度分量最大绝对误差 = {r['max_abs_err_of_small_grads']:.3e}")
    ok = all(r["passed"] for r in results)
    print(f"\n判定（相对误差 < {REL_TOL:g}；小梯度分量绝对误差 < {ABS_TOL:g}）: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------
# 参数解析
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rl",
        description="rl-alignment-lab：零依赖（仅 numpy）手写 PPO / GRPO 的可验证奖励训练闭环",
    )
    p.add_argument("--version", action="version", version=f"rl-alignment-lab {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--n-train", type=int, default=1000)
        sp.add_argument("--n-eval", type=int, default=200)
        sp.add_argument("--eval-seed", type=int, default=12345)
        sp.add_argument("--eval-interval", type=int, default=20)
        sp.add_argument("--outcome-weight", type=float, default=1.0)
        sp.add_argument("--process-weight", type=float, default=0.3)
        sp.add_argument("--out", type=str, default=DEFAULT_OUT)
        sp.add_argument("--d-emb", type=int, default=32)
        sp.add_argument("--hidden", type=int, default=128)

    def add_rl(sp):
        sp.add_argument("--iterations", type=int, default=80)
        sp.add_argument("--prompts-per-iter", type=int, default=32)
        sp.add_argument("--grpo-prompts-per-iter", type=int, default=12)
        sp.add_argument("--inner-epochs", type=int, default=2)
        sp.add_argument("--lr", type=float, default=3e-4)
        sp.add_argument("--clip-eps", type=float, default=0.2)
        sp.add_argument("--gamma", type=float, default=1.0)
        sp.add_argument("--lam", type=float, default=0.95)
        sp.add_argument("--kl-coef", type=float, default=0.05)
        sp.add_argument("--ent-coef", type=float, default=0.02)
        sp.add_argument("--vf-coef", type=float, default=0.5)
        sp.add_argument("--group-size", type=int, default=8)
        sp.add_argument("--eps-std", type=float, default=1e-4)

    sp_all = sub.add_parser("all", help="完整闭环（推荐入口）")
    add_common(sp_all)
    add_rl(sp_all)
    sp_all.set_defaults(func=cmd_all)

    sp_tr = sub.add_parser("train", help="只训练一个算法")
    add_common(sp_tr)
    add_rl(sp_tr)
    sp_tr.add_argument("--algo", choices=["ppo", "grpo", "sft", "random"], default="ppo")
    sp_tr.set_defaults(func=cmd_train)

    sp_bl = sub.add_parser("baseline", help="随机策略 + 行为克隆基线")
    add_common(sp_bl)
    sp_bl.set_defaults(func=cmd_baseline)

    sp_ev = sub.add_parser("eval", help="载入检查点评估")
    add_common(sp_ev)
    sp_ev.add_argument("--algo", type=str, default="ppo")
    sp_ev.set_defaults(func=cmd_eval)

    sp_fa = sub.add_parser("failure", help="失败模式分类")
    sp_fa.add_argument("--algo", type=str, default="ppo")
    sp_fa.add_argument("--out", type=str, default=DEFAULT_OUT)
    sp_fa.set_defaults(func=cmd_failure)

    sp_rp = sub.add_parser("report", help="重建结果汇总")
    sp_rp.add_argument("--out", type=str, default=DEFAULT_OUT)
    sp_rp.set_defaults(func=cmd_report)

    sp_gc = sub.add_parser("gradcheck", help="有限差分梯度校验")
    sp_gc.add_argument("--seed", type=int, default=0)
    sp_gc.add_argument("--n-checks", type=int, default=8)
    sp_gc.add_argument("--d-emb", type=int, default=8)
    sp_gc.add_argument("--hidden", type=int, default=12)
    sp_gc.set_defaults(func=cmd_gradcheck)

    return p


def main(argv: list[str] | None = None) -> int:
    # Windows 控制台默认是 GBK(cp936)，中文日志与特殊符号会直接抛
    # UnicodeEncodeError。这里显式把标准流切到 UTF-8 并容错，保证跨平台一致。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover - 极老的/被重定向的流
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
