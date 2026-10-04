"""训练曲线绘制。

matplotlib 只在真正要画图时才导入，核心训练流程不依赖它 ——
这样 CI 里只需要 ``numpy`` + ``pytest`` 就能跑通测试、短训练与可复现性检查。

中文字体在 Linux CI 上通常不存在，matplotlib 会刷一堆 "Glyph missing" 警告
并且画出空心方块。这里先探测可用的中日韩字体，找不到就自动退回英文标签。
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["plot_curves", "plot_comparison", "has_cjk_font"]

_COLORS = {
    "ppo": "#1f77b4",
    "grpo": "#d62728",
    "sft": "#2ca02c",
    "random": "#7f7f7f",
    "ppo_sft_init": "#9467bd",
    "grpo_sft_init": "#8c564b",
    "ppo_outcome_only": "#e377c2",
}

_CJK_FONTS = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "PingFang SC",
]

# 每张图的标题：(中文, 英文)
_TITLES = {
    "eval_success_rate": ("评测集成功率", "Eval success rate"),
    "train_mean_reward": ("训练批平均 reward", "Train batch mean reward"),
    "kl_k3": ("KL(π‖π_ref) k3 估计", "KL(pi||pi_ref), k3 estimate"),
    "entropy": ("策略熵", "Policy entropy"),
    "value_loss": ("Critic 价值损失", "Critic value loss"),
    "group_reward_var": ("组内 reward 方差", "Within-group reward variance"),
}


def has_cjk_font() -> bool:
    """当前环境是否有可用的中日韩字体。"""
    try:
        from matplotlib import font_manager
    except Exception:  # pragma: no cover
        return False
    try:
        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:  # pragma: no cover
        return False
    return any(name in available for name in _CJK_FONTS)


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover - 环境缺 matplotlib 时走这里
        raise RuntimeError(f"需要 matplotlib 才能画图（核心训练不依赖它）: {exc}") from exc


def _prepare(plt) -> bool:
    """配置字体，返回是否可以用中文标签。"""
    try:
        from matplotlib import font_manager

        available = {f.name for f in font_manager.fontManager.ttflist}
    except Exception:  # pragma: no cover
        return False
    for name in _CJK_FONTS:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + list(plt.rcParams["font.sans-serif"])
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


def _series(metrics: list[dict], key: str):
    xs, ys = [], []
    for row in metrics:
        if key in row and row[key] is not None:
            xs.append(row["iter"])
            ys.append(row[key])
    return xs, ys


def plot_curves(metrics_by_algo: dict[str, list[dict]], out_path: str | Path) -> None:
    """画 6 张诊断曲线：成功率、平均 reward、KL、熵、价值损失、组内 reward 方差。"""
    plt = _plt()
    zh = _prepare(plt)
    # 不用 tight_layout / constrained_layout：两者在标签较长时都会告警
    # （"margins cannot be made large enough" / "axes sizes collapsed to zero"），
    # 显式留边距更可控，结果也稳定。
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        ("eval_success_rate", axes[0][0]),
        ("train_mean_reward", axes[0][1]),
        ("kl_k3", axes[0][2]),
        ("entropy", axes[1][0]),
        ("value_loss", axes[1][1]),
        ("group_reward_var", axes[1][2]),
    ]
    for key, ax in panels:
        drew = False
        for algo, metrics in metrics_by_algo.items():
            xs, ys = _series(metrics, key)
            if xs:
                ax.plot(xs, ys, marker="o", markersize=3, label=algo, color=_COLORS.get(algo))
                drew = True
        ax.set_title(_TITLES[key][0] if zh else _TITLES[key][1])
        ax.set_xlabel("迭代 / iteration" if zh else "iteration")
        ax.grid(alpha=0.3)
        if drew:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "无数据" if zh else "no data", ha="center",
                    va="center", transform=ax.transAxes, color="gray")
    fig.suptitle(
        "PPO / GRPO 训练诊断曲线（真实运行结果）" if zh else "PPO / GRPO training diagnostics (real runs)"
    )
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.06, right=0.98, hspace=0.32, wspace=0.24)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_comparison(summary: dict[str, dict], out_path: str | Path) -> None:
    """画各方法成功率对比柱状图（带 ±1 二项标准误）。"""
    plt = _plt()
    zh = _prepare(plt)
    names = list(summary.keys())
    sampled = [summary[n]["success_rate"] * 100 for n in names]
    greedy = [summary[n].get("greedy_success_rate") for n in names]
    has_greedy = any(g is not None for g in greedy)
    greedy = [0.0 if g is None else g * 100 for g in greedy]
    errs = [summary[n].get("success_rate_stderr", 0.0) * 100 for n in names]

    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(names)), 5.2))
    x = range(len(names))
    if has_greedy:
        ax.bar([i - 0.2 for i in x], sampled, width=0.4,
               label="采样解码 / sampling" if zh else "sampling",
               yerr=errs, capsize=3, color="#4c72b0")
        ax.bar([i + 0.2 for i in x], greedy, width=0.4,
               label="贪心解码 / greedy" if zh else "greedy", color="#dd8452")
    else:
        ax.bar(list(x), sampled, width=0.5,
               label="采样解码 / sampling" if zh else "sampling",
               yerr=errs, capsize=3, color="#4c72b0")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("评测集成功率 (%)" if zh else "eval success rate (%)")
    ax.set_title(
        "各方法在固定评测集上的成功率（误差棒 = ±1 二项标准误）"
        if zh
        else "Success rate on the fixed eval set (error bars = ±1 binomial SE)"
    )
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    for i, s in enumerate(sampled):
        ax.text(i - 0.2 if has_greedy else i, s + 0.3, f"{s:.1f}", ha="center", fontsize=8)
    # x 轴标签旋转后需要更多底部空间，显式预留
    fig.subplots_adjust(top=0.88, bottom=0.26, left=0.08, right=0.98)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
