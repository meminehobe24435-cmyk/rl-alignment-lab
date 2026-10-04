"""命令行入口：子命令解析与端到端可运行性。"""

from __future__ import annotations

import json

import pytest

from rlalign.cli import build_parser, main

TINY = ["--n-train", "30", "--n-eval", "12", "--iterations", "3", "--eval-interval", "3"]


def test_parser_exposes_all_subcommands():
    parser = build_parser()
    for name in ("all", "train", "baseline", "eval", "failure", "report", "gradcheck"):
        assert name in parser._subparsers._group_actions[0].choices


def test_parser_requires_subcommand():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_version_flag_exits():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_gradcheck_subcommand_passes(tmp_path):
    rc = main(["gradcheck", "--n-checks", "4", "--d-emb", "6", "--hidden", "10"])
    assert rc == 0


def test_random_baseline_subcommand(tmp_path):
    out = tmp_path / "r"
    rc = main(["train", "--algo", "random", "--out", str(out), "--n-train", "20", "--n-eval", "10"])
    assert rc == 0
    summary = json.loads((out / "summary_random.json").read_text(encoding="utf-8"))
    assert summary["success_rate"] == 0.0
    assert (out / "failure_random.json").exists()


def test_sft_subcommand_writes_checkpoint(tmp_path):
    out = tmp_path / "s"
    rc = main(["train", "--algo", "sft", "--out", str(out), "--n-train", "20", "--n-eval", "10"])
    assert rc == 0
    assert (out / "checkpoint_sft.npz").exists()
    assert (out / "records_sft.json").exists()
    summary = json.loads((out / "summary_sft.json").read_text(encoding="utf-8"))
    assert 0.0 <= summary["success_rate"] <= 1.0


def test_ppo_subcommand_writes_metrics(tmp_path):
    out = tmp_path / "p"
    rc = main(
        ["train", "--algo", "ppo", "--out", str(out), "--n-train", "20", "--n-eval", "10",
         "--iterations", "3", "--eval-interval", "3"]
    )
    assert rc == 0
    lines = (out / "metrics_ppo.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    row = json.loads(lines[0])
    assert "policy_loss" in row and "value_loss" in row and "kl_k3" in row
    assert (out / "checkpoint_ppo.npz").exists()


def test_grpo_subcommand_writes_metrics_with_group_variance(tmp_path):
    out = tmp_path / "g"
    rc = main(
        ["train", "--algo", "grpo", "--out", str(out), "--n-train", "20", "--n-eval", "10",
         "--iterations", "3", "--eval-interval", "3", "--group-size", "4",
         "--grpo-prompts-per-iter", "3"]
    )
    assert rc == 0
    row = json.loads((out / "metrics_grpo.jsonl").read_text(encoding="utf-8").strip().splitlines()[0])
    assert "group_reward_var" in row
    assert "value_loss" not in row  # GRPO 没有 Critic


def test_eval_subcommand_roundtrip(tmp_path):
    out = tmp_path / "e"
    main(["train", "--algo", "ppo", "--out", str(out), "--n-train", "20", "--n-eval", "10",
          "--iterations", "2", "--eval-interval", "2"])
    rc = main(["eval", "--algo", "ppo", "--out", str(out), "--n-train", "20", "--n-eval", "10"])
    assert rc == 0
    ev = json.loads((out / "eval_ppo.json").read_text(encoding="utf-8"))
    assert "success_rate" in ev and "success_rate_stderr" in ev


def test_failure_subcommand(tmp_path, capsys):
    out = tmp_path / "f"
    main(["train", "--algo", "random", "--out", str(out), "--n-train", "20", "--n-eval", "10"])
    rc = main(["failure", "--algo", "random", "--out", str(out)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "unbalanced_paren" in text or "syntax" in text
    assert (out / "failure_random.json").exists()


def test_failure_subcommand_missing_file_returns_2(tmp_path):
    rc = main(["failure", "--algo", "nope", "--out", str(tmp_path)])
    assert rc == 2


def test_report_subcommand_missing_file_returns_2(tmp_path):
    rc = main(["report", "--out", str(tmp_path)])
    assert rc == 2


def test_report_subcommand_rebuilds_markdown(tmp_path):
    out = tmp_path / "rp"
    main(["train", "--algo", "random", "--out", str(out), "--n-train", "20", "--n-eval", "10"])
    from rlalign.artifacts import write_json

    write_json(
        out / "results.json",
        {
            "meta": {"seed": 0, "n_train": 20, "n_eval": 10, "param_count": {"policy": 1, "critic": 2}},
            "baselines": {},
            "failure": {},
            "configs": {},
        },
    )
    rc = main(["report", "--out", str(out)])
    assert rc == 0
    assert "rl-alignment-lab 运行结果" in (out / "results.md").read_text(encoding="utf-8")


def test_all_subcommand_end_to_end(tmp_path):
    out = tmp_path / "all"
    rc = main(["all", "--out", str(out), *TINY])
    assert rc == 0
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    for key in ("random", "sft", "ppo", "grpo", "ppo_sft_init", "grpo_sft_init", "ppo_outcome_only"):
        assert key in results["baselines"], f"缺少基线 {key}"
    assert (out / "results.md").exists()
    assert (out / "tasks_train.json").exists()
    assert (out / "tasks_eval.json").exists()
    # 指标文件必须存在且非空
    for algo in ("ppo", "grpo", "ppo_sft_init", "grpo_sft_init", "ppo_outcome_only"):
        assert (out / f"metrics_{algo}.jsonl").stat().st_size > 0


def test_all_uses_outcome_only_config_for_ablation(tmp_path):
    """消融必须真的把过程奖励权重设成 0，否则它就不是消融。"""
    out = tmp_path / "ab"
    main(["all", "--out", str(out), *TINY])
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    # 纯结果奖励下随机初始化的策略拿不到任何信号
    assert results["baselines"]["ppo_outcome_only"]["mean_reward"] == 0.0
