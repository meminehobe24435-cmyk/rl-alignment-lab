"""数值梯度校验（有限差分）：证明手推的反向传播是对的。

这是本项目最重要的正确性证据。方法很朴素但很严格：

1. 定义一个标量损失 ``L(θ)``（对 logits 的固定随机线性组合，
   它触及网络的每一条前向路径）；
2. **解析梯度**：手推的 ``backward_sequence`` 算出 ``∂L/∂θ``；
3. **数值梯度**：中心差分 ``(L(θ+εe_i) − L(θ−εe_i)) / 2ε``；
4. 逐参数比较。

判定准则（两部分，缺一不可）
---------------------------
中心差分的**绝对误差下限**约为 ``δ·|L| / ε``（δ 为 float64 机器精度，ε 为步长）。
本项目的 ``|L| ~ O(1)``、``ε = 1e-6``，所以数值梯度的绝对噪声约 ``1e-10``。
这决定了：

* **梯度幅度 ≥ 1e-4 的分量**：噪声引起的相对误差上限约 ``1e-10/1e-4 = 1e-6``，
  因此可以严格要求**相对误差 < 1e-5**（这也是 README 里上报的数字）；
* **梯度幅度 < 1e-4 的分量**：相对误差会被噪声完全支配（分母太小），
  此时改用**绝对误差 < 1e-8** 判定 —— 这仍然比真实的反传 bug 小 4~5 个数量级
  （实测两个真 bug 的绝对误差都在 1e-1 量级，被判据直接抓住）。

两条准则一起覆盖全部被抽样的参数分量，且都远严于任何真实 bug 的表现。
"""

from __future__ import annotations

import numpy as np

from .env import BOS_ID, encode
from .policy import PolicyConfig, PolicyNet, ValueNet

__all__ = [
    "numeric_grad",
    "relative_error",
    "check_policy_gradients",
    "check_value_gradients",
    "REL_FLOOR",
    "REL_TOL",
    "ABS_TOL",
]

# 梯度幅度达到该量级才用相对误差判定（噪声相对化后可忽略）
REL_FLOOR = 1e-4
REL_TOL = 1e-5
# 小梯度分量的绝对误差上限（数值噪声约 1e-10，留两个数量级余量）
ABS_TOL = 1e-8


def numeric_grad(f, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """对任意标量函数 ``f`` 关于 ``x`` 的中心差分梯度。"""
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x)
    flat = x.reshape(-1)
    gflat = g.reshape(-1)
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + eps
        lp = float(f(x))
        flat[i] = orig - eps
        lm = float(f(x))
        flat[i] = orig
        gflat[i] = (lp - lm) / (2.0 * eps)
    return g


def relative_error(analytic: np.ndarray, numeric: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """逐元素相对误差（分母加了下限，避免 0/0 导致无意义的 1.0）。"""
    analytic = np.asarray(analytic, dtype=np.float64)
    numeric = np.asarray(numeric, dtype=np.float64)
    denom = np.maximum(floor, np.abs(analytic) + np.abs(numeric))
    return np.abs(analytic - numeric) / denom


def _prompt_from_text(text: str, prompt_len: int) -> np.ndarray:
    ids = encode(text)
    if len(ids) > prompt_len:
        raise ValueError("提示过长")
    out = np.full(prompt_len, 0, dtype=np.int64)
    out[: len(ids)] = ids
    return out


def _drive_check(module, scalar_loss, seed: int, eps: float, n_per_param: int, tag: str) -> dict:
    """通用驱动：对 ``module`` 的每个参数张量抽样做有限差分比对。"""
    pick = np.random.default_rng(seed + 1)
    checks: list[dict] = []
    worst: dict = {"rel_err": -1.0, "param": "", "index": -1, "analytic": 0.0, "numeric": 0.0}
    worst_abs = {"abs_err": -1.0, "param": "", "index": -1, "analytic": 0.0, "numeric": 0.0}
    n_sig = 0
    n_tiny = 0

    for pname, arr, grad in module.parameters():
        flat = arr.reshape(-1)
        gflat = grad.reshape(-1)
        n = flat.size
        k = min(n_per_param, n)
        idxs = pick.choice(n, size=k, replace=False)
        saved = arr.copy()
        for i in idxs:
            i = int(i)
            orig = flat[i]
            flat[i] = orig + eps
            lp = scalar_loss()
            flat[i] = orig - eps
            lm = scalar_loss()
            flat[i] = orig

            num = (lp - lm) / (2.0 * eps)
            ana = float(gflat[i])
            abs_err = abs(ana - num)
            rel = float(relative_error(np.array([ana]), np.array([num]))[0])
            rec = {
                "module": tag,
                "param": pname,
                "index": i,
                "analytic": ana,
                "numeric": num,
                "abs_err": abs_err,
                "rel_err": rel,
            }
            checks.append(rec)

            if max(abs(ana), abs(num)) >= REL_FLOOR:
                n_sig += 1
                if rel > worst["rel_err"]:
                    worst = rec
            else:
                n_tiny += 1
                if abs_err > worst_abs["abs_err"]:
                    worst_abs = rec
        arr[...] = saved

    rel_ok = worst["rel_err"] < REL_TOL
    abs_ok = worst_abs["abs_err"] < ABS_TOL
    return {
        "module": tag,
        "max_rel_err": float(worst["rel_err"]),
        "max_abs_err_of_small_grads": float(worst_abs["abs_err"]),
        "n_checks": len(checks),
        "n_significant": n_sig,
        "n_small": n_tiny,
        "worst": worst,
        "worst_abs": worst_abs,
        "passed": bool(rel_ok and abs_ok),
        "checks": checks,
    }


def check_policy_gradients(
    cfg: PolicyConfig | None = None,
    seed: int = 0,
    eps: float = 1e-6,
    n_per_param: int = 6,
) -> dict:
    """对策略网络（含注意力）的每个参数张量做有限差分校验。"""
    cfg = cfg or PolicyConfig()
    rng = np.random.default_rng(seed)
    policy = PolicyNet(cfg, rng)
    prompt = _prompt_from_text("3*(a+b)-c", cfg.prompt_len)
    T = 6
    toks = np.array([4, 17, 19, 13, 20, 18], dtype=np.int64)[:T]
    coeffs = rng.normal(size=(T, cfg.vocab_size))

    def scalar_loss() -> float:
        logits, _, _ = policy.forward_sequence(prompt, toks)
        return float(np.sum(logits * coeffs))

    policy.zero_grad()
    logits, caches, seq_cache = policy.forward_sequence(prompt, toks)
    policy.backward_sequence(coeffs, caches, seq_cache, prompt)

    return _drive_check(policy, scalar_loss, seed, eps, n_per_param, "policy")


def check_value_gradients(
    cfg: PolicyConfig | None = None,
    seed: int = 0,
    eps: float = 1e-6,
    n_per_param: int = 6,
) -> dict:
    """对 Critic（价值网络）做同样的有限差分校验。"""
    cfg = cfg or PolicyConfig()
    rng = np.random.default_rng(seed)
    critic = ValueNet(cfg, rng)
    h0 = rng.normal(size=cfg.d_emb)
    T = 5
    toks = np.array([4, 17, 19, 13, 20], dtype=np.int64)[:T]
    coeffs = rng.normal(size=T)

    def scalar_loss() -> float:
        return float(np.sum(critic.values_for_sequence(h0, toks) * coeffs))

    critic.zero_grad()
    _v, caches = critic.values_for_sequence(h0, toks, cache=True)
    for t in range(T):
        critic.step_backward(float(coeffs[t]), caches[t])

    return _drive_check(critic, scalar_loss, seed, eps, n_per_param, "critic")
