"""2.9 话题 4：计算 y3 的贝叶斯错误率。

数据生成流程（抄自 generate_data.py）:
  s[i,c] = bias_c + Σ β_{c,j} x_{i,j}
  s += N(0, 0.1^2)
  s -= [0.864, 10.548, -0.463, 0.729]
  p = softmax(s / 2)
  y ~ Cat(p)

贝叶斯最优：argmax_c p_c(x)；期望准确率 = E_x[max_c p_c(x)]；错误率 = 1 - Acc。
"""
import numpy as np
import pandas as pd
import importlib.util
spec = importlib.util.spec_from_file_location("gd", "linear/generate_data.py")
gd = importlib.util.module_from_spec(spec); spec.loader.exec_module(gd)


def compute_bayes():
    cfg = gd.CONFIG
    n = 100_000
    rng = np.random.default_rng(2026)
    # 重放采样 x
    x = {}
    for name, sp in cfg["x_specs"].items():
        if sp["dist"] == "normal":
            x[name] = rng.normal(sp["mean"], np.sqrt(sp["var"]), n)
        else:
            x[name] = rng.uniform(sp["low"], sp["high"], n)
    # 计算 scores
    spec3 = cfg["y_specs"]["y3"]
    k = spec3["num_classes"]
    scores = np.zeros((n, k))
    for c, cp in enumerate(spec3["class_params"]):
        s = np.full(n, cp.get("bias", 0.0))
        for xn, co in cp.get("linear", {}).items():
            s += co * x[xn]
        scores[:, c] = s
    # 加噪声 + 固定 bias 偏移
    scores += rng.normal(0.0, spec3["noise_std"], scores.shape)
    scores -= np.array([0.864, 10.548, -0.463, 0.729])
    # temperature=2 softmax
    probs = gd.softmax(scores)  # 内部 temperature=2
    # 类别边缘分布
    marg = probs.mean(0)
    print(f"类别边缘分布:  {marg}")
    # 贝叶斯最优准确率（与 argmax 一致）
    best_acc = probs.max(1).mean()
    print(f"Bayes optimal accuracy (argmax): {best_acc:.4f}")
    print(f"Bayes error rate: {1-best_acc:.4f}")
    # 但数据用的是抽样 → 最优分类器在抽样标签上的 acc = E_x Σ_c p_c(x) · 1[c = argmax p]
    #                                             = E_x max_c p_c(x) = 上面同一个值
    # 与 softmax 回归测试准确率 0.39 对比
    print(f"\n与实际测试准确率 0.39 对比：差距 {(best_acc - 0.39)*100:.1f}pct")


if __name__ == "__main__":
    compute_bayes()
