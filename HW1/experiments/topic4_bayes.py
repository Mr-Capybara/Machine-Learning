# topic4: 按 generate_data.py 重放 y3，算贝叶斯错误率
import numpy as np
import importlib.util as iu

spec = iu.spec_from_file_location("gd", "linear/generate_data.py")
gd = iu.module_from_spec(spec); spec.loader.exec_module(gd)


def main():
    cfg = gd.CONFIG
    n = 100_000
    g = np.random.default_rng(2026)

    # 重放 x
    x = {}
    for nm, sp in cfg["x_specs"].items():
        if sp["dist"] == "normal":
            x[nm] = g.normal(sp["mean"], np.sqrt(sp["var"]), n)
        else:
            x[nm] = g.uniform(sp["low"], sp["high"], n)

    y3 = cfg["y_specs"]["y3"]
    K = y3["num_classes"]
    s = np.zeros((n, K))
    for c, cp in enumerate(y3["class_params"]):
        s[:, c] = cp.get("bias", 0.0) + sum(co * x[nm] for nm, co in cp.get("linear", {}).items())
    s += g.normal(0, y3["noise_std"], s.shape)
    s -= np.array([0.864, 10.548, -0.463, 0.729])

    p = gd.softmax(s)  # temperature=2 已写在 generate_data 里
    best = p.max(1).mean()
    print(f"类别边缘分布:  {p.mean(0)}")
    print(f"Bayes optimal accuracy (argmax): {best:.4f}")
    print(f"Bayes error rate: {1 - best:.4f}")
    print(f"\n与实际测试准确率 0.39 对比：差距 {(best - 0.39) * 100:.1f}pct")


if __name__ == "__main__":
    main()
