"""2.9 话题 2：让 9 个基函数都与 y1 相关时的正则化实验。

构造 y1 = sum_{i=1..3} [a_i * x_i + b_i * cos(x_i) + c_i * x_i^2] + bias + noise
所有 9 个系数都非零，再用 L1/L2 回归看能否恢复出真实系数。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def gen_data(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0.0, 1.0, n)
    x2 = rng.normal(2.0, 2.0, n)
    x3 = rng.uniform(-3.0, 3.0, n)
    # 9 项真实系数（大致同量级）
    true = {
        "x1": 0.8, "x2": -0.6, "x3": 0.7,
        "cos(x1)": 0.5, "cos(x2)": -0.4, "cos(x3)": 0.3,
        "x1^2": 0.2, "x2^2": -0.15, "x3^2": 0.25,
    }
    y = (true["x1"]*x1 + true["x2"]*x2 + true["x3"]*x3
         + true["cos(x1)"]*np.cos(x1) + true["cos(x2)"]*np.cos(x2) + true["cos(x3)"]*np.cos(x3)
         + true["x1^2"]*x1**2 + true["x2^2"]*x2**2 + true["x3^2"]*x3**2
         + 0.5 + 0.05*rng.standard_normal(n))
    Phi = np.column_stack([x1, x2, x3,
                           np.cos(x1), np.cos(x2), np.cos(x3),
                           x1**2, x2**2, x3**2])
    names = ["x1","x2","x3","cos(x1)","cos(x2)","cos(x3)","x1^2","x2^2","x3^2"]
    return Phi, y, names, true


def fit(Phi_tr, y_tr, Phi_va, y_va, reg="l2", lam=1e-2, lr=1e-3, epochs=8000):
    X = torch.tensor(Phi_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr.reshape(-1,1), dtype=torch.float32)
    model = nn.Linear(X.shape[1], 1, bias=True)
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        p = model(X)
        data_loss = mse(p, yt)
        w = model.weight
        reg_loss = torch.abs(w).sum() if reg=="l1" else (w**2).sum()
        (data_loss + lam*reg_loss).backward()
        opt.step()
    with torch.no_grad():
        Xv = torch.tensor(Phi_va, dtype=torch.float32)
        yv = torch.tensor(y_va.reshape(-1,1), dtype=torch.float32)
        val = mse(model(Xv), yv).item()
    return model.weight.detach().numpy().reshape(-1), float(model.bias.item()), val


def main():
    Phi, y, names, true = gen_data()
    # 归一化到 [0,1]
    lo, hi = Phi[:800].min(0), Phi[:800].max(0)
    Phi = (Phi - lo) / (hi - lo)
    Phi_tr, y_tr = Phi[:800], y[:800]
    Phi_va, y_va = Phi[800:], y[800:]
    print("真实系数（对 [0,1] 归一化后的 Phi，系数应该被重新 scale，所以这里只看相对大小/稀疏性）:")
    for k, v in true.items():
        print(f"  {k:10s}: {v:+.3f}")
    for reg in ("l2", "l1"):
        print(f"\n===== {reg.upper()} =====")
        for lam in [1e-4, 1e-2, 1, 5]:
            w, b, val = fit(Phi_tr, y_tr, Phi_va, y_va, reg=reg, lam=lam)
            print(f"--- lam={lam} val_mse={val:.4f} ---")
            for n, wi in zip(names, w):
                flag = "  (真实系数 ≠ 0)" if abs(true[n]) > 1e-9 else ""
                print(f"  {n:10s}: {wi:+.4f}{flag}")


if __name__ == "__main__":
    main()
