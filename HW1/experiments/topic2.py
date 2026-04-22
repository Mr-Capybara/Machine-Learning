# topic2: 9 项全非零时，L1/L2 能不能学回真实系数
import numpy as np
import torch, torch.nn as nn

NAMES = ["x1","x2","x3","cos(x1)","cos(x2)","cos(x3)","x1^2","x2^2","x3^2"]
TRUE  = [0.8, -0.6, 0.7, 0.5, -0.4, 0.3, 0.2, -0.15, 0.25]
BIAS  = 0.5


def make(n=1000, seed=0):
    g = np.random.default_rng(seed)
    x1 = g.normal(0, 1, n)
    x2 = g.normal(2, 2, n)
    x3 = g.uniform(-3, 3, n)
    phi = np.c_[x1, x2, x3, np.cos(x1), np.cos(x2), np.cos(x3), x1**2, x2**2, x3**2]
    y = phi @ np.array(TRUE) + BIAS + 0.05 * g.standard_normal(n)
    return phi, y


def fit(Xtr, ytr, Xva, yva, reg, lam, lr=1e-3, steps=8000):
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr[:, None], dtype=torch.float32)
    net = nn.Linear(Xt.shape[1], 1)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((net(Xt) - yt) ** 2).mean()
        w = net.weight
        loss = loss + lam * (w.abs().sum() if reg == "l1" else (w ** 2).sum())
        loss.backward(); opt.step()
    with torch.no_grad():
        Xv = torch.tensor(Xva, dtype=torch.float32)
        val = ((net(Xv).squeeze() - torch.tensor(yva, dtype=torch.float32)) ** 2).mean().item()
    return net.weight.detach().numpy().ravel(), val


def main():
    torch.manual_seed(0)
    phi, y = make()
    # [0,1] 归一化，用训练集统计
    lo, hi = phi[:800].min(0), phi[:800].max(0)
    phi = (phi - lo) / (hi - lo)
    Xtr, ytr, Xva, yva = phi[:800], y[:800], phi[800:], y[800:]

    print("真实系数（对 [0,1] 归一化后的 Phi，系数应该被重新 scale，所以这里只看相对大小/稀疏性）:")
    for n, v in zip(NAMES, TRUE):
        print(f"  {n:10s}: {v:+.3f}")

    for reg in ("l2", "l1"):
        print(f"\n===== {reg.upper()} =====")
        for lam in [1e-4, 1e-2, 1, 5]:
            w, val = fit(Xtr, ytr, Xva, yva, reg, lam)
            print(f"--- lam={lam} val_mse={val:.4f} ---")
            for n, wi, tv in zip(NAMES, w, TRUE):
                tag = "  (真实系数 ≠ 0)" if abs(tv) > 1e-9 else ""
                print(f"  {n:10s}: {wi:+.4f}{tag}")


if __name__ == "__main__":
    main()
