# topic3: KNN 在 y2/y3 上的表现，重点看数据量的影响
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor as KNR, KNeighborsClassifier as KNC


def load():
    tr = pd.read_csv("linear/train.csv").values
    te = pd.read_csv("linear/test.csv").values
    Xtr, Xte = tr[:, :-3], te[:, :-3]
    lo, hi = Xtr.min(0), Xtr.max(0)
    Xtr = (Xtr - lo) / (hi - lo)
    Xte = (Xte - lo) / (hi - lo)
    return Xtr, Xte, tr[:, -2], te[:, -2], tr[:, -1].astype(int), te[:, -1].astype(int)


def mse(m, X, y): return float(((m.predict(X) - y) ** 2).mean())
def acc(m, X, y): return float((m.predict(X) == y).mean())


def main():
    Xtr, Xte, y2tr, y2te, y3tr, y3te = load()
    metrics = ["euclidean", "manhattan", "cosine"]
    ks      = [1, 3, 5, 10, 30, 100]

    print("===== y2 回归：K / 度量 对比 =====")
    for mt in metrics:
        row = [f"{mt:<10s}"]
        for k in ks:
            row.append(f"k={k}:{mse(KNR(n_neighbors=k, metric=mt).fit(Xtr, y2tr), Xte, y2te):.3f}")
        print(" ".join(row))

    print("\n===== y3 分类：K / 度量 对比 =====")
    for mt in metrics:
        row = [f"{mt:<10s}"]
        for k in ks:
            row.append(f"k={k}:{acc(KNC(n_neighbors=k, metric=mt).fit(Xtr, y3tr), Xte, y3te):.3f}")
        print(" ".join(row))

    # 以下是本题重点：数据量扫描
    ns = [50, 100, 300, 500, 800, 1000]
    sub = [np.random.RandomState(0).permutation(len(Xtr))[:n] for n in ns]

    print("\n===== y2 数据量影响（k=10, euclidean）=====")
    for n, idx in zip(ns, sub):
        v = mse(KNR(n_neighbors=10, metric="euclidean").fit(Xtr[idx], y2tr[idx]), Xte, y2te)
        print(f"n_train={n:<4d} test_mse={v:.4f}")

    print("\n===== y2 数据量 x k 值 （cosine）=====")
    ks2 = [1, 5, 10, 30]
    print(f"{'n_train':<8s} " + " ".join(f"k={k:<3d}" for k in ks2))
    for n, idx in zip(ns, sub):
        row = [f"{n:<8d}"]
        for k in ks2:
            if k > n:
                row.append("  -  "); continue
            row.append(f"{mse(KNR(n_neighbors=k, metric='cosine').fit(Xtr[idx], y2tr[idx]), Xte, y2te):.3f}")
        print(" ".join(row))

    print("\n===== y3 数据量影响（k=10, euclidean）=====")
    for n, idx in zip(ns, sub):
        v = acc(KNC(n_neighbors=10, metric="euclidean").fit(Xtr[idx], y3tr[idx]), Xte, y3te)
        print(f"n_train={n:<4d} test_acc={v:.3f}")


if __name__ == "__main__":
    main()
