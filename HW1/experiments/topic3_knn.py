"""2.9 话题 3：用 KNN 拟合 y2 / y3。

y2 是 35 维稠密线性回归；y3 是 4 类分类（且 y3 本身有贝叶斯错误率上界）。
对比 K、距离度量、数据量对 KNN 的影响。
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier


def load():
    df_tr = pd.read_csv("linear/train.csv")
    df_te = pd.read_csv("linear/test.csv")
    X_tr, X_te = df_tr.values[:, :-3], df_te.values[:, :-3]
    y2_tr, y2_te = df_tr.values[:, -2], df_te.values[:, -2]
    y3_tr, y3_te = df_tr.values[:, -1].astype(int), df_te.values[:, -1].astype(int)
    # min-max 归一化（用训练集统计）
    lo, hi = X_tr.min(0), X_tr.max(0)
    X_tr = (X_tr - lo) / (hi - lo)
    X_te = (X_te - lo) / (hi - lo)
    return X_tr, X_te, y2_tr, y2_te, y3_tr, y3_te


def main():
    X_tr, X_te, y2_tr, y2_te, y3_tr, y3_te = load()

    print("===== y2 回归：K / 度量 对比 =====")
    for metric in ["euclidean", "manhattan", "cosine"]:
        row = [f"{metric:<10s}"]
        for k in [1, 3, 5, 10, 30, 100]:
            m = KNeighborsRegressor(n_neighbors=k, metric=metric)
            m.fit(X_tr, y2_tr)
            mse = float(np.mean((m.predict(X_te) - y2_te)**2))
            row.append(f"k={k}:{mse:.3f}")
        print(" ".join(row))
    # 和线性模型 test_mse=0.042 对比

    print("\n===== y3 分类：K / 度量 对比 =====")
    for metric in ["euclidean", "manhattan", "cosine"]:
        row = [f"{metric:<10s}"]
        for k in [1, 3, 5, 10, 30, 100]:
            m = KNeighborsClassifier(n_neighbors=k, metric=metric)
            m.fit(X_tr, y3_tr)
            acc = float(np.mean(m.predict(X_te) == y3_te))
            row.append(f"k={k}:{acc:.3f}")
        print(" ".join(row))

    print("\n===== y2 数据量影响（k=10, euclidean）=====")
    for n in [50, 100, 300, 500, 800, 1000]:
        idx = np.random.RandomState(0).permutation(len(X_tr))[:n]
        m = KNeighborsRegressor(n_neighbors=10, metric="euclidean")
        m.fit(X_tr[idx], y2_tr[idx])
        mse = float(np.mean((m.predict(X_te) - y2_te)**2))
        print(f"n_train={n:<4d} test_mse={mse:.4f}")


if __name__ == "__main__":
    main()
