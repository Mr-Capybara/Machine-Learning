import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import trange
import matplotlib.pyplot as plt


def load_text_dataset(filename, positive='joy', negative='sadness'):
    """
    从文件filename读入文本数据集
    """
    data = pd.read_csv(filename)
    is_positive = data.Emotion == positive
    is_negative = data.Emotion == negative
    data = data[is_positive | is_negative]
    X = data.Text  # 输入文本
    y = np.array(data.Emotion == positive) * 2 - 1  # 1: positive, -1: negative
    return X, y


def vectorize(train, val):
    """
    将训练集和验证集中的文本转成向量表示

    Args：
        train - 训练集，大小为 num_instances 的文本数组
        val - 测试集，大小为 num_instances 的文本数组
    Return：
        train_normalized - 向量化的训练集 (num_instances, num_features)
        val_normalized - 向量化的测试集 (num_instances, num_features)
    """
    tfidf = TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True)
    train_normalized = tfidf.fit_transform(train).toarray()
    val_normalized = tfidf.transform(val).toarray()
    return train_normalized, val_normalized


def linear_svm_subgrad_descent(X, y, alpha=0.05, lambda_reg=0.0001, num_iter=60000, batch_size=16):
    """
    线性SVM的随机次梯度下降

    参数：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        alpha - 浮点数。梯度下降步长，可自行调整为默认值以外的值或扩展为步长策略
        lambda_reg - 正则化系数，可自行调整为默认值以外的值
        num_iter - 要运行的迭代次数，可自行调整为默认值以外的值
        batch_size - 批大小，可自行调整为默认值以外的值

    返回：
        theta_hist - 参数向量的历史，大小的 2D numpy 数组 (num_iter+1, num_features)
        loss hist - 小批量损失函数的历史，数组大小(num_iter)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    theta_hist[0] = theta = np.zeros(num_features)  # Initialize theta
    loss_hist = np.zeros(num_iter)  # Initialize loss_hist

    # TODO 3.5.1
    rng = np.random.RandomState(0)
    for t in trange(num_iter):
        idx = rng.choice(num_instances, size=batch_size, replace=False)
        xb, yb = X[idx], y[idx]
        margin = yb * (xb @ theta)           # (bs,)
        active = margin < 1                  # 样本进入 hinge
        # hinge 部分的次梯度: -y_i * x_i  (仅当 active)
        hinge_grad = -(yb[active, None] * xb[active]).sum(axis=0) / batch_size
        grad = lambda_reg * theta + hinge_grad
        theta = theta - alpha * grad
        theta_hist[t + 1] = theta
        hinge = np.maximum(0.0, 1 - yb * (xb @ theta)).mean()
        loss_hist[t] = 0.5 * lambda_reg * (theta @ theta) + hinge

    return theta_hist, loss_hist


def kernel_svm_subgrad_descent(X, y, alpha=0.1, lambda_reg=0.0001, num_iter=6000, batch_size=16,
                               kernel="rbf", gamma=0.1):
    """
    Kernel SVM的随机次梯度下降（Pegasos-kernel 风格）

    对偶表示 w = Σ_j θ_j Φ(x_j)，优化目标:
        (λ/2) Σ_{i,j} θ_i θ_j K[i,j] + (1/m) Σ_i max(0, 1 - y_i (Kθ)_i)

    参数：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        alpha - 初始步长（未使用时走 1/(λt) 自适应；保留接口兼容）
        lambda_reg - 正则化系数
        num_iter - 迭代次数
        batch_size - 批大小
        kernel - "linear" or "rbf"
        gamma - RBF 核宽度
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.zeros(num_instances)
    theta_hist = np.zeros((num_iter + 1, num_instances))
    loss_hist = np.zeros((num_iter + 1,))

    # TODO 3.5.2
    # 预先计算核矩阵 K[i,j] = k(x_i, x_j)
    if kernel == "linear":
        K = X @ X.T
    elif kernel == "rbf":
        sq = np.sum(X ** 2, axis=1)
        K = np.exp(-gamma * (sq[:, None] + sq[None, :] - 2 * X @ X.T))
    else:
        raise ValueError(f"unknown kernel: {kernel}")

    rng = np.random.RandomState(0)
    for t in trange(num_iter):
        idx = rng.choice(num_instances, size=batch_size, replace=False)
        # 自适应步长 η_t = 1 / (λ * (t+1))，Pegasos 的经典选择
        eta_t = 1.0 / (lambda_reg * (t + 1))

        # 判断 batch 内哪些样本违反 margin
        f_batch = K[idx] @ theta                     # (bs,)
        active = (y[idx] * f_batch) < 1              # (bs,)

        # 稀疏次梯度更新：
        # 1) 正则项梯度 λw 对应 λ·Σ_j θ_j Φ(x_j)，在对偶坐标里等价于 θ ← (1-η_tλ) θ
        # 2) hinge 项梯度 -(1/|B|) Σ_{i∈active} y_i Φ(x_i)
        #    对偶坐标下只修改 θ_{idx[i]} 自己，得 θ_{idx[i]} += (η_t/|B|) y_i
        theta = (1.0 - eta_t * lambda_reg) * theta
        if active.any():
            np.add.at(theta, idx[active],
                      (eta_t / batch_size) * y[idx][active])

        theta_hist[t + 1] = theta

        # 记录 batch 上的 hinge 作为训练进度
        hinge = np.maximum(0.0, 1.0 - y[idx] * (K[idx] @ theta)).mean()
        loss_hist[t + 1] = hinge  # 正则项随训练监控时用不上，省略

    return theta_hist, loss_hist, K


def accuracy_score(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def f1_score_binary(y_true, y_pred, positive=1):
    tp = int(np.sum((y_pred == positive) & (y_true == positive)))
    fp = int(np.sum((y_pred == positive) & (y_true != positive)))
    fn = int(np.sum((y_pred != positive) & (y_true == positive)))
    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    rec = tp / (tp + fn) if tp + fn > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0


def confusion_matrix_binary(y_true, y_pred):
    # 行=真实(+1,-1)，列=预测(+1,-1)
    labels = [1, -1]
    cm = np.zeros((2, 2), dtype=int)
    for i, t in enumerate(labels):
        for j, p in enumerate(labels):
            cm[i, j] = int(np.sum((y_true == t) & (y_pred == p)))
    return cm, labels


def main():
    # 加载所有数据
    X_train, y_train = load_text_dataset("data_train.csv", "joy", "sadness")
    X_val, y_val = load_text_dataset("data_val.csv")
    print("Training Set Size: {} Validation Set Size: {}".format(len(X_train), len(X_val)))
    print("Training Set Text:", X_train, sep='\n')

    # 将训练集和验证集中的文本转成向量表示
    X_train_vect, X_val_vect = vectorize(X_train, X_val)
    X_train_vect = np.hstack((X_train_vect, np.ones((X_train_vect.shape[0], 1))))  # 增加偏置项
    X_val_vect = np.hstack((X_val_vect, np.ones((X_val_vect.shape[0], 1))))  # 增加偏置项

    # SVM的随机次梯度下降训练
    # TODO
    print("\n===== Linear SVM (subgrad) =====")
    theta_hist, loss_hist = linear_svm_subgrad_descent(
        X_train_vect, y_train,
        alpha=0.05, lambda_reg=1e-4, num_iter=20000, batch_size=32)
    w_lin = theta_hist[-1]
    pred_val = np.sign(X_val_vect @ w_lin)
    pred_val[pred_val == 0] = 1
    acc = accuracy_score(y_val, pred_val)
    f1 = f1_score_binary(y_val, pred_val, positive=1)
    cm, labels = confusion_matrix_binary(y_val, pred_val)
    print(f"[Linear] val_acc={acc:.4f} f1={f1:.4f}")
    print(f"confusion matrix (rows=true {labels}, cols=pred {labels}):\n{cm}")

    plt.figure()
    plt.plot(loss_hist)
    plt.xlabel("iter"); plt.ylabel("batch loss"); plt.title("Linear SVM subgrad loss")
    plt.savefig("svm_linear_loss.png", dpi=120); plt.close()

    # Kernel SVM的随机次梯度下降训练
    # TODO
    print("\n===== Kernel SVM (subgrad) =====")
    # 说明：TF-IDF 向量维度 ~5000 且非常稀疏，向量对之间的欧氏距离平方大多在 1~2 级别。
    # 对 RBF 核 exp(-γ||x-x'||²)，γ 若≥0.1 核矩阵会接近对角阵，模型退化为常数预测。
    # 因此这里把 γ 搜得更小（1e-3 ~ 1e-1），并额外对比 linear kernel 作为参照。
    # 同时把迭代数提高到 4000 保证对偶参数收敛。
    sq_train = np.sum(X_train_vect ** 2, axis=1)
    sq_val = np.sum(X_val_vect ** 2, axis=1)

    best = {"acc": -1.0, "pred": None}
    kernel_grid = [
        ("linear", None, 1e-4),
        ("linear", None, 1e-3),
        ("rbf",    1e-3, 1e-4),
        ("rbf",    1e-2, 1e-4),
        ("rbf",    5e-2, 1e-4),
        ("rbf",    1e-1, 1e-4),
    ]
    for kernel, gamma, lam in kernel_grid:
        alpha_hist, loss_hist_k, _ = kernel_svm_subgrad_descent(
            X_train_vect, y_train,
            alpha=0.5, lambda_reg=lam, num_iter=4000, batch_size=32,
            kernel=kernel, gamma=gamma if gamma is not None else 0.0)
        alpha_final = alpha_hist[-1]
        if kernel == "linear":
            K_val = X_val_vect @ X_train_vect.T
        else:
            K_val = np.exp(-gamma * (sq_val[:, None] + sq_train[None, :]
                                     - 2 * X_val_vect @ X_train_vect.T))
        pred = np.sign(K_val @ alpha_final)
        pred[pred == 0] = 1
        acc_k = accuracy_score(y_val, pred)
        f1_k_cur = f1_score_binary(y_val, pred, positive=1)
        tag = f"{kernel}" + (f"(γ={gamma})" if gamma is not None else "")
        print(f"kernel={tag:<14s} lambda={lam:<6g} val_acc={acc_k:.4f} f1={f1_k_cur:.4f}")
        if acc_k > best["acc"]:
            best = {"acc": acc_k, "kernel": kernel, "gamma": gamma,
                    "lambda": lam, "alpha": alpha_final, "pred": pred}
    pred = best["pred"]
    f1_k = f1_score_binary(y_val, pred, positive=1)
    cm_k, _ = confusion_matrix_binary(y_val, pred)
    tag = f"{best['kernel']}" + (f"(γ={best['gamma']})" if best['gamma'] is not None else "")
    print(f"[Kernel-best] {tag} lambda={best['lambda']} "
          f"val_acc={best['acc']:.4f} f1={f1_k:.4f}")
    print(f"confusion matrix:\n{cm_k}")

    # 计算SVM模型在验证集上的准确率，F1-Score以及混淆矩阵
    # TODO
    # —— 选择两个模型中表现更好者作为最终模型 ——
    if acc >= best["acc"]:
        print("\n>> Final model: Linear SVM")
        print(f"acc={acc:.4f}, f1={f1:.4f}, cm=\n{cm}")
    else:
        print("\n>> Final model: Kernel SVM (RBF)")
        print(f"acc={best['acc']:.4f}, f1={f1_k:.4f}, cm=\n{cm_k}")


if __name__ == '__main__':
    main()
