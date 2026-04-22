import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from softmax_util import train_multiclass_softmax_with_model_selection, evaluate_multiclass_softmax

def split_data(X, y1, y2, y3, split_size=[0.8, 0.2], shuffle=False, random_seed=None):
    """
    对数据集进行划分

    Args：
        X - 特征向量，一个大小为 (num_instances, num_features) 的二维 numpy 数组
        yi - 标签向量，一个大小为 (num_instances) 的一维 numpy 数组
        split_size - 划分比例，期望为一个浮点数列表，如[0.8, 0.2]表示将数据集划分为两部分，比例为80%和20%
        shuffle - 是否打乱数据集
        random_seed - 随机种子
        
    Return：
        X_list - 划分后的特征向量列表
        yi_list - 划分后的标签向量列表
    """
    assert sum(split_size) == 1
    num_instances = X.shape[0]
    if shuffle:
        rng = np.random.RandomState(random_seed)
        indices = rng.permutation(num_instances)
        X = X[indices]
        y1 = y1[indices]
        y2 = y2[indices]
        y3 = y3[indices]
    
    # TODO 2.1.1 （about 7 lines)
    

def feature_normalization(train, val, test):
    """将训练集中的所有特征值映射至[0,1]，对测试集上的每个特征也需要使用相同的仿射变换

    Args：
        train - 训练集，一个大小为 (num_instances, num_features) 的二维 numpy 数组
        val - 验证集，一个大小为 (num_instances, num_features) 的二维 numpy 数组
        test - 测试集，一个大小为 (num_instances, num_features) 的二维 numpy 数组
    Return：
        train_normalized - 特征归一化后的训练集
        val_normalized - 特征归一化后的验证集
        test_normalized - 特征归一化后的测试集

    """
    # TODO 2.1.2 (about 8 lines)


def build_basis_features(X_raw):
    """
    只使用 x1, x2, x3 构造9维基函数:
    [x1, x2, x3, sin(x1), sin(x2), sin(x3), x1^2, x2^2, x3^2]
    X_raw: shape (n, num_features), 假设前3列是 x1,x2,x3

    Returns
    -------
    Phi : numpy.ndarray, shape (n_samples, 9), dtype=float
        基函数变换后的设计矩阵（每行对应一个样本，每列对应一个基函数），
        列顺序固定为：
        1) x1
        2) x2
        3) x3
        4) sin(x1)
        5) sin(x2)
        6) sin(x3)
        7) x1^2
        8) x2^2
        9) x3^2

    feature_names : list[str], length=9
        `Phi` 各列对应的特征名称列表，顺序与 `Phi` 列严格一致，例如：
        ["x1", "x2", "x3", "cos(x1)", "cos(x2)", "cos(x3)", "x1^2", "x2^2", "x3^2"]
    """
    x1 = X_raw[:, 0]
    x2 = X_raw[:, 1]
    x3 = X_raw[:, 2]

    # TODO 2.2 基函数 （3~10行）


def train_linear_with_regularization(
    X_train, y_train, X_val, y_val,
    reg_type="l2", # "l1" or "l2"
    lambda_reg=1e-2,
    lr=1e-2,
    epochs=5000,
    verbose_every=500
):
    """
    线性模型: y = w^T x + b
    损失: MSE + lambda_reg * (L1或L2正则)
    """
    device = torch.device("cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32, device=device)

    model = nn.Linear(X_train.shape[1], 1, bias=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        pred = model(X_train_t)
        data_loss = mse(pred, y_train_t)

        w = model.weight  # shape (1, d)
        if reg_type == "l1":
            reg_loss = torch.abs(w).sum()
        elif reg_type == "l2":
            reg_loss = (w ** 2).sum()
        else:
            raise ValueError("reg_type must be 'l1' or 'l2'")

        loss = data_loss + lambda_reg * reg_loss
        loss.backward()
        optimizer.step()

        if epoch % verbose_every == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_mse = mse(val_pred, y_val_t).item()
            print(f"[{reg_type}] epoch={epoch:4d} train_obj={loss.item():.6f} val_mse={val_mse:.6f}")

    # 输出收敛后的权重
    w_final = model.weight.detach().cpu().numpy().reshape(-1)
    b_final = float(model.bias.detach().cpu().numpy().reshape(-1)[0])

    return model, w_final, b_final


def compute_regularized_square_loss(X, y, theta, lambda_reg):
    """
    给定一组 X, y, theta，计算用 X*theta 预测 y 的岭回归损失函数

    Args：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        theta - 参数向量，数组大小 (num_features)
        lambda_reg - 正则化系数

    Return：
        loss - 损失函数，标量
    """
    # TODO 2.3.2 (2~7 lines)



def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    计算岭回归损失函数的梯度

    参数：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        theta - 参数向量，数组大小（num_features）
        lambda_reg - 正则化系数

    返回：
        grad - 梯度向量，数组大小（num_features）
    """
    # TODO 2.3.4 (2~7 lines)


def grad_checker(X, y, theta, lambda_reg, epsilon=0.01, tolerance=1e-4):
    """梯度检查
    如果实际梯度和近似梯度的欧几里得距离超过容差，则梯度计算不正确。

    Args：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        theta - 参数向量，数组大小（num_features）
        lambda_reg - 正则化系数
        epsilon - 步长
        tolerance - 容差

    Return：
        梯度是否正确

    """
    grad_computed = compute_regularized_square_loss_gradient(X, y, theta, lambda_reg)
    num_features = theta.shape[0]
    grad_approx = np.zeros(num_features)

    for h in np.identity(num_features):
        J0 = compute_regularized_square_loss(X, y, theta - epsilon * h, lambda_reg)
        J1 = compute_regularized_square_loss(X, y, theta + epsilon * h, lambda_reg)
        grad_approx += (J1 - J0) / (2 * epsilon) * h
    dist = np.linalg.norm(grad_approx - grad_computed)
    return dist <= tolerance


def grad_descent(X, y, lambda_reg, alpha=0.1, num_iter=1000, check_gradient=False):
    """
    全批量梯度下降算法

    Args：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        lambda_reg - 正则化系数，可自行调整为默认值以外的值
        alpha - 梯度下降的步长，可自行调整为默认值以外的值
        num_iter - 要运行的迭代次数，可自行调整为默认值以外的值
        check_gradient - 更新时是否检查梯度

    Return：
        theta_hist - 存储迭代中参数向量的历史，大小为 (num_iter+1, num_features) 的二维 numpy 数组
        loss_hist - 全批量损失函数的历史，大小为 (num_iter) 的一维 numpy 数组
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    theta_hist[0] = theta = np.zeros(num_features)  # Initialize theta
    loss_hist = np.zeros(num_iter)  # Initialize loss_hist
    for i in range(num_iter):  
        # TODO 2.4.2 (3~5 lines)
        pass

    return theta_hist,loss_hist



def stochastic_grad_descent(X_train, y_train, X_val, y_val, lambda_reg, alpha=0.1, num_iter=1000, batch_size=1):
    """
    随机梯度下降，并随着训练过程在验证集上验证

    参数：
        X_train - 训练集特征向量，数组大小 (num_instances, num_features)
        y_train - 训练集标签向量，数组大小 (num_instances)
        X_val - 验证集特征向量，数组大小 (num_instances, num_features)
        y_val - 验证集标签向量，数组大小 (num_instances)
        alpha - 梯度下降的步长，可自行调整为默认值以外的值
        lambda_reg - 正则化系数，可自行调整为默认值以外的值
        num_iter - 要运行的迭代次数，可自行调整为默认值以外的值
        batch_size - 批大小，可自行调整为默认值以外的值

    返回：
        theta_hist - 参数向量的历史，大小的 2D numpy 数组 (num_iter+1, num_features)
        loss hist - 小批量正则化损失函数的历史，数组大小(num_iter)
        validation hist - 验证集上全批量均方误差（不带正则化项）的历史，数组大小(num_iter)
    """
    num_instances, num_features = X_train.shape[0], X_train.shape[1]
    theta_hist = np.zeros((num_iter + 1, num_features))  # Initialize theta_hist
    theta_hist[0] = theta = np.zeros(num_features)  # Initialize theta
    loss_hist = np.zeros(num_iter)  # Initialize loss_hist
    validation_hist = np.zeros(num_iter)  # Initialize validation_hist

    # TODO 2.6.2

    return theta_hist,loss_hist,validation_hist


def main():
    # 加载数据集
    print('loading the dataset')

    df = pd.read_csv('train.csv', delimiter=',')
    X = df.values[:, :-3]
    y1 = df.values[:, -3]
    y2 = df.values[:, -2]
    y3 = df.values[:, -1]

    print('Split into Train and Val')
    (X_train_raw, X_val_raw), (y1_train, y1_val), (y2_train, y2_val), (y3_train, y3_val) = split_data(X, y1, y2, y3, split_size=[0.8, 0.2], shuffle=True, random_seed=0)

    df_test = pd.read_csv('test.csv', delimiter=',')
    X_test_raw = df_test.values[:, :-3]
    y1_test = df_test.values[:, -3]
    y2_test = df_test.values[:, -2]
    y3_test = df_test.values[:, -1]

    print("Scaling all to [0, 1]")
    X_train, X_val, X_test = feature_normalization(X_train_raw, X_val_raw, X_test_raw)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # 增加偏置项
    X_val = np.hstack((X_val, np.ones((X_val.shape[0], 1))))  # 增加偏置项
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))  # 增加偏置项

    # 2.2 基函数与正则化
    # ========== 基函数特征 ==========
    Phi_train, feature_names = build_basis_features(X_train_raw[:, :3])
    Phi_val, _ = build_basis_features(X_val_raw[:, :3])
    Phi_test, _ = build_basis_features(X_test_raw[:, :3])
    # Phi_train, Phi_val, Phi_test = feature_normalization(Phi_train, Phi_val, Phi_test)

    # 目标：y1
    target_train = y1_train
    target_val = y1_val

    # ========== L2 正则实验 ==========
    print("\n===== L2 Regularization =====")
    for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1, 5]:
        print(f"\n--- lambda={lam} ---")
        _, w_l2, b_l2 = train_linear_with_regularization(
            Phi_train, target_train, Phi_val, target_val,
            reg_type="l2", lambda_reg=lam, lr=1e-3, epochs=10000, verbose_every=1000
        )
        print("bias:", b_l2)
        print("weights:")
        for n, w in zip(feature_names, w_l2):
            print(f"  {n:8s}: {w:+.6f}")

    # ========== L1 正则实验 ==========
    print("\n===== L1 Regularization =====")
    for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1, 5]:
        print(f"\n--- lambda={lam} ---")
        _, w_l1, b_l1 = train_linear_with_regularization(
            Phi_train, target_train, Phi_val, target_val,
            reg_type="l1", lambda_reg=lam, lr=1e-3, epochs=10000, verbose_every=1000
        )
        print("bias:", b_l1)
        print("weights:")
        for n, w in zip(feature_names, w_l1):
            print(f"  {n:8s}: {w:+.6f}")
    
    # TODO 2.5 (调用grad_descent函数，调整超参数，观察实验结果)
    

    # TODO 2.6.3

    
    # 2.7 分类问题
    print("\n===== y3 Multi-class Classification (Softmax + Model Selection) =====")
    y3_train_cls = y3_train.astype(np.int64)
    y3_val_cls = y3_val.astype(np.int64)
    y3_test_cls = y3_test.astype(np.int64)

    best_model, clf_hist, best_info = train_multiclass_softmax_with_model_selection(
        X_train=X_train,
        y_train=y3_train_cls,
        X_val=X_val,
        y_val=y3_val_cls,
        num_classes=4,
        lr=1e-3,
        epochs=1000,
        batch_size=20,
        weight_decay=1e-4,
        verbose_every=100,
        select_by="val_loss",  # 或 "val_acc"
    )

    evaluate_multiclass_softmax(best_model, X_test, y3_test_cls)
    


if __name__ == "__main__":
    main()
