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
    

def kernel_svm_subgrad_descent(X, y, alpha=0.1, lambda_reg=0.0001, num_iter=6000, batch_size=16):
    """
    Kernel SVM的随机次梯度下降

    参数：
        X - 特征向量，数组大小 (num_instances, num_features)
        y - 标签向量，数组大小 (num_instances)
        alpha - 浮点数。初始梯度下降步长
        lambda_reg - 正则化系数
        num_iter - 遍历整个训练集的次数（即次数）
        batch_size - 批大小

    返回：
        theta_hist - 参数向量的历史，大小的 2D numpy 数组 (num_iter, num_features)
        loss hist - 正则化损失函数向量的历史，数组大小(num_iter,)
    """
    num_instances, num_features = X.shape[0], X.shape[1]
    theta = np.zeros(num_instances)  # Initialize theta
    theta_hist = np.zeros((num_iter+1, num_instances))  # Initialize theta_hist
    loss_hist = np.zeros((num_iter+1,))  # Initialize loss_hist

    # TODO 3.5.2


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

    # Kernel SVM的随机次梯度下降训练
    # TODO

    # 计算SVM模型在验证集上的准确率，F1-Score以及混淆矩阵
    # TODO



if __name__ == '__main__':
    main()
