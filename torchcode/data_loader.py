# Standard library
import os
import pickle
import gzip

# Third-party libraries
import numpy as np
from torch.utils.data import TensorDataset
import torch


def load_data():
    # 首先构造相对于本模块文件的 data 文件路径，确保无论从哪里运行都能定位到数据文件
    module_dir = os.path.dirname(__file__)
    data_path = os.path.join(module_dir, 'data', 'mnist.pkl.gz')

    # 打开 gzip 压缩文件（以二进制模式读取）
    with gzip.open(data_path, 'rb') as f:
        # 使用 latin1 在 Python3 中载入由 Python2 创建的 pickle
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()

    # 训练集：把每个样本展平为长度 784 的一维向量，标签为整数索引
    X_train = np.array([np.reshape(x, (784,)) for x in tr_d[0]], dtype=np.float32)
    y_train = np.array([int(y) for y in tr_d[1]], dtype=np.int64)

    # 验证集
    X_val = np.array([np.reshape(x, (784,)) for x in va_d[0]], dtype=np.float32)
    y_val = np.array([int(y) for y in va_d[1]], dtype=np.int64)

    # 测试集
    X_test = np.array([np.reshape(x, (784,)) for x in te_d[0]], dtype=np.float32)
    y_test = np.array([int(y) for y in te_d[1]], dtype=np.int64)

    # 转为 PyTorch 张量
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    return train_ds, val_ds, test_ds


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
