# Standard library
import os
import pickle
import gzip

# Third-party libraries
import numpy as np


def load_data():
    # 首先构造相对于本模块文件的 data 文件路径，确保无论从哪里运行都能定位到数据文件
    module_dir = os.path.dirname(__file__)
    data_path = os.path.join(module_dir, 'data', 'mnist.pkl.gz')

    # 打开 gzip 压缩文件（以二进制模式读取）
    with gzip.open(data_path, 'rb') as f:
        # 使用 latin1 在 Python3 中载入由 Python2 创建的 pickle（保持 numpy arrays 正确）
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    
    # 将训练数据的输入 reshape 为列向量 (784, 1)
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    # 将训练标签转换为 one-hot 向量（用于计算成本与反向传播）
    training_results = [vectorized_result(y) for y in tr_d[1]]
    # 在 Python3 中 zip 返回迭代器，转换为列表以便可多次迭代与索引
    training_data = list(zip(training_inputs, training_results))

    # 验证集的输入同样 reshape 为列向量，但保留标签为整数（用于评估时比较）
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))

    # 测试集处理与验证集相同
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return (training_data, validation_data, test_data)


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
