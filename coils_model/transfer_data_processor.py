"""加载迁移学习数据并进行预处理的模块
"""

import os  
import json

import numpy as np 
import pandas as pd
import torch 
from torch.utils.data import TensorDataset 


def set_random_seed(seed=33):
    """设置所有随机种子，以确保结果可复现"""
    np.random.seed(seed)

    return seed

RANDOM_SEED = set_random_seed()


base = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(base, 'coils_model', 'transfer_data', 'experiment_data.xlsx')
meta_path = os.path.join(base, 'saved_models', 'meta.json')


def load_data(  input_cols: list[int] = [0,1,2,3,4,7,8],
                output_cols: list[int] = [10,11],
                test_ratio: float = 0.2) -> tuple[TensorDataset, TensorDataset]:
    """
    参数说明：
      - input_cols: 输入特征列。
      - output_cols: 输出标签值列。
      - test_ratio: 测试集比例(相对于全部数据)。

    返回： (train_ds, test_ds)
    """
    # 读取预训练数据的meta，用于数据预处理
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    # 读取迁移学习数据
    data = pd.read_excel(data_dir, header=0).to_numpy() 
    
    # 将数据类型转换为 float32，以匹配 PyTorch 的默认精度
    X = data[:, input_cols].astype(np.float32)
    Y = data[:, output_cols].astype(np.float32)

    # 对输入特征进行标准化，使用预训练数据的统计量（均值和标准差）
    x_mean = meta['x_mean']
    x_std = meta['x_std']
    X = (X - np.array(x_mean)) / np.array(x_std)

    # 将 numpy 数组转换为 PyTorch 张量
    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)

    # 构造 dataset
    dataset = TensorDataset(X_t, Y_t)

    # 数据集划分
    n = len(dataset)  
    indices = np.random.permutation(n) # 打乱索引

    # 根据指定的比例计算每个子集的样本数量
    n_test = int(n * test_ratio)
    n_train = n - n_test 
    
    # 若训练集为空，提示用户调整比例参数
    if n_train <= 0:
        raise ValueError("划分比例导致训练集为空，请调整 test_ratio")

    # 使用切片根据计算出的数量划分索引
    train_idx = indices[:n_train]
    test_idx = indices[n_train:n_train + n_test]

    # 内部辅助函数：根据索引列表从 原dataset 中提取子集并拼接成 新dataset
    def subset_from_indices(ds: TensorDataset, idxs: np.ndarray) -> TensorDataset:
        # 按索引取出样本并用 torch.stack 将它们沿第 0 维重新合并
        xs = torch.stack([ds[i][0] for i in idxs], dim=0)
        ys = torch.stack([ds[i][1] for i in idxs], dim=0)
        return TensorDataset(xs, ys)

    # 构建最终的训练/验证/测试数据集；若验证或测试集为空则返回空的 TensorDataset（shape 为 0）
    train_ds = subset_from_indices(dataset, train_idx)
    test_ds = subset_from_indices(dataset, test_idx) if len(test_idx) > 0 else TensorDataset(torch.empty(0), torch.empty(0))

    return train_ds, test_ds


if __name__ == '__main__':
    # 作为脚本执行时，进行数据划分并保存，打印出数据集信息
    try:
        training_data, test_data = load_data(input_cols=[0,1,2,3,4,7,8], output_cols=[10,11], test_ratio = 0.2)

        torch.save(training_data, 'saved_models/transfer_training_data.pt')
        torch.save(test_data, 'saved_models/transfer_test_data.pt')
        
        # 打印每个子集样本数，帮助确认划分是否正确
        print(f"  train: {len(training_data)} samples")
        print(f"  test:  {len(test_data)} samples")
        # 打印训练集第一个样本的输入/输出形状便于调试（若训练集为空则显示 None）
        print("Input shape:", training_data[0][0].shape if len(training_data) else None)
        print("Output shape:", training_data[0][1].shape if len(training_data) else None)
        
    except Exception as e:
        print(f"Self-test failed: {e}")
        import traceback
        traceback.print_exc()   # 打印完整堆栈