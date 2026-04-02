"""数据预处理工具模块

主要功能：
- 从默认数据路径加载原始数据
- 将前 `input_cols` 列作为输入特征(X)，将后 `output_cols` 列作为输出标签(Y)
- 对输入特征进行标准化（可选）
- 将数据转换为 PyTorch 的 `TensorDataset`，并按比例划分为训练/验证/测试集

示例用法：
    train_ds, val_ds, test_ds, meta = load_data()
返回：
    train_ds, val_ds, test_ds : torch.utils.data.TensorDataset
    meta : dict 包含用于反归一化的统计信息（如 'x_mean', 'x_std'）以及原始数据

"""

import os  
import json

import numpy as np 
import pandas as pd
import torch 
from torch.utils.data import TensorDataset 
from sklearn.ensemble import IsolationForest


def set_random_seed(seed=33):
    """设置所有随机种子，以确保结果可复现"""
    np.random.seed(seed)

    return seed

RANDOM_SEED = set_random_seed()


def _default_data_path() -> str:
    # 返回模块目录下的默认数据路径
    base = os.path.dirname(__file__)
    data_dir = os.path.join(base, 'data')
    return data_dir


def load_data(path: str | None = None,
                input_cols: list[int] = [0,1,2,3,4,7,8],
                output_cols: list[int] = [10,11],
                normalize: bool = True,
                val_ratio: float = 0.1,
                test_ratio: float = 0.1) -> tuple[TensorDataset, TensorDataset, TensorDataset, dict]:
    """加载数据并返回 Train/Val/Test: TensorDataset。

    参数说明：
      - path: xlsx 文件路径，默认使用模块内置数据路径。
      - input_cols: 输入特征列。
      - output_cols: 输出标签值列。
      - normalize: 是否对输入做标准化(均值 0、方差 1)。
      - test_ratio, val_ratio: 测试集与验证集比例(相对于全部数据)。
      - random_seed: 随机种子，便于可复现拆分。

    返回： (train_ds, val_ds, test_ds, meta)
    meta 包含原始数据、标准化统计量等信息的字典，便于后续反归一化或分析。
    """

    # 如果未指定路径，则使用默认数据路径
    if path is None:
        path = _default_data_path()

    # 检查路径是否存在
    if not os.path.exists(path):
        raise FileNotFoundError(f"directory not found: {path}")

    # 读取路径中所有 XLSX 文件路径位置
    if os.path.isdir(path):
        xlsx_files = sorted([os.path.join(path, fn) for fn in os.listdir(path)
                             if fn.lower().endswith('.xlsx')])
        if not xlsx_files:
            raise FileNotFoundError(f"No XLSX files found in directory: {path}")

        parts = []
        for fn in xlsx_files:
            part = pd.read_excel(fn, header=0) # 读取Excel文件，fn为文件路径，header=0表示第一行作为列名
            parts.append(part.to_numpy()) # 将DataFrame转换为NumPy数组并添加到列表中

        # 检查每个 XLSX 文件的列数是否一致
        cols = [p.shape[1] for p in parts]
        if len(set(cols)) != 1:
            raise ValueError(f"XLSX files in directory have inconsistent column counts: {cols}")
        
        data = np.vstack(parts)     # 合成为一个大数组
        meta: dict = {'raw_data': data.tolist()} # 创建meta，使用Python基础数据格式保存
        print(f"Loaded {len(parts)} XLSX files, total shape: {data.shape}")

    # 对数据进行清洗
    # n_estimators - 森林中孤立树的数量
    # max_samples - 每棵树训练的样本数量/比例 'auto' 取 min(256, n_samples)
    # contamination	- 数据集中异常值的比例
    iso = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.12, random_state=RANDOM_SEED)
    outliers = iso.fit_predict(data[:, :]) # 基于输入、输出联合特征进行异常检测，返回1表示正常样本，-1表示异常样本
    data = data[outliers == 1]  
    
    print(f"After outlier removal, total shape: {data.shape}")
    
    # 将数据类型转换为 float32，以匹配 PyTorch 的默认精度
    X = data[:, input_cols].astype(np.float32)
    Y = data[:, output_cols].astype(np.float32)

    # 对输入特征进行归一化
    if normalize:
        x_mean = X.mean(axis=0, keepdims=True)  # keepdims=True 保持维度以便后续广播
        x_std = X.std(axis=0, keepdims=True)
        x_std[x_std == 0.0] = 1.0 # 防止除以零（方差为0说明为一组常数，归一化后全为0）
        X = (X - x_mean) / x_std
        # 将统计量保存到 meta 以供反归一化或在部署时使用
        meta['x_mean'] = x_mean.tolist()
        meta['x_std'] = x_std.tolist()

    # 将 numpy 数组转换为 PyTorch 张量
    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)

    # 构造 dataset
    dataset = TensorDataset(X_t, Y_t)

    # 数据集划分
    n = len(dataset)  
    indices = np.random.permutation(n) 

    # 根据指定的比例计算每个子集的样本数量
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val # 使用减法避免int转换导致的舍入误差，确保总数不变
    
    # 若训练集为空，提示用户调整比例参数
    if n_train <= 0:
        raise ValueError("划分比例导致训练集为空，请调整 test_ratio, val_ratio")

    # 使用切片根据计算出的数量划分索引
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    # 内部辅助函数：根据索引列表从 原dataset 中提取子集并拼接成 新dataset
    def subset_from_indices(ds: TensorDataset, idxs: np.ndarray) -> TensorDataset:
        # 按索引取出样本并用 torch.stack 将它们沿第 0 维重新合并
        xs = torch.stack([ds[i][0] for i in idxs], dim=0)
        ys = torch.stack([ds[i][1] for i in idxs], dim=0)
        return TensorDataset(xs, ys)

    # 构建最终的训练/验证/测试数据集；若验证或测试集为空则返回空的 TensorDataset（shape 为 0）
    train_ds = subset_from_indices(dataset, train_idx)
    val_ds = subset_from_indices(dataset, val_idx) if len(val_idx) > 0 else TensorDataset(torch.empty(0), torch.empty(0))
    test_ds = subset_from_indices(dataset, test_idx) if len(test_idx) > 0 else TensorDataset(torch.empty(0), torch.empty(0))

    return train_ds, val_ds, test_ds, meta


if __name__ == '__main__':
    # 作为脚本执行时，进行数据划分并保存，打印出数据集信息
    try:
        training_data, validation_data, test_data, meta = load_data(input_cols=[0,1,2,3,4,7,8], output_cols=[10,11], normalize=True, val_ratio = 0.15, test_ratio = 0.1)

        torch.save(training_data, 'saved_models/training_data.pt')
        torch.save(validation_data, 'saved_models/validation_data.pt')
        torch.save(test_data, 'saved_models/test_data.pt')
        with open('saved_models/meta.json', 'w') as f: json.dump(meta, f)
        
        # 打印每个子集样本数，帮助确认划分是否正确
        print(f"  train: {len(training_data)} samples")
        print(f"  val:   {len(validation_data)} samples")
        print(f"  test:  {len(test_data)} samples")
        # 打印训练集第一个样本的输入/输出形状便于调试（若训练集为空则显示 None）
        print("Input shape:", training_data[0][0].shape if len(training_data) else None)
        print("Output shape:", training_data[0][1].shape if len(training_data) else None)
        
    except Exception as e:
        print(f"Self-test failed: {e}")
        import traceback
        traceback.print_exc()   # 打印完整堆栈