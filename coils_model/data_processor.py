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

import numpy as np 
import torch 
from torch.utils.data import TensorDataset 


def _default_data_path() -> str:
    # 返回模块目录下的默认数据路径
    base = os.path.dirname(__file__)
    data_dir = os.path.join(base, 'data')
    return data_dir


def load_data(path: str | None = None,
                input_cols: int = 6,
                output_cols: int = 3,
                normalize: bool = True,
                val_ratio: float = 0.2,
                test_ratio: float = 0.0,
                random_seed: int = 42) -> tuple[TensorDataset, TensorDataset, TensorDataset, dict]:
    """加载数据并返回 Train/Val/Test TensorDataset。

    参数说明：
      - path: CSV 文件路径，默认使用模块内置数据路径。
      - input_cols: 输入特征列数(从左侧开始)，默认 6。
      - output_cols: 输出列数(从右侧开始，且不包含最后一列)，默认 3。
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

    # 读取路径中所有 CSV 并合并
    if os.path.isdir(path):
        csv_files = sorted([os.path.join(path, fn) for fn in os.listdir(path)
                            if fn.lower().endswith('.csv')])
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in directory: {path}")

        parts = []
        for fn in csv_files:
            part = np.genfromtxt(fn, delimiter=',', dtype=float, skip_header=1) # 跳过表头
            parts.append(part)

        # 检查所有部分的列数是否一致
        cols = [p.shape[1] for p in parts]
        if len(set(cols)) != 1:
            raise ValueError(f"CSV files in directory have inconsistent column counts: {cols}")

        data = np.vstack(parts) 
        meta: dict = {'raw_data': data.tolist()} # 创建meta，使用Python基础数据格式保存原始数据以备后用
        print(f"Loaded {len(parts)} CSV files, total shape: {data.shape}")

    # 将数据类型转换为 float32，以匹配 PyTorch 的默认精度
    X = data[:, :input_cols].astype(np.float32)
    Y = data[:, -output_cols:-1].astype(np.float32)

    # 对输入特征进行归一化
    if normalize:
        x_mean = X.mean(axis=0, keepdims=True)
        x_std = X.std(axis=0, keepdims=True)
        x_std[x_std == 0.0] = 1.0 # 防止除以零
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
    rng = np.random.RandomState(random_seed)  # 使用 numpy 的 RandomState（随机数生成器） 以保证可复现
    indices = np.arange(n)  
    rng.shuffle(indices) 

    # 根据指定的比例计算每个子集的样本数量
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_test - n_val
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
    # 当作为脚本执行时，运行一个简单的自检：尝试加载文件并打印出数据集信息
    try:
        train_ds, val_ds, test_ds, meta = load_data()
        # 打印每个子集样本数，帮助确认划分是否正确
        print(f"  train: {len(train_ds)} samples")
        print(f"  val:   {len(val_ds)} samples")
        print(f"  test:  {len(test_ds)} samples")
        # 打印训练集第一个样本的输入/输出形状便于调试（若训练集为空则显示 None）
        print("Input shape:", train_ds[0][0].shape if len(train_ds) else None)
        print("Output shape:", train_ds[0][1].shape if len(train_ds) else None)
    except Exception as e:
        # 捕获并打印任何异常
        print("Self-test failed:", e)