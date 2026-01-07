import os  

import numpy as np 
import torch 
from torch.utils.data import TensorDataset 


def _default_data_path() -> str:
    # 返回模块目录下的默认数据路径
    base = os.path.dirname(__file__)
    data_dir = os.path.join(base, 'coils_version', 'data')
    return data_dir


if __name__ == "__main__":
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

        data = np.vstack(parts) # 将所有部分沿行方向合并
        meta: dict = {'raw_data': data}
        print(f"Loaded {len(parts)} CSV files, total shape: {data.shape}")

    # 将前 6 列作为输入 X，最后 3 列作为目标 Y，并将数据类型转换为 float32，以匹配 PyTorch 的默认精度并减少内存占用
    X = data[:, :6].astype(np.float32)
    Y = data[:, -3:].astype(np.float32)

    # 对输入特征进行标准化
    # 计算每个特征的均值与标准差，keepdims=True 保持二维形状以便执行标准化时广播
    x_mean = X.mean(axis=0, keepdims=True)
    x_std = X.std(axis=0, keepdims=True)
    # 若某一列方差为 0，则将其设为 1，以避免除零错误，表明该列是常数，标准化后全为 0
    x_std[x_std == 0.0] = 1.0
    # 执行标准化
    X = (X - x_mean) / x_std
    # 将统计量保存到 meta 以供反归一化或在部署时使用
    meta['x_mean'] = x_mean
    meta['x_std'] = x_std

    # 将 numpy 数组转换为 PyTorch 张量 
    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)

    # 构造 dataset
    dataset = TensorDataset(X_t, Y_t)

    # 划分 indices（先 shuffle）
    # 数据集划分：先对索引进行打乱（shuffle），然后根据指定比例切分为 train/val/test
    n = len(dataset)  # 总样本数
    print(meta['raw_data'][1:5])
    print(X[1:5])
    X_orig = X * x_std + x_mean
    print(X_orig[1:5])
