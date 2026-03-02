"""超参数优化脚本（支持 Optuna 或回退为随机搜索）

主要功能：
  - 在较短的训练周期内搜索最优超参数（学习率、正则化、dropout、batch_size 等），
    以期提高验证集的平均相对误差。

说明：
  - 使用 Optuna 对模型进行超参数优化。
  - 每个 trial 会：实例化模型 -> 在训练子集上训练若干个 epoch -> 在验证集上评估目标指标。
  - 目标函数：验证集的平均相对误差。

运行示例：
  python -m coils_model.hyperopt --trials 50 --epochs 5 --train-size 2000

"""

# Standard library
import argparse
import json
import os
import optuna
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

# My library (包内相对导入，确保作为包运行时能找到模块)
from . import data_processor, FC_model


def set_random_seed(seed=33):
    """设置所有随机种子，以确保不同参数组合在相同随机条件下比较"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

RANDOM_SEED = set_random_seed()


def train_for_one_trial(model: torch.nn.Module,
                        train_ds: torch.utils.data.Dataset,
                        val_ds: torch.utils.data.Dataset,
                        device: torch.device,
                        lr: float = 1e-3,
                        weight_decay: float = 1e-4,
                        batch_size: int = 64,
                        epochs: int = 5,
                        training_data_size: int = 3000,) -> float:
    """
    对单个超参数配置进行短训练并返回验证集的平均相对误差。

    参数说明：
        - model: 待训练的模型实例。
        - train_ds: 训练数据集。
        - val_ds: 验证数据集。
        - device: 训练设备。
        - lr: 学习率。
        - weight_decay: 权重衰减（L2 正则化）。
        - batch_size: 批量大小。
        - epochs: 训练周期数。
        - training_data_size: 用于训练的样本数量（从训练集中截取子集）。
        - dropout_p: 包含在model内，不需要额外传入。
    """

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 数据加载器
    n_train = min(training_data_size, len(train_ds))
    train_subset = Subset(train_ds, list(range(n_train)))
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 简短的训练循环
    model.train()
    for epoch in range(1, epochs + 1):
        for tx, ty in train_loader:
            tx = tx.to(device)
            ty = ty.to(device)
            optimizer.zero_grad()
            preds = model(tx)
            loss = criterion(preds, ty)
            loss.backward()
            optimizer.step()

    # 性能验证
    model.eval()
    val_total = 0
    val_sum = 0.0
    with torch.no_grad():
        for vx, vy in val_loader:
            vx = vx.to(device)
            vy = vy.to(device)
            v_preds = model(vx)
            rel = (v_preds - vy).abs() / vy.abs()
            per_sample = rel.mean(dim=1)  # 每个样本的平均相对误差(电感和电阻相对误差的平均值)
            val_sum += per_sample.sum().item()
            val_total += per_sample.size(0)

        val_avg = val_sum / val_total
        return val_avg


def run_optuna(trials: int,
               train_ds: torch.utils.data.Dataset,
               val_ds: torch.utils.data.Dataset,
               net_dims: list[int],
               device: torch.device,
               epochs: int,
               training_data_size: int) -> dict[str, any]:
    """
    使用 Optuna 进行超参数搜索的封装。
    """
    # 内部优化对象函数
    def objective(trial: 'optuna.trial.Trial') -> float:
        # 为每个 trial 派生一个专用种子，保证在该 trial 内可重复，但不同 trial 间仍然独立
        trial_seed = RANDOM_SEED + trial.number
        set_random_seed(trial_seed)
        # 记录到 trial 的用户属性，便于后续复现最优 trial
        trial.set_user_attr('seed', int(trial_seed))

        # 样本超参数空间定义
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1) # 对数均匀采样
        wd = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
        dropout = trial.suggest_uniform('dropout', 0.0, 0.5) # 均匀采样
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256]) # 离散选择

        model = FC_model.FullyConnectedNet(net_dims, dropout_p=dropout)
        val_avg = train_for_one_trial(model, train_ds, val_ds, device,
                                      lr=lr, weight_decay=wd, dropout_p=dropout,
                                      batch_size=batch_size, epochs=epochs,
                                      training_data_size=training_data_size)
        return float(val_avg)

    # 创建 Optuna study 并开始优化
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))

    study.optimize(objective, n_trials=trials)

    # 使用字典记录最佳结果
    best = {'value': study.best_value, 'params': study.best_trial.params}
    best['seed'] = study.best_trial.user_attrs.get('seed', None)

    return best


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for coils_model')
    parser.add_argument('--trials', type=int, default=20, help='Number of trials')
    parser.add_argument('--epochs', type=int, default=3, help='Epochs per trial (short)')
    parser.add_argument('--train-size', type=int, default=2000, help='Number of training samples to use per trial')
    parser.add_argument('--net-dims', type=str, default='6,32,64,64,32,2', help='Comma-separated net dims')
    parser.add_argument('--no-optuna', action='store_true', help='Force using random search even if optuna is available')
    args = parser.parse_args()

    net_dims = [int(x) for x in args.net_dims.split(',') if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据（按包内 data_processor）
    train_ds, val_ds, test_ds, meta = data_processor.load_data(val_ratio=0.2, test_ratio=0.0, random_seed=RANDOM_SEED)

    if not args.no_optuna:
        try:
            print('Starting Optuna optimization...')
            best = run_optuna(args.trials, train_ds, val_ds, net_dims, device, args.epochs, args.train_size)
            print('Optuna best:', best)
        except Exception as e:
            print('Optuna not available or failed, falling back to random search. Error:', e)
            best = objective_random_search(params_space, args.trials, train_ds, val_ds, net_dims, device, args.epochs, args.train_size)
            print('Random search best:', best)
    else:
        print('Running random search...')
        best = objective_random_search(params_space, args.trials, train_ds, val_ds, net_dims, device, args.epochs, args.train_size)
        print('Random search best:', best)

    # 将最佳结果保存到文件，便于后续加载
    os.makedirs('saved_models', exist_ok=True)
    with open('saved_models/hpo_best.json', 'w') as f:
        json.dump({'best': best}, f, indent=2)

    print('Saved best params to saved_models/hpo_best.json')


if __name__ == '__main__':
    main()
