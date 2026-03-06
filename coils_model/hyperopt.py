"""使用 Optuna 对全连接网络做超参数优化的最小脚本

功能：
- 优化隐藏层数量(num_layers)
- 优化每层神经元数（每层单独采样）
- 优化学习率(learning_rate)

使用说明（在 Windows PowerShell 中）：
    python coils_model\hyperopt.py --trials 30 --epochs 20

注意：该脚本尽量复用 `coils_model.FC_model.FullyConnectedNet` 与
`coils_model.data_processor.load_data`，但训练循环为简化版，便于在 Optuna 中快速评估。
"""

import os
import argparse # 命令行参数解析（接口）
import optuna

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 作为包运行时相对导入，直接运行时绝对导入
try:
    from .FC_model import FullyConnectedNet
    from .data_processor import load_data
except Exception:
    from FC_model import FullyConnectedNet
    from data_processor import load_data


def build_net_dims(input_dim: int, output_dim: int, hidden_units: list[int]) -> list[int]:
    """根据输入维度、输出维度和隐藏层单元列表构建 net_dims 列表:
        [input_dim, hidden1, hidden2, ..., output_dim]
    """
    return [input_dim] + hidden_units + [output_dim]


def objective(trial: optuna.trial.Trial, epochs: int = 20, batch_size: int = 64, training_data_size: int | None = None) -> float:
    """Optuna 的目标函数：

    - 从数据加载器获取训练/验证集
    - 根据 trial 采样网络结构与学习率
    - 用一个精简训练循环训练若干 epoch，并返回最后一个 epoch 的验证集平均损失(MSE)
    - 使用 AdamW 优化器与 MSELoss
    - 在每个 epoch 后向 Optuna 报告中间值并支持剪枝
    
    采样的超参数包括：
    - 隐藏层数量
    - 每层单独采样神经元数量
    - adamW学习率
    - 权重衰减系数
    """

    # 加载数据
    base = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base, 'saved_models')

    train_ds = torch.load(os.path.join(data_path, 'training_data.pt'), weights_only=False)
    val_ds = torch.load(os.path.join(data_path, 'validation_data.pt'), weights_only=False)

    # 确保验证集存在
    if len(val_ds) == 0:
        raise RuntimeError("验证集为空，请调整 data_processor.load_data 的划分参数以包含验证集")

    # 从 dataset 推断输入与输出维度
    input_dim = train_ds[0][0].shape[0]
    output_dim = train_ds[0][1].shape[0]

    # 采样超参数
    # 隐藏层数量，1-4层
    n_hidden = trial.suggest_int("n_hidden", 1, 4)
    hidden_units = []
    for i in range(n_hidden):
        # 每层神经元数量，8-256个，使用对数尺度均匀采样
        units = trial.suggest_int(f"n_units_layer{i}", 8, 256, log=True)
        hidden_units.append(units)

    # 学习率，1e-5 到 1e-1，使用对数尺度均匀采样
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    # 权重衰减系数，1e-6 到 1e-2，使用对数尺度均匀采样
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    net_dims = build_net_dims(input_dim, output_dim, hidden_units)

    # 构建模型并移动到设备（优先 GPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullyConnectedNet(net_dims)
    model.to(device)

    # 加载数据到 DataLoader，支持可选的训练数据子集大小用于快速测试
    if training_data_size is not None:
        n_train = min(training_data_size, len(train_ds))
        train_subset = torch.utils.data.Subset(train_ds, list(range(n_train)))
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 损失、优化器、学习率调度器
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) 
    scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',     
                factor=0.5,       
                patience=5, 
                threshold=1e-4,
                threshold_mode='rel'           
                )
    
    # 训练循环
    for epoch in range(1, epochs + 1):
        model.train()
        for tx, ty in train_loader:
            tx = tx.to(device)
            ty = ty.to(device)
            optimizer.zero_grad()
            
            preds = model.model(tx)
            loss = criterion(preds, ty)
            loss.backward()
            optimizer.step()

        # 评估验证集损失
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(device)
                vy = vy.to(device)
                preds = model.model(vx)
                l = criterion(preds, vy)
                val_loss += l.item() * vx.size(0)
                n_val += vx.size(0)

        val_loss = val_loss / n_val
        
        scheduler.step(val_loss) 

        # 每个epoch结束后向 Optuna 报告中间结果并判断剪枝
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned() # 如果剪枝，抛出异常，并让 Optuna 记录该 trial

    # 返回最终 epoch 的验证损失作为优化目标
    return float(val_loss)


def run_study(n_trials: int = 20, epochs: int = 20, batch_size: int = 64, training_data_size: int | None = None):
    """
    运行 Optuna study 并打印最优结果。
    """
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(direction="minimize", sampler=sampler)

    try:
        study.optimize(lambda t: objective(t, epochs=epochs, batch_size=batch_size, training_data_size=training_data_size), n_trials=n_trials)
    except KeyboardInterrupt:
        print("用户中断，停止优化。")

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for k, v in trial.params.items():
        print(f"    {k}: {v}")


def set_random_seed(seed=33):
    """设置所有随机种子，以确保结果可复现"""
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


if __name__ == '__main__':
    RANDOM_SEED = set_random_seed()
    
    parser = argparse.ArgumentParser(description="使用 Optuna 对全连接网络做超参数搜索（最小示例）")
    parser.add_argument("--trials", type=int, default=200, help="Optuna trials 数量")
    parser.add_argument("--epochs", type=int, default=150, help="每个 trial 的训练 epoch 数")
    parser.add_argument("--batch_size", type=int, default=64, help="训练批大小")
    parser.add_argument("--training_data_size", type=int, default=None, help="用于训练的样本数量(None 表示全部）")
    args = parser.parse_args()

    run_study(n_trials=args.trials, epochs=args.epochs, batch_size=args.batch_size, training_data_size=args.training_data_size)
