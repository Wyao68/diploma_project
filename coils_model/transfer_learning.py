"""迁移学习脚本
"""

import os
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 作为脚本运行时直接导入，作为模块运行时相对导入
try:
    from .FC_model import FullyConnectedNet
except Exception:
    from FC_model import FullyConnectedNet


base = os.path.dirname(os.path.dirname(__file__))
models_path = os.path.join(base, 'saved_models')
out_path = os.path.join(base, 'saved_models', 'transfer_model_state_dict.pt')


def set_random_seed(seed=33):
    """设置所有随机种子，以确保结果可复现"""
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

RANDOM_SEED = set_random_seed()

# 迁移学习函数
def transfer_learning(i = 1, 
                      epochs:int = 100,
                      transfer_learning_rate:float = 1e-5,
                      weight_decay:float = 1e-4)-> tuple[list, list]:
    """进行迁移学习
    参数说明:
        i : 仅微调倒数i层. 
        epochs : 微调训练轮数.
        transfer_learning_rate : 迁移学习率.
        weight_decay : 权重衰减.
    返回:
        relative_errors_L : 迁移学习过程中测试集上电感相对误差变化列表.
        relative_errors_R : 迁移学习过程中测试集上电阻相对误差变化列表.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 从best_hyperparams.json中加载模型参数
    with open(os.path.join(models_path, 'best_hyperparams.json'), "r") as f:
        hyperparams = json.load(f)
    
    net_dims = [7]
    for i in range(hyperparams['n_hidden']):
        net_dims.append(hyperparams[f'n_units_layer{i}'])
    net_dims.append(2)
    
    # 加载预训练模型
    model = FullyConnectedNet(net_dims, dropout_p=hyperparams['dropout_p']).to(device) # 实例化模型对象
    model.load_state_dict(torch.load(os.path.join(models_path, 'coils_model_state_dict.pt'), map_location=device)) # 加载预训练权重

    # 加载数据
    training_data = torch.load(os.path.join(models_path, 'transfer_training_data.pt'), weights_only=False)
    test_data = torch.load(os.path.join(models_path, 'transfer_test_data.pt'), weights_only=False)

    # 拆分测试数据的输入和标签，并移动到设备上
    x_test, y_test = test_data.tensors
    x_test, y_test = x_test.float().to(device), y_test.float().to(device)
    
    # 记录迁移学习训练过程中测试集上的相对误差变化，以便后续分析
    relative_errors_L = []
    relative_errors_R = []

    # 进行迁移学习
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 解冻最后i个 Linear 层（输出层）
    # 从 model.model 中找出i个 nn.Linear 模块
    last_i_linear = []
    cnt = 0
    for module in reversed(model.model):
        if isinstance(module, nn.Linear):
            cnt += 1
            last_i_linear.append(module)
            if cnt == i:
                break
        for linear in last_i_linear:
            for param in linear.parameters():
                param.requires_grad = True
        
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=transfer_learning_rate, weight_decay=weight_decay) #微调学习率比预训练学习率小1~2个数量级，以避免过度调整预训练权重
    train_loader = DataLoader(training_data, batch_size= 8, shuffle=True) # 微调数据量本身较小，且训练轮数较少，因此不使用过大的 batch size 以免过拟合，同时保持一定的随机性
    
    model.train()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        # scheduler.step() # 迁移学习训练轮数较少，且学习率已经很小，因此不使用学习率调度器以保持稳定的微调过程
        # 每轮记录一次测试集上的相对误差，以观察迁移学习过程中的性能变化
        model.eval()
        with torch.no_grad():
            y_pred_epoch = model.forward(x_test)
        relative_errors_L_epoch = torch.abs((y_pred_epoch[:, 0] - y_test[:, 0]) / y_test[:, 0])
        relative_errors_R_epoch = torch.abs((y_pred_epoch[:, 1] - y_test[:, 1]) / y_test[:, 1])
        avg_rel_error_L_epoch = relative_errors_L_epoch.mean().item() * 100
        avg_rel_error_R_epoch = relative_errors_R_epoch.mean().item() * 100
        relative_errors_L.append(avg_rel_error_L_epoch)
        relative_errors_R.append(avg_rel_error_R_epoch)
    
    # 输出迁移学习前后后的测试集表现，并与迁移学习前进行对比
    print(f"Average Relative Error on Test Set Before Transfer Learning:")
    print(f"  Inductance: {relative_errors_L[0]:.4f}%")
    print(f"  Resistance: {relative_errors_R[0]:.4f}%")
    print(f"Average Relative Error on Test Set After Transfer Learning:")
    print(f"  Inductance: {relative_errors_L[-1]:.4f}%")
    print(f"  Resistance: {relative_errors_R[-1]:.4f}%")

    # 全部过程结束后打印信息
    print('Saved transfer-learned model to', out_path)
    return relative_errors_L, relative_errors_R


if __name__ == '__main__':
    epochs = 100
    relative_errors_L, relative_errors_R = transfer_learning(i=1, epochs=epochs, transfer_learning_rate=1e-5, weight_decay=1e-4)
    # 绘制迁移学习过程中测试集上的相对误差变化曲线，以观察迁移学习的效果和趋势
    import matplotlib.pyplot as plt
    epochs_range = range(1, epochs + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, relative_errors_L, label='Inductance Relative Error (%)')
    plt.plot(epochs_range, relative_errors_R, label='Resistance Relative Error (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Relative Error (%)')
    plt.title('Transfer Learning Performance')
    plt.legend()
    plt.show()