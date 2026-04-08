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


def transfer_learning(i = 1, 
                      epochs:int = 100,
                      transfer_learning_rate:float = 1e-5,
                      weight_decay:float = 1e-4):
    """进行迁移学习
    参数说明:
        i : 仅微调倒数i层. 
        epochs : 微调训练轮数.
        transfer_learning_rate : 迁移学习率.
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

    # 计算预训练模型在测试集上的表现，用于后续对比
    x_test, y_test = test_data.tensors
    x_test, y_test = x_test.float().to(device), y_test.float().to(device)
    
    model.eval()
    with torch.no_grad():
        y_pred = model.forward(x_test)
        
    relative_errors_L_before = torch.abs((y_pred[:, 0] - y_test[:, 0]) / y_test[:, 0])
    relative_errors_R_before = torch.abs((y_pred[:, 1] - y_test[:, 1]) / y_test[:, 1])
    avg_rel_error_L_before = relative_errors_L_before.mean().item() * 100
    avg_rel_error_R_before = relative_errors_R_before.mean().item() * 100
    print(f"Average Relative Error on Test Set Before Transfer Learning:")
    print(f"  Inductance: {avg_rel_error_L_before:.4f}%")
    print(f"  Resistance: {avg_rel_error_R_before:.4f}%")

    # 进行迁移学习
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 解冻最后一个 Linear 层（输出层）
    # 从 model.model 中找出i个 nn.Linear 模块
    last_linear = None
    cnt = 0
    for module in reversed(model.model):
        if isinstance(module, nn.Linear):
            cnt += 1
            last_linear = module
            if cnt == i:
                break
        for param in last_linear.parameters():
            param.requires_grad = True
        
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=transfer_learning_rate, weight_decay=weight_decay) #微调学习率比预训练学习率小1~2个数量级，以避免过度调整预训练权重
    train_loader = DataLoader(training_data, batch_size= 8, shuffle=True) # 微调数据量本身较小，且训练轮数较少，因此不使用过大的 batch size 以免过拟合，同时保持一定的随机性
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # 计算迁移学习后的模型在测试集上的表现
    model.eval()
    with torch.no_grad():
        y_pred_after = model.forward(x_test)

    # 分别计算迁移学习后电感和电阻的平均相对误差
    relative_errors_L_after = torch.abs((y_pred_after[:, 0] - y_test[:, 0]) / y_test[:, 0])
    relative_errors_R_after = torch.abs((y_pred_after[:, 1] - y_test[:, 1]) / y_test[:, 1])
    avg_rel_error_L_after = relative_errors_L_after.mean().item() * 100
    avg_rel_error_R_after = relative_errors_R_after.mean().item() * 100
    print(f"Average Relative Error on Test Set After Transfer Learning:")
    print(f"  Inductance: {avg_rel_error_L_after:.4f}%")
    print(f"  Resistance: {avg_rel_error_R_after:.4f}%")

    # 保存微调后的权重
    torch.save(model.state_dict(), out_path)
    print('Saved transfer-learned model to', out_path)


if __name__ == '__main__':
    transfer_learning(i=1, epochs=100, transfer_learning_rate=1e-5, weight_decay=1e-4)
