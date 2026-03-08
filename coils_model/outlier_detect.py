import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

import FC_model

base = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(base, 'saved_models')


def load_data():
    """加载测试数据集"""
    return torch.load(os.path.join(data_path, 'test_data.pt'), weights_only=False)


def load_model(state_path: str, net_dims: list[int], device: torch.device) -> tuple[torch.nn.Module, dict]:
    """加载模型与训练时的数据统计信息。
    
    参数说明：
    输入：
      - state_path: 模型参数文件路径(torch 保存的 state_dict)。
      - net_dims: 网络层维度列表，用于实例化 `FullyConnectedNet`。
    输出：
      - model: 已加载权重并移动到指定设备的 `FullyConnectedNet` 实例。
      - meta: 元数据字典，包含标准化所需的统计量。
      
    """
    model = FC_model.FullyConnectedNet(net_dims)

    # 加载模型权重
    model.load_state_dict(torch.load(state_path, map_location=device))

    # 读取模型训练时的统计量 meta
    with open(os.path.join(data_path, 'meta.json'), "r") as f: meta = json.load(f)

    return model.to(device), meta

def plot(L_per_sample_errs: np.ndarray, R_per_sample_errs: np.ndarray):
    fig1 = plt.figure(figsize=(12, 9))
    
    # 绘制电感的误差竖线图
    ax1 = fig1.add_subplot(211)
    # 使用vlines绘制竖线：x位置，y起点，y终点
    ax1.vlines(np.arange(0, len(L_per_sample_errs)), 
            ymin=0, 
            ymax=list(map(lambda x: x*100, L_per_sample_errs)),
            color="#0A14DA", 
            linewidth=1.5)
    ax1.set_xlim([0, len(L_per_sample_errs)])
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Relative Error (%)')
    ax1.set_title("Model's Relative Error on Inductance (L) Prediction")
    # 统计信息
    ax1.axhline(y=np.mean(L_per_sample_errs)*100, color='green', linestyle='--', 
            label=f'Mean: {np.mean(L_per_sample_errs)*100:.2f}%')
    ax1.legend()
    
    # 绘制电阻的误差竖线图
    ax2 = fig1.add_subplot(212)
    ax2.vlines(np.arange(0, len(R_per_sample_errs)), 
            ymin=0, 
            ymax=list(map(lambda x: x*100, R_per_sample_errs)),
            color="#DA0A0A", 
            linewidth=1.5)
    ax2.set_xlim([0, len(R_per_sample_errs)])
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Relative Error (%)')
    ax2.set_title("Model's Relative Error on Resistance (R) Prediction")
    # 统计信息
    ax2.axhline(y=np.mean(R_per_sample_errs)*100, color='green', linestyle='--',
            label=f'Mean: {np.mean(R_per_sample_errs)*100:.2f}%')
    ax2.legend()
    
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载模型数据
    model, meta = load_model(os.path.join(data_path, 'coils_model_state_dict.pt'), [8, 67, 199, 10, 2], device=device)
    test_ds = load_data()
    
    # 在测试集上评估模型并计算每个样本的相对误差
    model.eval()
    with torch.no_grad():
        x_test, y_test = test_ds.tensors
        x_test, y_test = x_test.to(device), y_test.to(device)
        y_pred = model.forward(x_test)

        L_true = y_test[:, 0].cpu().numpy()
        R_true = y_test[:, 1].cpu().numpy()
        L_pred = y_pred[:, 0].cpu().numpy()
        R_pred = y_pred[:, 1].cpu().numpy()

        L_per_sample_errs = np.abs(L_pred - L_true) / np.abs(L_true)
        R_per_sample_errs = np.abs(R_pred - R_true) / np.abs(R_true)

    # 将相对误差大于100%的样本标记为异常值，打印样本输入和对应的真实/预测值
    x_ori = x_test.cpu().numpy() * np.array(meta['x_std']) + np.array(meta['x_mean'])
    
    outlier_indices = np.where((L_per_sample_errs > 1.0) | (R_per_sample_errs > 1.0))[0]
    print(f"Detected {len(outlier_indices)} outliers with relative error > 100%:")
    for idx in outlier_indices:
        print(f"Sample Index: {idx}")
        print(f"Input Features: {x_ori[idx]}")
        print(f"True L: {L_true[idx]:.4f}, Predicted L: {L_pred[idx]:.4f}, Relative Error: {L_per_sample_errs[idx]*100:.2f}%")
        print(f"True R: {R_true[idx]:.4f}, Predicted R: {R_pred[idx]:.4f}, Relative Error: {R_per_sample_errs[idx]*100:.2f}%")
        print("-" * 50)

    # 绘制误差竖线图
    plot(L_per_sample_errs, R_per_sample_errs)
    
if __name__ == "__main__":
    main()