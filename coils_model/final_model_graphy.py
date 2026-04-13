import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

import FC_model

base = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(base, 'saved_models')

def build_input_grid(n_list:list[int],
                     d_out:float,
                     w_list:list[float],
                     s:float,
                     angle:float,
                     mean:np.ndarray,
                     std:np.ndarray):
    """
    构建所有 (n, w) 组合的输入特征矩阵，顺序与模型要求一致。
    
    参数:
        n_list: 匝数列表
        d: 外径 (mm)
        w_list: 线宽列表 (mm)
        p: 匝间距 (mm)
        angle: 倒角角度 (度)
        mean, std: 归一化参数 (长度为7的数组)
    
    返回:
        X_norm: 归一化后的特征矩阵 shape (len(n_list)*len(w_list), 7)
        n_grid, w_grid: 用于绘图的网格坐标
    """
    # 确保 mean 和 std 是一维数组，防止X_norm计算时广播错误
    mean = np.asarray(mean).flatten()
    std = np.asarray(std).flatten()
    
    h = 0.070  # 铜厚 mm
    rho_cu = 1.68e-8  # 电阻率 Ω·m
    mu0 = 4 * np.pi * 1e-7  # H/m
    
    rows = []
    n_grid = []
    w_grid = []
    
    for n in n_list:
        for w in w_list:
            # 计算内径 mm
            d_in = d_out - 2 * (n - 1) * (w + s) - 2 * w
            if d_in <= 0:
                raise ValueError(f"内径非正: n={n}, w={w}, d_in={d_in:.2f} mm")
            
            d_avg = (d_out + d_in) / 2.0    # 平均直径 mm
            rho = (d_out - d_in) / (d_out + d_in)  # 填充比
            
            # 计算直流电感 L_dc (µH)
            L_dc = (2.34 * mu0 * (n**2 * (d_avg) / 2) / (1 + 2.73 * rho)) * 1e3   # µH
            
            length_m = (4 * n * d_avg) * 1e-3   # 导线长 m
            area_m2 = w * h * 1e-6              # 导线横截面积 m²

            # 计算直流电阻 R_dc (Ω)
            R_dc = rho_cu * length_m / area_m2   # Ω
            
            # 按正确顺序构建特征向量
            features = np.array([n, d_out, w, angle, s, L_dc, R_dc], dtype=np.float32)
            features_norm = (features - mean) / std
            rows.append(features_norm)
            n_grid.append(n)
            w_grid.append(w)
    
    X = np.array(rows)
    # 重塑网格用于曲面绘图 (注意行优先: n变化慢, w变化快)
    n_unique = len(n_list)
    w_unique = len(w_list)
    n_grid = np.array(n_grid).reshape(n_unique, w_unique)
    w_grid = np.array(w_grid).reshape(n_unique, w_unique)
    return X, n_grid, w_grid

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 从best_hyperparams.json中加载模型参数
    with open(os.path.join(data_path, 'best_hyperparams.json'), "r") as f:
        hyperparams = json.load(f)
    
    net_dims = [7]
    for i in range(hyperparams['n_hidden']):
        net_dims.append(hyperparams[f'n_units_layer{i}'])
    net_dims.append(2)

    # 加载模型
    model = FC_model.FullyConnectedNet(net_dims, dropout_p=hyperparams['dropout_p']).to(device) # 实例化模型对象
    model.load_state_dict(torch.load(os.path.join(data_path, 'transfer_model_state_dict.pt'), map_location=device)) # 加载迁移学习后的模型权重

    with open(os.path.join(data_path, 'meta.json'), "r") as f: 
        meta = json.load(f)
    
    x_mean = np.array(meta['x_mean'], dtype=np.float32)
    x_std = np.array(meta['x_std'], dtype=np.float32)
    
    # 构建输入数据
    # w取0.4~1.2mm，步长0.1mm
    n_vals = np.arange(2, 9).tolist()
    w_vals = np.arange(0.4, 1.8, 0.01).tolist()
    d_out = 50.0
    s = 0.4
    angle = 45.0
    
    X_norm, n_grid, w_grid = build_input_grid(n_list=n_vals, d_out=d_out, w_list=w_vals, s=s, angle=angle, mean=x_mean, std=x_std)
    X_norm = torch.from_numpy(X_norm).float().to(device)
    
    # print(X_norm.shape) # 打印输入矩阵形状进行调试，应为 (len(n_vals)*len(w_vals), 7) = (8*9, 7) = (72, 7)
    # 进行预测
    model.eval()
    with torch.no_grad():
        y_pred = model.forward(X_norm)
    
    L_ac = y_pred[:, 0].cpu().numpy()
    R_ac = y_pred[:, 1].cpu().numpy()
    
    # 计算Q值
    f_mhz = 6.78
    # Q = 2π * (L*1e-6) * (f*1e6) / R = 2π * L * f / R
    Q = 2 * np.pi * L_ac * f_mhz / R_ac

    # 重塑为网格形状 (n_unique, w_unique)
    n_unique = len(n_vals)
    w_unique = len(w_vals)
    Q_grid = Q.reshape(n_unique, w_unique)
    
    # 绘制三维曲面图
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(n_grid, w_grid, Q_grid, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Turns (n)')
    ax.set_ylabel('Conductor Width (mm)')
    ax.set_zlabel('Q Factor')
    ax.set_title('Predicted Q Factor at 6.78 MHz')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    
    # 添加注释，说明外径、匝距和倒圆角角度固定并标明Q值范围和最大Q值
    ax.text2D(0.05, 0.95, f"Outer Diameter: {d_out} mm, Pitch: {s} mm, Angle: {angle}°", transform=ax.transAxes)
    max_Q = np.max(Q_grid)
    max_idx = np.unravel_index(np.argmax(Q_grid), Q_grid.shape)
    ax.text2D(0.05, 0.90, f"Max Q: {max_Q:.2f} at n={n_grid[max_idx]:.0f}, w={w_grid[max_idx]:.2f} mm", transform=ax.transAxes)
    ax.text2D(0.05, 0.85, f"Q Range: {Q.min():.2f} ~ {Q.max():.2f}", transform=ax.transAxes)
    
    plt.show()

    
if __name__ == "__main__":
    main()