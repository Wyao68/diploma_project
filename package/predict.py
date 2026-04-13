"""
Streamlit 封装：用于对已保存的 coils_model 神经网络做在线推理的简单 UI。
"""

# standard library
import streamlit as st
import numpy as np
import torch
import json
import os
import sys

# my library
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from coils_model import FC_model

st.set_page_config(page_title="PCB coil model at 6.78MHz", layout="centered")
st.title("PCB coil model at 6.78MHz")
st.write("A data-driven model from the geometric parameters of PCB coils to electrical parameters.")

st.write("Please input the geometric parameters of the PCB coil:")
st.write("The copper thickness is fixed at 2 oz (70um).")

n = st.number_input("Turns", min_value=1, max_value=8, value=3, step=1, help="Number of turns")
d = st.number_input("Size (mm)", min_value=1.0, max_value=150.0, value=40.0, step=5.0, help="Outer diameter (mm)")
w = st.number_input("Conductor width (mm)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, help="Conductor width (mm)")
s = st.number_input("Pitch (mm)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, help="Coil pitch (mm)")
angle = st.number_input("Fillet angle (degrees)", min_value=0.0, max_value=90.0, value=45.0, step=5.0, help="Fillet angle (degrees)")

h = 0.070  # 铜厚 mm
rho_cu = 1.68e-8  # 电阻率 Ω·m
mu0 = 4 * np.pi * 1e-7  # H/m
d_out = d  # 线圈外径 mm

d_in = d_out - 2 * (n - 1) * (w + s) - 2 * w    # 内径 mm
d_avg = (d_out + d_in) / 2.0    # 平均直径 mm
rho = (d_out - d_in) / (d_out + d_in)  # 填充比

# 计算直流电感 L_dc (µH)
L_dc = (2.34 * mu0 * (n**2 * (d_avg) / 2) / (1 + 2.73 * rho)) * 1e3   # µH

length_m = (4 * n * d_avg) * 1e-3   # 导线长 m
area_m2 = w * h * 1e-6              # 导线横截面积 m²

# 计算直流电阻 R_dc (Ω)
R_dc = rho_cu * length_m / area_m2   # Ω

# 如果输入参数不合法，提示用户
if d_in <= 0:
    st.error("Invalid input parameters: the inner diameter must be positive. Please adjust the size, turns, pitch, or conductor width.")
    st.stop()

# assemble the input parameters 
input_data = np.array([n, d_out, w, angle, s, L_dc, R_dc], dtype=np.float32)

# load the model and meta
@st.cache_resource
def load_model_and_meta(data_path):
    """加载归一化参数、超参数和模型权重，返回模型、设备、均值和标准差"""
    # 检查必要文件是否存在
    required_files = ['meta.json', 'best_hyperparams.json', 'transfer_model_state_dict.pt']
    for f in required_files:
        if not os.path.exists(os.path.join(data_path, f)):
            st.error(f"Missing required file: {f} in {data_path}")
            st.stop()

    # 加载归一化参数
    with open(os.path.join(data_path, 'meta.json'), "r") as f:
        meta = json.load(f)
    x_mean = np.array(meta['x_mean'], dtype=np.float32)
    x_std = np.array(meta['x_std'], dtype=np.float32)

    # 加载超参数
    with open(os.path.join(data_path, 'best_hyperparams.json'), "r") as f:
        hyperparams = json.load(f)

    # 构建网络结构
    net_dims = [7]
    for i in range(hyperparams['n_hidden']):
        net_dims.append(hyperparams[f'n_units_layer{i}'])
    net_dims.append(2)

    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化模型并加载权重
    model = FC_model.FullyConnectedNet(net_dims, dropout_p=hyperparams['dropout_p']).to(device)
    model.load_state_dict(torch.load(
        os.path.join(data_path, 'transfer_model_state_dict.pt'),
        map_location=device
    ))
    model.eval()  # 设置为评估模式

    return model, device, x_mean, x_std

# 获取项目根目录和模型文件夹路径
base = os.path.dirname(os.path.dirname(__file__)) 
data_path = os.path.join(base, 'saved_models')

# 加载模型及相关参数
model, device, x_mean, x_std = load_model_and_meta(data_path)

# ------------------- 预测 -------------------
if st.button("Predict", help="Predict the inductance, resistance, and Q-factor based on the input geometric parameters."):
    # 归一化
    input_norm = (input_data - x_mean) / x_std
    # 转换为张量并添加批次维度
    x_tensor = torch.from_numpy(input_norm).to(device)

    # 推理
    with torch.no_grad():
        y_pred = model.forward(x_tensor).cpu().numpy()  # shape: (1, 2)

    L_pred = y_pred[0, 0].item()
    R_pred = y_pred[0, 1].item()
    Q = L_pred * 1e-6 * 6.78e6 * 2 * np.pi / R_pred

    st.write(f"**Predicted inductance:** {L_pred:.3f} uH")
    st.write(f"**Predicted resistance:** {R_pred:.3f} Ohm")
    st.write(f"**Predicted Q-factor:** {Q:.3f}")
    
    # print(input_data)
    # print(input_norm)
    # print(y_pred)   
    
# ------------------- 优化 -------------------
if st.button("optimize", help="Find the optimal combination of turns, conductor width, pitch, and fillet angle that maximizes the Q-factor for a fixed outer diameter."):
    # 固定输入的外径d，通过网格搜索的方式寻找最佳的n、w、s、angle组合，使得Q最大，并给出Q值最大的3个组合和对应的Q值。
    d_fixed = d_out
    n_values = range(2, 8)  # 2-7 turns
    w_values = np.arange(0.2, 1.8, 0.01).tolist()  # 导线宽度 mm
    s_values = np.arange(0.2, 1.2, 0.01).tolist()  # 导线间距 mm
    angle_values = np.arange(0, 90, 5).tolist()  # 倒圆角半径 degrees
    
    batch_size = 5000  # 批量推理的大小
    features_opt = []
    combinations = []
    top_combinations = dict()  # 用于存储每个匝数对应的最佳组合
    
    for n in n_values:
        for w in w_values:
            for s in s_values:
                for angle in angle_values:
                    d_in = d_fixed - 2 * (n - 1) * (w + s) - 2 * w
                    if d_in <= 0:
                        continue  # 跳过不合法的组合
                    
                    d_avg = (d_fixed + d_in) / 2.0
                    rho = (d_fixed - d_in) / (d_fixed + d_in)
                    L_dc = (2.34 * mu0 * (n**2 * (d_avg) / 2) / (1 + 2.73 * rho)) * 1e3
                    length_m = (4 * n * d_avg) * 1e-3
                    area_m2 = w * h * 1e-6
                    R_dc = rho_cu * length_m / area_m2
                    
                    input_data_opt = np.array([n, d_fixed, w, angle, s, L_dc, R_dc], dtype=np.float32)
                    input_norm_opt = (input_data_opt - np.asarray(x_mean).flatten()) / np.asarray(x_std).flatten() # 确保 mean 和 std 是一维数组，防止广播错误

                    features_opt.append(input_norm_opt)  # 进行批量推理，避免在循环内频繁调用模型推理导致性能问题
                    combinations.append((n, w, s, angle))
                    
                    # 每当积累到一定数量的组合时，就进行一次批量推理，计算对应的Q值，并清空特征列表以准备下一批组合
                    if len(features_opt) >= batch_size:
                        # 批量推理             
                        with torch.no_grad():
                            x_tensor_opt = torch.from_numpy(np.array(features_opt)).float().to(device)
                            y_pred_opt = model.forward(x_tensor_opt).cpu().numpy() 
    
                        L_pred_opt = y_pred_opt[:, 0] 
                        R_pred_opt = y_pred_opt[:, 1]
                        Q_opt = L_pred_opt * 1e-6 * 6.78e6 * 2 * np.pi / R_pred_opt
                        
                        # 保留该批次Q值最大的组合
                        top_indices = np.argsort(Q_opt)[-20:][::-1] # 取出每个批次中Q值最大的20个组合的索引
                        for idx in top_indices:
                            Q = Q_opt[idx]
                            n_opt, w_opt, s_opt, angle_opt = combinations[idx]
                            # 如果当前匝数的组合还未记录，或者当前组合的Q值更高，则更新该匝数对应的最佳组合
                            if n_opt not in top_combinations:
                                top_combinations[n_opt] = (n_opt, w_opt, s_opt, angle_opt, Q)
                            else:
                                _, _, _, _, Q_current_best = top_combinations[n_opt]
                                if Q > Q_current_best:
                                    top_combinations[n_opt] = (n_opt, w_opt, s_opt, angle_opt, Q)
                                    
                        # 清空特征和组合列表以准备下一批组合
                        features_opt.clear()
                        combinations.clear()
                    
    st.write("**Recommended combinations for maximum Q-factor:**")
    for n_opt, w_opt, s_opt, angle_opt, Q_recommended in top_combinations.values():
        st.write(f"Turns: {n_opt}, Outer diameter: {d_fixed:.2f} mm, Conductor width: {w_opt:.2f} mm, Pitch: {s_opt:.2f} mm, Fillet angle: {angle_opt:.1f} degrees, Q-factor: {Q_recommended:.3f}")

# usage example - streamlit run "C:\Users\86153\Desktop\diploma_project\package\predict.py"