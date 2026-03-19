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
p = st.number_input("Pitch (mm)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, help="Coil pitch (mm)")
angle = st.number_input("Fillet angle (degrees)", min_value=0.0, max_value=90.0, value=45.0, step=5.0, help="Fillet angle (degrees)")

h = 0.070  # mm, 铜导线厚度
d_out = d  # mm, 线圈外径
d_in = d_out - 2 * (n - 1) * (p + w) - 2 * w # mm, 线圈内径
R_dc = 1.68e-8 * (4 * n * (d_out + d_in) / 2) / (w * h) * 1e3  # Ohm, 直流电阻
L_dc = 2.34 * 1.257e-6 * (n**2 * (d_out + d_in) / 2) / (1 + 2.73 * (d_out - d_in) / (d_out + d_in)) * 1e3  # uH, 直流电感

# assemble the input parameters 
input_data = np.array([n, d, w, angle, p, L_dc, R_dc]).astype(np.float32)

# load the model and meta
@st.cache_resource
def load_model_and_meta(data_path):
    """加载归一化参数、超参数和模型权重，返回模型、设备、均值和标准差"""
    # 检查必要文件是否存在
    required_files = ['meta.json', 'best_hyperparams.json', 'coils_model_state_dict.pt']
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
        os.path.join(data_path, 'coils_model_state_dict.pt'),
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
if st.button("Predict"):
    # 归一化
    input_norm = (input_data - x_mean) / x_std
    # 转换为张量并添加批次维度
    x_tensor = torch.from_numpy(input_norm).to(device)

    # 推理
    with torch.no_grad():
        y_pred = model(x_tensor)  # shape: (1, 2)

    L_pred = y_pred[0, 0].item()
    R_pred = y_pred[0, 1].item()
    Q = L_pred * 1e-6 * 6.78e6 * 2 * np.pi / R_pred

    st.write(f"**Predicted inductance:** {L_pred:.2f} uH")
    st.write(f"**Predicted resistance:** {R_pred:.2f} Ohm")
    st.write(f"**Predicted Q-factor:** {Q:.2f}")
    
# usage example - streamlit run "C:\Users\86153\Desktop\diploma_project\streamlit\predict.py"