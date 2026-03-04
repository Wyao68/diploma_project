"""Streamlit 封装：用于对已保存的 coils_model 神经网络做在线推理的简单 UI。

"""

# standard library
import json
import os
from pathlib import Path
import numpy as np
import torch
import streamlit as st
import sys

# 确保项目根目录在 sys.path 中，以便可以直接导入仓库内的包
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    # 插入到 sys.path 开头，优先匹配本地仓库的模块而非已安装同名包
    sys.path.insert(0, str(project_root))

# my library
from coils_model import FC_model

@st.cache_resource #缓存资源密集型对象，避免每次重新运行脚本时都重新加载或计算它们
def load_model(state_path: str, net_dims: list[int]) -> tuple[torch.nn.Module, dict]:
    """加载模型与训练时的数据统计信息。
    
    参数说明：
    输入：
      - state_path: 模型参数文件路径（torch 保存的 state_dict）。
      - net_dims: 网络层维度列表，用于实例化 `FullyConnectedNet`。
    输出：
      - model: 已加载权重并移动到指定设备的 `FullyConnectedNet` 实例。
      - meta: 来自 `data_processor.load_data` 的元数据字典，包含标准化所需的统计量。
      
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FC_model.FullyConnectedNet(net_dims)

    # 确认模型参数文件存在
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Model file not found: {state_path}")

    # 加载模型权重
    model.load_state_dict(torch.load(state_path, map_location=device))

    # 读取模型训练时的统计量 meta
    with open("saved_models\\meta.json", "r") as f: meta = json.load(f)

    return model.to(device), meta


# 标准化用户输入
def normalize_input(x: np.ndarray, meta: dict) -> np.ndarray:
        """标准化用户输入特征向量 x。
        
        参数：
            - x: 输入参数。
            - meta: 训练集的统计量字典，包含 'x_mean' 和 'x_std'。

        返回：标准化后输入参数。
        
        """
        # 没有可用统计量时，不做任何变换，直接返回原始输入
        if meta is None or 'x_mean' not in meta or 'x_std' not in meta:
                return x
            
        # 这里用户输入的是ndarray，所以需要reshape
        x_mean = np.asarray(meta['x_mean']).reshape(-1)
        x_std = np.asarray(meta['x_std']).reshape(-1)
        
        # 防止标准差中出现 0 
        x_std[x_std == 0] = 1.0

        return (x - x_mean) / x_std


# 从输入特征向量预测输出
def predict_from_inputs(model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 转为 float32 的 torch 张量并移动到目标设备(tensor才可以直接to device)
    xt = torch.from_numpy(x.astype(np.float32)).to(device)

    # 单样本时增加 batch 维度以便于神经网络处理
    if xt.dim() == 1:
        xt = xt.unsqueeze(0) # 在第0维（最外层）增加一个维度

    with torch.no_grad():
        out = model(xt)

    # numpy只能在cpu上操作，所以先转回cpu再转numpy，并去掉batch维度
    return out.cpu().numpy().reshape(-1)


def main():
    # 页面标题
    st.title('6.78MHz Coils Model')
    st.markdown('### 6.78MHz下PCB线圈电气参数预测模型')
    
    # 简要说明文字
    st.caption('线圈工作频率固定为 6.78MHz，采用双层并联PCB结构，铜厚为两盎司。')
    st.caption('下方输入 5 个线圈参数，点击 Run model 预测线圈电气参数。')
    
    # 默认模型文件路径
    parent_dir = Path(__file__).resolve().parent.parent
    state_path = os.path.join(parent_dir, 'saved_models', 'coils_model_state_dict.pt')
    
    net_dims = [5, 32, 64, 64, 32, 2] 
    model, meta = load_model(state_path, net_dims)
    
    # 参数名称和单位
    param_names = [
        "线圈匝数",
        "线圈尺寸",
        "线宽",
        "倒圆角角度",
        "匝间距",
        "工作频率"
    ]
    
    param_units = [
        "匝",
        "mm",
        "mm",
        "°",
        "mm",
        "MHz"
    ]
    
    # 输入控件区域
    st.subheader('线圈参数输入')
    
    # 创建两列布局，每列3个参数
    col1, col2 = st.columns(2)
    
    inputs: list[float] = []
    
    for i in range(6):
        # 分配列：前3个参数在col1，后3个在col2
        col = col1 if i < 3 else col2
        
        with col:
            if i == 5:  # 工作频率为固定值
                st.markdown(f"**{param_names[i]}**")
                # 使用st.code或st.markdown显示固定值，使其视觉上类似输入框
                st.markdown(f"6.78 {param_units[i]}")
            elif i == 0:  
                # 线圈匝数为整数
                label = f"{param_names[i]} ({param_units[i]})"
                val = col.number_input(
                    label=label, 
                    step=1, 
                    format="%d",
                    key=f"input_{i}"  # 添加key避免Streamlit重复组件警告
                )
                inputs.append(val)
            else:
                # 可编辑的参数
                label = f"{param_names[i]} ({param_units[i]})"
                val = col.number_input(
                    label=label, 
                    step=0.5, 
                    format="%.2f",
                    key=f"input_{i}"  
                )
                inputs.append(val)
    
    # 使用 pathlib 构建图像路径
    img_path = Path(__file__).resolve().parent / 'graphs' / 'parameters_show.svg'
    st.image(str(img_path), caption='参数示意图', width=700)

    # 运行按钮
    if st.button('🔧 Run model', type="primary", use_container_width=True):
        # 显示加载动画
        with st.spinner('正在计算线圈参数...'):
            # 将输入转换为numpy数组
            x = np.array(inputs, dtype=np.float32)
            
            # 标准化输入
            x_norm = normalize_input(x, meta)
            
            # 调用预测函数
            out = predict_from_inputs(model, x_norm)
            
            # 显示结果
            st.subheader('📈 预测结果')
            
            # 使用列布局显示结果
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                st.metric(
                    label="电感值 L",
                    value=f"{out[0]:.2f}",
                    delta="μH" 
                )
                
            with result_col2:
                st.metric(
                    label="电阻值 R",
                    value=f"{out[1]:.2f}",
                    delta="Ω"
                )
            
            with result_col3:
                st.metric(
                    label="品质因数 Q",
                    value=f"{out[0]*2*3.14159*6.78/out[1]:.2f}",
                    delta="无量纲"
                )
            
            # 添加详细结果表格
            st.markdown("### 详细输出")
            result_data = {
                "参数": ["电感值 (L)", "电阻值 (R)", "品质因数 (Q)"],
                "值": [f"{out[0]:.2f}", f"{out[1]:.2f}", f"{out[0]*2*3.14159*6.78/out[1]:.2f}"],
                "单位": ["μH", "Ω", "无量纲"]
            }
            st.table(result_data)


if __name__ == '__main__':
    main()
