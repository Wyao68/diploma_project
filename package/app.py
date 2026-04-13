"""Streamlit app entrypoint for Streamlit Community Cloud."""

# ============================================================
# 必须在所有导入之前执行：强制 PyTorch 使用 CPU、限制线程数
# ============================================================
import os
import sys

# 1. 禁用 CUDA（让 PyTorch 找不到 GPU）
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 2. 限制 PyTorch 线程数（减少内存占用）
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 3. 确保项目路径可导入
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 4. 导入 PyTorch 并强制 CPU 设备（在所有其他导入之前）
import torch
torch.set_num_threads(1)           # 限制线程数
torch.set_default_device('cpu')    # 所有张量默认在 CPU 上创建

# 5. 现在导入你的主 UI 代码
from predict import *

# usage example - streamlit run "C:\Users\86153\Desktop\diploma_project\package\app.py"