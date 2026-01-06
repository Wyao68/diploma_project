import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 加载数据
df = pd.read_csv('N2.csv')
print(f"数据集形状: {df.shape}")
print(df.head())

# 2. 检查数据质量（示例）
print("\n基本统计信息:")
print(df.describe())
print("\n检查缺失值:")
print(df.isnull().sum())

# 假设我们在仿真时，某些极端参数组合可能失败，产生异常值（例如Q值极低）
# 这里假设Q值小于5的数据点为异常值（根据你的数据观察调整阈值）
df_clean = df[df['Q'] > 5].copy()
print(f"\n清理后数据量: {len(df_clean)} / {len(df)}")

# 3. 定义特征 (X) 和 标签 (y)
# 根据开题报告，输入为几何参数和频率
feature_columns = ['N', 'a [mm]', 'd [mm]', 'my_theta [deg]', 'p [mm]', 'Freq [MHz]']
# 输出为电气参数（开题报告中自感L、交流电阻R为关键输出， Q和SRF可由它们推导，但你的数据直接包含Q）
# 我们先构建一个多任务学习的基础：同时预测 L, R, Q
label_columns = ['L [uH]', 'R [¦Ø]', 'Q']  # 注意你的R列名中有特殊字符

X = df_clean[feature_columns].values
y = df_clean[label_columns].values

# 4. 划分数据集 (70%训练, 20%验证, 10%测试)
# 先分出训练+验证集（90%）和测试集（10%）
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
# 再从训练+验证集中分出训练集（70/90 ≈ 78%）和验证集（20/90 ≈ 22%）
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.222, random_state=42, shuffle=True) # 0.222 ≈ 20%/90%

print(f"\n数据划分完成:")
print(f"  训练集: {X_train.shape[0]} 个样本")
print(f"  验证集: {X_val.shape[0]} 个样本")
print(f"  测试集: {X_test.shape[0]} 个样本")

# 5. 特征标准化（非常重要！）
# 使用训练集的统计量来拟合scaler，然后应用到所有数据集
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
# 注意：y_test 暂时不缩放，在最终评估时才用scaler_y.inverse_transform还原回物理量纲

print("\n特征标准化完成。")