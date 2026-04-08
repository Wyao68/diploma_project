import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base = os.path.dirname(__file__)
data_path = os.path.join(base, 'experiment_data.xlsx')


def main():
    # ---- 读取 data_contrast.xlsx 中的数据 ----
    if os.path.exists(data_path):
        try:
            df = pd.read_excel(data_path, header=None)   # 使用 pandas 读取含有公式单元格的excel时必须在excel中保存公式计算结果，否则公式单元格会被 pandas 读取为 NaN
            data_vol = df.to_numpy()  
        except Exception as e:
            print('Failed to read Excel file:', e)
            return
        
        # 与训练时一致的输入列索引
        emulation_cols = [27,28]    # 28，29列为仿真值的相对误差
        with_characteristic_cols = [30,31]  # 31，32列为有特征工程模型的预测值的相对误差
        without_characteristic_cols = [33,34]  # 34，35列为无特征工程模型的预测值的相对误差

        emulation_relative_err = data_vol[2:, emulation_cols] 
        with_characteristic_relative_err = data_vol[2:, with_characteristic_cols]
        without_characteristic_relative_err = data_vol[2:, without_characteristic_cols]
        
        # 打乱顺序并将误差值分为两列：L和R
        np.random.seed(28)  # 设置随机种子以确保结果可复现
        indices = np.arange(emulation_relative_err.shape[0])
        np.random.shuffle(indices)
        emulation_L_relative_err = emulation_relative_err[indices, 0]
        emulation_R_relative_err = emulation_relative_err[indices, 1]
        with_characteristic_L_relative_err = with_characteristic_relative_err[indices, 0]
        with_characteristic_R_relative_err = with_characteristic_relative_err[indices, 1]
        without_characteristic_L_relative_err = without_characteristic_relative_err[indices, 0]
        without_characteristic_R_relative_err = without_characteristic_relative_err[indices, 1]
        
        avg_err_emulation_L = []
        avg_err_with_characteristic_L = []
        avg_err_without_characteristic_L = []
        avg_err_emulation_R = []
        avg_err_with_characteristic_R = []
        avg_err_without_characteristic_R = []
        
        for i in range(emulation_relative_err.shape[0]):
            avg_err_emulation_L.append(np.mean(emulation_L_relative_err[:i+1]))
            avg_err_with_characteristic_L.append(np.mean(with_characteristic_L_relative_err[:i+1]))
            avg_err_without_characteristic_L.append(np.mean(without_characteristic_L_relative_err[:i+1]))
            avg_err_emulation_R.append(np.mean(emulation_R_relative_err[:i+1]))
            avg_err_with_characteristic_R.append(np.mean(with_characteristic_R_relative_err[:i+1]))
            avg_err_without_characteristic_R.append(np.mean(without_characteristic_R_relative_err[:i+1]))
            
        # ---- 绘制平均误差折线图 ----
        fig = plt.figure(figsize=(10, 6))
        
        # 电感的平均误差折线图
        ax1 = fig.add_subplot(121) 
        ax1.plot(np.arange(len(avg_err_emulation_L)), 
                avg_err_emulation_L,
                label='FEM', color="#09F015")  
        ax1.plot(np.arange(len(avg_err_with_characteristic_L)), 
                avg_err_with_characteristic_L,
                label='With Characteristic Engineering', color="#0A14DA") 
        ax1.plot(np.arange(len(avg_err_without_characteristic_L)), 
                avg_err_without_characteristic_L,
                label='Without Characteristic Engineering', color="#FF0000") 
        
        ax1.set_xlim([10, len(avg_err_emulation_L)])
        ax1.grid(True, alpha=0.3)
        
        # 电阻的平均误差折线图    
        ax2 = fig.add_subplot(122)
        ax2.plot(np.arange(len(avg_err_emulation_R)), 
                avg_err_emulation_R,
                label='FEM', color="#09F015") 
        ax2.plot(np.arange(len(avg_err_with_characteristic_R)), 
                avg_err_with_characteristic_R,
                label='With Characteristic Engineering', color="#0A14DA") 
        ax2.plot(np.arange(len(avg_err_without_characteristic_R)),
                avg_err_without_characteristic_R,
                label='Without Characteristic Engineering', color="#FF0000")
        
        ax2.set_xlim([10, len(avg_err_emulation_R)])
        ax2.grid(True, alpha=0.3)

        ax1.set_xlabel('Number of Samples')
        ax1.set_ylabel('Average Relative Error (%)')  
        ax2.set_xlabel('Number of Samples')
        ax2.set_ylabel('Average Relative Error (%)')
        ax1.set_title('Average Relative Error for Inductance (L) Prediction')
        ax2.set_title('Average Relative Error for Resistance (R) Prediction')
        ax1.legend()
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        # ---- 绘制误差方差的条形图 ----
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        # 电感的误差方差条形图
        axes[0].bar(['FEM', 'Without Characteristic', 'With Characteristic'], 
                    [emulation_L_relative_err.var(), without_characteristic_L_relative_err.var(), with_characteristic_L_relative_err.var()], 
                    color=["#09F015", "#FF0000", "#0A14DA"])
        axes[0].set_title('Variance of Relative Error for Inductance (L)') 
        axes[0].set_xlabel('prediction method')
        axes[0].set_ylabel('Variance(%)')
        axes[0].legend()
        
        # 电阻的误差方差条形图
        axes[1].bar(['FEM', 'Without Characteristic', 'With Characteristic'], 
                    [emulation_R_relative_err.var(), without_characteristic_R_relative_err.var(), with_characteristic_R_relative_err.var()], 
                    color=["#09F015", "#FF0000", "#0A14DA"])
        axes[1].set_title('Variance of Relative Error for Resistance (R)') 
        axes[1].set_xlabel('prediction method')
        axes[1].set_ylabel('Variance(%)')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
        
if __name__ == "__main__":       
    main()
    