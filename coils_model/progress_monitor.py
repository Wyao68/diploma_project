"""训练过程可视化模块

主要功能：
- 运行全连接神经网络模型进行训练
- 生成训练和验证过程的数据图表，包括损失、准确率等
- 保存训练过程中的数据以供后续分析
"""

# Standard library
import json

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
                
                
# 绘图函数(包含多个子图)
def plot_training_progress(training_loss,
                            val_L_Max_relevant_errs, 
                            val_L_Avg_relevant_errs, 
                            val_R_Max_relevant_errs, 
                            val_R_Avg_relevant_errs, 
                            validate_loss, 
                            tra_L_Max_relevant_errs, 
                            tra_L_Avg_relevant_errs, 
                            tra_R_Max_relevant_errs, 
                            tra_R_Avg_relevant_errs,
                            L_per_sample_errs, 
                            R_per_sample_errs,
                            x_min=50):
    
    # 绘制训练过程中的各项指标变化曲线
    fig1 = plt.figure(figsize=(12, 9))
    
    ax1 = fig1.add_subplot(231)
    ax1.plot(np.arange(x_min, len(training_loss)), 
        training_loss[x_min:len(training_loss)],
        color="#000000EB")
    ax1.set_xlim([x_min, len(training_loss)])
    ax1.grid(True)
    ax1.set_xlabel('Epoch')
    ax1.set_title('Loss on training data')

    ax2 = fig1.add_subplot(232)
    ax2.plot(np.arange(x_min, len(val_L_Max_relevant_errs)), 
        list(map(lambda x: x*100, val_L_Max_relevant_errs[x_min:len(val_L_Max_relevant_errs)])),
        color="#6F00FFFF")
    ax2.set_xlim([x_min, len(val_L_Max_relevant_errs)])
    ax2.grid(True)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Max Relevant Error (%)')
    ax2.set_title('Max Rel Error for L prediction')

    ax3 = fig1.add_subplot(233)
    ax3.plot(np.arange(x_min, len(val_R_Max_relevant_errs)), 
        list(map(lambda x: x*100, val_R_Max_relevant_errs[x_min:len(val_R_Max_relevant_errs)])),
        color="#FFF200FF")
    ax3.set_xlim([x_min, len(val_R_Max_relevant_errs)])
    ax3.grid(True)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Max Relevant Error (%)')
    ax3.set_title('Max Rel Error for R prediction')

    ax4 = fig1.add_subplot(234)
    ax4.plot(np.arange(x_min, len(val_L_Avg_relevant_errs)), 
        list(map(lambda x: x*100, val_L_Avg_relevant_errs[x_min:len(val_L_Avg_relevant_errs)])),
        color="#2504FFFF")
    ax4.set_xlim([x_min, len(val_L_Avg_relevant_errs)])
    ax4.grid(True)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Average Relevant Error (%)')
    ax4.set_title('Average Rel Error for L prediction')

    ax5 = fig1.add_subplot(235)
    ax5.plot(np.arange(x_min, len(val_R_Avg_relevant_errs)), 
        list(map(lambda x: x*100, val_R_Avg_relevant_errs[x_min:len(val_R_Avg_relevant_errs)])),
        color="#FE0808FF")
    ax5.set_xlim([x_min, len(val_R_Avg_relevant_errs)])
    ax5.grid(True)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Average Relevant Error (%)')
    ax5.set_title('Average Rel Error for R prediction')

    ax6 = fig1.add_subplot(236)
    ax6.plot(np.arange(x_min, len(val_L_Avg_relevant_errs)), 
        list(map(lambda x: x*100, val_L_Avg_relevant_errs[x_min:len(val_L_Avg_relevant_errs)])),
        color="#1EFF00FF",
        label='L on validation data')
    ax6.plot(np.arange(x_min, len(tra_L_Avg_relevant_errs)), 
        list(map(lambda x: x*100, tra_L_Avg_relevant_errs[x_min:len(tra_L_Avg_relevant_errs)])),
        color="#FFAF03",
        label='L on training data')
    ax6.plot(np.arange(x_min, len(val_R_Avg_relevant_errs)), 
        list(map(lambda x: x*100, val_R_Avg_relevant_errs[x_min:len(val_R_Avg_relevant_errs)])),
        color="#1EFF0062",
        label='R on validation data')
    ax6.plot(np.arange(x_min, len(tra_R_Avg_relevant_errs)), 
        list(map(lambda x: x*100, tra_R_Avg_relevant_errs[x_min:len(tra_R_Avg_relevant_errs)])),
        color="#FFAF036D",
        label='R on training data')
    ax6.set_xlim([x_min, len(tra_R_Avg_relevant_errs)])
    ax6.grid(True)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Average Relevant Error (%)')
    ax6.set_title('Average Rel Error on val & tra data')
    ax6.legend(loc='upper right')

    fig2 = plt.figure(figsize=(12, 9))
    
    # 绘制电感的误差竖线图
    ax7 = fig2.add_subplot(211)
    # 使用vlines绘制竖线：x位置，y起点，y终点
    ax7.vlines(np.arange(0, len(L_per_sample_errs)), 
            ymin=0, 
            ymax=list(map(lambda x: x*100, L_per_sample_errs)),
            color="#0A14DA", 
            linewidth=1.5)
    ax7.set_xlim([0, len(L_per_sample_errs)])
    ax7.grid(True, alpha=0.3)
    ax7.set_xlabel('Sample Index')
    ax7.set_ylabel('Relative Error (%)')
    ax7.set_title("Model's Relative Error on Inductance (L) Prediction")
    # 统计信息
    ax7.axhline(y=np.mean(L_per_sample_errs)*100, color='green', linestyle='--', 
            label=f'Mean: {np.mean(L_per_sample_errs)*100:.2f}%')
    ax7.legend()
    
    # 绘制电阻的误差竖线图
    ax8 = fig2.add_subplot(212)
    ax8.vlines(np.arange(0, len(R_per_sample_errs)), 
            ymin=0, 
            ymax=list(map(lambda x: x*100, R_per_sample_errs)),
            color="#DA0A0A", 
            linewidth=1.5)
    ax8.set_xlim([0, len(R_per_sample_errs)])
    ax8.grid(True, alpha=0.3)
    ax8.set_xlabel('Sample Index')
    ax8.set_ylabel('Relative Error (%)')
    ax8.set_title("Model's Relative Error on Resistance (R) Prediction")
    # 统计信息
    ax8.axhline(y=np.mean(R_per_sample_errs)*100, color='green', linestyle='--',
            label=f'Mean: {np.mean(R_per_sample_errs)*100:.2f}%')
    ax8.legend()
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 读取保存的训练/评估结果并生成图表
    with open("saved_models\\training_progress.json", "r") as f:
        training_loss, val_L_Max_relevant_errs, val_L_Avg_relevant_errs, val_R_Max_relevant_errs, val_R_Avg_relevant_errs,\
        validate_loss, tra_L_Max_relevant_errs, tra_L_Avg_relevant_errs, tra_R_Max_relevant_errs, tra_R_Avg_relevant_errs,\
        L_per_sample_errs, R_per_sample_errs = json.load(f)

    plot_training_progress(training_loss, 
                            val_L_Max_relevant_errs, 
                            val_L_Avg_relevant_errs, 
                            val_R_Max_relevant_errs, 
                            val_R_Avg_relevant_errs,
                            validate_loss,
                            tra_L_Max_relevant_errs,
                            tra_L_Avg_relevant_errs,
                            tra_R_Max_relevant_errs,
                            tra_R_Avg_relevant_errs,
                            L_per_sample_errs, 
                            R_per_sample_errs,
                            x_min=0)  
    
    print("Training progress plots generated successfully.")
