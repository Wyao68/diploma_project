"""训练过程可视化模块

主要功能：
- 运行全连接神经网络模型进行训练
- 生成训练和验证过程的数据图表，包括损失、准确率等
- 保存训练过程中的数据以供后续分析
"""

# Standard library
import json

# My library
import data_processor
import FC_model

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
                
                
# 绘图函数(包含多个子图)
def plot_training_progress(training_loss, 
                           val_Max_relevant_err, 
                           val_Avg_relevant_err, 
                           tra_Max_relevant_err, 
                           tra_Avg_relevant_err, 
                           num_epochs,
                           x_min=50):
    
    fig = plt.figure(figsize=(12, 8))
    
    ax1 = fig.add_subplot(231)
    ax1.plot(np.arange(x_min, num_epochs), 
        training_loss[x_min:num_epochs],
        color="#0004E8D2")
    ax1.set_xlim([x_min, num_epochs])
    ax1.grid(True)
    ax1.set_xlabel('Epoch')
    ax1.set_title('Loss on the training data')

    ax2 = fig.add_subplot(232)
    ax2.plot(np.arange(x_min, num_epochs), 
        list(map(lambda x: x*100, val_Max_relevant_err[x_min:num_epochs])),
        color="#DA0A0A")
    ax2.set_xlim([x_min, num_epochs])
    ax2.grid(True)
    ax2.set_xlabel('Epoch')
    ax2.set_title('Max Relevant Error on the validation data')

    ax3 = fig.add_subplot(233)
    ax3.plot(np.arange(x_min, num_epochs), 
        list(map(lambda x: x*100, val_Avg_relevant_err[x_min:num_epochs])),
        color="#32FC40")
    ax3.set_xlim([x_min, num_epochs])
    ax3.grid(True)
    ax3.set_xlabel('Epoch')
    ax3.set_title('Average Relevant Error on the validation data')

    ax4 = fig.add_subplot(234)
    ax4.plot(np.arange(x_min, num_epochs), 
        list(map(lambda x: x*100, val_Max_relevant_err[x_min:num_epochs])),
        color="#FF2A2A",
        label='Max Relevant Error')
    ax4.plot(np.arange(x_min, num_epochs), 
        list(map(lambda x: x*100, val_Avg_relevant_err[x_min:num_epochs])),
        color="#3010E6",
        label='Average Relevant Error')
    ax4.set_xlim([x_min, num_epochs])
    ax4.grid(True)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Relevant Error (%)')
    ax4.set_title('Max & Avg Relevant Error on validation data')
    ax4.legend(loc='upper right')

    ax5 = fig.add_subplot(235)
    ax5.plot(np.arange(x_min, num_epochs), 
        list(map(lambda x: x*100, val_Max_relevant_err[x_min:num_epochs])),
        color="#FF2A2A",
        label='validation data')
    ax5.plot(np.arange(x_min, num_epochs), 
        list(map(lambda x: x*100, tra_Max_relevant_err[x_min:num_epochs])),
        color="#3010E6",
        label='training data')
    ax5.set_xlim([x_min, num_epochs])
    ax5.grid(True)
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Max Relevant Error (%)')
    ax5.set_title('Max Relevant Error on val & tra data')
    ax5.legend(loc='upper right')   

    ax6 = fig.add_subplot(236)
    ax6.plot(np.arange(x_min, num_epochs), 
        list(map(lambda x: x*100, val_Avg_relevant_err[x_min:num_epochs])),
        color="#FF2A2A",
        label='validation data')
    ax6.plot(np.arange(x_min, num_epochs), 
        list(map(lambda x: x*100, tra_Avg_relevant_err[x_min:num_epochs])),
        color="#3010E6",
        label='training data')
    ax6.set_xlim([x_min, num_epochs])
    ax6.grid(True)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Average Relevant Error (%)')
    ax6.set_title('Average Relevant Error on val & tra data')
    ax6.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    num_epochs = 100 # 训练轮数
    
    training_data, validation_data, test_data, meta = data_processor.load_data()
    net = FC_model.FullyConnectedNet([6, 32, 64, 64, 32, 2]) 
    
    training_loss, val_Max_relevant_err, val_Avg_relevant_err, validate_loss, tra_Max_relevant_err, tra_Avg_relevant_err \
        = net.running(training_data, validation_data, training_data_size=4000 ,epochs=num_epochs, batch_size=64)

    with open("saved_models\\training_progress.json", "w") as f:
        json.dump([training_loss, val_Max_relevant_err, val_Avg_relevant_err, validate_loss, tra_Max_relevant_err, tra_Avg_relevant_err], f)
    
    # 读取保存的训练/评估结果并生成图表
    with open("saved_models\\training_progress.json", "r") as f:
        training_loss, val_Max_relevant_err, val_Avg_relevant_err, validate_loss, tra_Max_relevant_err, tra_Avg_relevant_err = json.load(f)

    plot_training_progress(training_loss, 
                           val_Max_relevant_err, 
                           val_Avg_relevant_err, 
                           tra_Max_relevant_err, 
                           tra_Avg_relevant_err, 
                           num_epochs,
                           x_min=50)  # 从第50轮开始绘图
    print("Training progress plots generated successfully.")
