# Standard library
import json
import random

# My library
import data_loader
import FC

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np


# 绘图主函数
def main(filename, num_epochs,
         training_cost_xmin, 
         test_accuracy_xmin,
         test_cost_xmin, 
         training_accuracy_xmin,
         training_set_size):

    # 运行网络训练并保存结果，然后根据保存的结果生成图表
    run_network(filename, num_epochs, training_set_size)
    make_plots(filename, num_epochs, 
               training_cost_xmin,
               test_accuracy_xmin,
               test_cost_xmin, 
               training_accuracy_xmin)

                       
def run_network(filename, num_epochs, training_set_size=1000):
    training_data, validation_data, test_data = data_loader.load_data_wrapper() # 加载 MNIST 数据集
    net = FC.FullyConnectedNet([784, 20, 20, 10]) # 建立网络结构
    
    test_cost, test_accuracy, training_cost, training_accuracy \
        = net.running(training_data, validation_data, training_data_size=training_set_size ,epochs=num_epochs, batch_size=500)
    # 将训练/评估结果保存到磁盘（JSON）。使用 with 确保文件正确关闭。
    with open(filename, "w") as f:
        json.dump([test_cost, test_accuracy, training_cost, training_accuracy], f)

def make_plots(filename, num_epochs, 
               training_cost_xmin=0, 
               test_accuracy_xmin=0, 
               test_cost_xmin=0, 
               training_accuracy_xmin=0):
    # 读取保存的训练/评估结果并生成图表
    with open(filename, "r") as f:
        test_cost, test_accuracy, training_cost, training_accuracy = json.load(f)

    # 依次绘制图表
    plot_training_cost(training_cost, num_epochs, training_cost_xmin)
    plot_test_accuracy(test_accuracy, num_epochs, test_accuracy_xmin)
    
    if extra_info.lower() == 'y':
        plot_test_cost(test_cost, num_epochs, test_cost_xmin)
        plot_training_accuracy(training_accuracy, num_epochs, training_accuracy_xmin)
    
    plot_overlay(test_accuracy, training_accuracy, num_epochs, min(test_accuracy_xmin, training_accuracy_xmin))

def plot_training_cost(training_cost, num_epochs, training_cost_xmin):
    # 绘制训练集上的成本随 epoch 变化曲线
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_cost_xmin, num_epochs), 
        training_cost[training_cost_xmin:num_epochs],
        color='#2A6EA6')
    ax.set_xlim([training_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the training data')
    plt.show()

def plot_test_accuracy(test_accuracy, num_epochs, test_accuracy_xmin):
    # 绘制测试集准确率曲线（将百分比转换为小数）
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(test_accuracy_xmin, num_epochs), 
        [accuracy*100 for accuracy in test_accuracy[test_accuracy_xmin:num_epochs]],
        color='#2A6EA6')
    ax.set_xlim([test_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy (%) on the test data')
    plt.show()

def plot_test_cost(test_cost, num_epochs, test_cost_xmin):
    # 绘制测试集上的成本曲线
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(test_cost_xmin, num_epochs), 
        test_cost[test_cost_xmin:num_epochs],
        color='#2A6EA6')
    ax.set_xlim([test_cost_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Cost on the test data')
    plt.show()

def plot_training_accuracy(training_accuracy, num_epochs, training_accuracy_xmin):
    # 绘制训练集的准确率
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_accuracy_xmin, num_epochs), 
        [accuracy*100 for accuracy in training_accuracy[training_accuracy_xmin:num_epochs]],
        color='#2A6EA6')
    ax.set_xlim([training_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy (%) on the training data')
    plt.show()

def plot_overlay(test_accuracy, training_accuracy, num_epochs, xmin):
    # 绘制测试集与训练集准确率的覆盖对比图（便于比较两者的变化趋势）
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(xmin, num_epochs), 
        [accuracy*100 for accuracy in test_accuracy], 
        color='#2A6EA6',
        label="Accuracy on the test data")
    ax.plot(np.arange(xmin, num_epochs), 
        [accuracy*100 for accuracy in training_accuracy], 
        color='#FFA933',
        label="Accuracy on the training data")
    ax.grid(True)
    ax.set_xlim([xmin, num_epochs])
    ax.set_xlabel('Epoch')
    ax.set_ylim([0, 100])
    plt.legend(loc="lower right")
    plt.show()


if __name__ == "__main__":
    # 交互式输入（提示用户）——在 Python 3 中使用 input()
    filename = 'visualization_results'
    num_epochs = int(input("Enter the number of epochs to run for: "))
    training_cost_xmin = int(input("training_cost_xmin (suggest 0): "))
    test_accuracy_xmin = int(input("test_accuracy_xmin (suggest 0): "))
    
    extra_info = input("Do you want to plot test cost and training accuracy? (y/n): ")
    if extra_info.lower() == 'y':
        test_cost_xmin = int(input("test_cost_xmin (suggest 0): "))
        training_accuracy_xmin = int(input("training_accuracy_xmin (suggest 0): "))
    else:
        test_cost_xmin = 0
        training_accuracy_xmin = 0
    
    training_set_size = int(input("training_set_size (suggest 5000): "))
    
    # 调用主函数开始训练并绘图
    main(filename, num_epochs, 
         training_cost_xmin, test_accuracy_xmin, 
         test_cost_xmin, training_accuracy_xmin,
         training_set_size)
