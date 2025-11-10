# Standard library
import json
import random
import sys

# My library
import mnist_loader
import network

# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np


# min用于设置图标的起始点
def main(filename, num_epochs,
         training_cost_xmin=200, 
         test_accuracy_xmin=200,
         test_cost_xmin=0, 
         training_accuracy_xmin=0,  
         training_set_size=1000, 
         lmbda=0.0, mu=0.0):
    """``filename`` is the name of the file where the results will be
    stored.  ``num_epochs`` is the number of epochs to train for.
    ``training_set_size`` is the number of images to train on.
    ``lmbda`` is the regularization parameter.  The other parameters
    set the epochs at which to start plotting on the x axis.
    """
    # 运行网络训练并保存结果，然后根据保存的结果生成图表
    run_network(filename, num_epochs, training_set_size, lmbda, mu)
    make_plots(filename, num_epochs, 
               training_cost_xmin,
               test_accuracy_xmin,
               test_cost_xmin, 
               training_accuracy_xmin,
               training_set_size)
                       
def run_network(filename, num_epochs, training_set_size=1000, lmbda=0.0, mu=0.0):
    """Train the network for ``num_epochs`` on ``training_set_size``
    images, and store the results in ``filename``.  Those results can
    later be used by ``make_plots``.  Note that the results are stored
    to disk in large part because it's convenient not to have to
    ``run_network`` each time we want to make a plot (it's slow).
    """
    # Make results more easily reproducible
    random.seed(12345678)
    np.random.seed(12345678) 
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper() # 加载 MNIST 数据集
    net = network.Network([784, 30, 10], network.sigmoid(), network.CrossEntropyCost()) # 建立网络结构
    # 随机梯度下降小批量为10，学习率为0.5
    test_cost, test_accuracy, training_cost, training_accuracy \
        = net.SGD(training_data[:training_set_size], num_epochs, mini_batch_size=10, eta=0.5,
                  lmbda = lmbda, mu=mu, 
                  evaluation_data=test_data, 
                  monitor_evaluation_cost=True, 
                  monitor_evaluation_accuracy=True, 
                  monitor_training_cost=True, 
                  monitor_training_accuracy=True)
    # 将训练/评估结果保存到磁盘（JSON）。使用 with 确保文件正确关闭。
    with open(filename, "w") as f:
        json.dump([test_cost, test_accuracy, training_cost, training_accuracy], f)

def make_plots(filename, num_epochs, 
               training_cost_xmin=200, 
               test_accuracy_xmin=200, 
               test_cost_xmin=0, 
               training_accuracy_xmin=0,
               training_set_size=1000):
    """Load the results from ``filename``, and generate the corresponding
    plots. """
    # 读取保存的训练/评估结果并生成图表
    with open(filename, "r") as f:
        test_cost, test_accuracy, training_cost, training_accuracy = json.load(f)

    # 依次绘制四个单独的图表以及一个覆盖对比图
    plot_training_cost(training_cost, num_epochs, training_cost_xmin)
    plot_test_accuracy(test_accuracy, num_epochs, test_accuracy_xmin)
    plot_test_cost(test_cost, num_epochs, test_cost_xmin)
    plot_training_accuracy(training_accuracy, num_epochs, 
                           training_accuracy_xmin, training_set_size)
    plot_overlay(test_accuracy, training_accuracy, num_epochs,
                 min(test_accuracy_xmin, training_accuracy_xmin),
                 training_set_size)

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
        [accuracy/100.0 
         for accuracy in test_accuracy[test_accuracy_xmin:num_epochs]],
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

def plot_training_accuracy(training_accuracy, num_epochs, 
                           training_accuracy_xmin, training_set_size):
    # 绘制训练集的准确率（按训练集大小归一化并以百分比表示）
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_accuracy_xmin, num_epochs), 
        [accuracy*100.0/training_set_size 
         for accuracy in training_accuracy[training_accuracy_xmin:num_epochs]],
        color='#2A6EA6')
    ax.set_xlim([training_accuracy_xmin, num_epochs])
    ax.grid(True)
    ax.set_xlabel('Epoch')
    ax.set_title('Accuracy (%) on the training data')
    plt.show()

def plot_overlay(test_accuracy, training_accuracy, num_epochs, xmin,
                 training_set_size):
    # 绘制测试集与训练集准确率的覆盖对比图（便于比较两者的变化趋势）
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(xmin, num_epochs), 
        [accuracy/100.0 for accuracy in test_accuracy], 
        color='#2A6EA6',
        label="Accuracy on the test data")
    ax.plot(np.arange(xmin, num_epochs), 
        [accuracy*100.0/training_set_size 
         for accuracy in training_accuracy], 
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
    filename = input("Enter a file name: ")
    num_epochs = int(input("Enter the number of epochs to run for: "))
    training_cost_xmin = int(input("training_cost_xmin (suggest 0): "))
    test_accuracy_xmin = int(input("test_accuracy_xmin (suggest 0): "))
    test_cost_xmin = int(input("test_cost_xmin (suggest 0): "))
    training_accuracy_xmin = int(input("training_accuracy_xmin (suggest 0): "))
    training_set_size = int(input("Training set size (suggest 1000): "))
    lmbda = float(input("Enter the regularization parameter, lambda (suggest: 0.1): "))
    mu = float(input("Enter the momentum parameter, mu (suggest: 0.5): "))

    # 调用主函数开始训练并绘图
    main(filename, num_epochs, training_cost_xmin, 
         test_accuracy_xmin, test_cost_xmin, training_accuracy_xmin,
         training_set_size, lmbda, mu)
