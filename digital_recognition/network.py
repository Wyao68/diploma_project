import json
import random
import sys

import numpy as np


# 定义激活函数
class sigmoid:
    @staticmethod
    def forward(x):
        return 1.0 / (1 + np.exp(-x))
    
    @staticmethod
    def derivative(y):
        return sigmoid.forward(y) * (1 - sigmoid.forward(y))
    

# 定义损失函数
class QuadraticCost():
    @staticmethod
    def forward(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def derivative(a, y):
        return (a - y)


# np.nan_to_num() 是NumPy中用于处理NaN和无穷大数值的函数，它将这些特殊值替换为有限的数值
class CrossEntropyCost():
    @staticmethod
    def forward(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def derivative(a, y):
        return (a - y) / (a * (1 - a))


# 将0~9之间的标量进行one-hot编码
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


# 建立网络并初始化权重和偏执(后续引入更好的方法来初始化)
# np.random.randn()生成标准正态分布（高斯分布）
class Network():
    def __init__(self, sizes, activator, cost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.activator = activator
        self.cost = cost
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # 第一层为输入没有偏置项
        self.weights = [np.random.randn(y, x)/np.sqrt(x)         
                        for x, y in zip(sizes[:-1], sizes[1:])]  # 压缩标准差，避免隐藏层神经元饱和
        self.velocity = [np.zeros_like(w) for w in self.weights]  # 初始化动量项为0
        
        
    def feedforward(self, a):
        '''前向传播'''
        for b, w in zip(self.biases, self.weights):
            a = self.activator.forward(np.dot(w, a) + b)
        return a
    
    
    # 同时使用正则化和动量法
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0, mu = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False, 
            monitor_training_cost=False,
            monitor_training_accuracy=False,):
        '''随机梯度下降
        training_data是由训练输入和目标输出组成的元组(x, y)组成的列表
        eta是学习率
        lmbda是正则化参数(正则化参数大小与训练集大小有关)
        mu是动量参数'''
        if evaluation_data: n_eva = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            np.random.shuffle(training_data) # 打乱原有的训练数据顺序
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)] # 划分小批量
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, mu, n) # 更新权重和偏置项
            print("Epoch {} complete".format(j))
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(accuracy, n_eva))
            print() # 空行分隔不同epoch的输出
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy # 返回四个监控列表用于绘图可视化训练过程
                

    # 更新小批量的权重和偏置项
    def update_mini_batch(self, mini_batch, eta, lmbda, mu, n):
        '''对一个随机小批量中训练数据更新网络的权重和偏置项'''
        m = len(mini_batch)
        nabla_b = [np.zeros_like(b) for b in self.biases]   # 初始化偏置项梯度为0
        nabla_w = [np.zeros_like(w) for w in self.weights]  # 初始化权重梯度为0
        # 求出这一组小批量总梯度
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] 
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # 对梯度求平均值
        nabla_b = [nb / m for nb in nabla_b]
        nabla_w = [nw / m for nw in nabla_w]
        # 计算包含正则项的总梯度
        nabla_w_total = [nw + (lmbda / n) * w for nw, w in zip(nabla_w, self.weights)]
        # 更新参数    
        self.biases = [b - eta * nb for b, nb in zip(self.biases, nabla_b)]  
        self.velocity = [mu * v - eta * nwt for v, nwt in zip(self.velocity, nabla_w_total)]
        self.weights = [w + v for w, v in zip(self.weights, self.velocity)]
        
        
    # 该函数只负责计算不包含正则项的梯度
    def backprop(self, x, y):
        '''反向传播算法，返回权重和偏置项的梯度'''
        nabla_b = [np.zeros_like(b) for b in self.biases]
        nabla_w = [np.zeros_like(w) for w in self.weights]
        # 前向传播
        activation = x
        activations = [x] # 存储每一层的激活值（第一层的激活值就是输入值）
        zs = [] # 存储每一层的加权输入值
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.activator.forward(z)
            activations.append(activation)
        # 反向传播
        delta = self.cost.derivative(activations[-1], y) * self.activator.derivative(zs[-1]) # delta为误差关于加权输入的偏导数，同时也是输出层的偏执梯度
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activator.derivative(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T) # range的特性使得-l-1不会越界，正好为输入层
        return (nabla_b, nabla_w)
    
    
    def accuracy(self, data, convert=False):
        '''评估网络在数据集上的表现
        convert表示是否将标签从one-hot向量转换为索引
        即验证集或测试集(常见情景)不需要convert'''
        if convert:
            test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                            for (x, y) in data]    
        else:
            test_results = [(np.argmax(self.feedforward(x)), y)
                            for (x, y) in data]    
        return sum(int(x == y) for (x, y) in test_results)
    
    
    def total_cost(self, data, lmbda, convert=False):
        '''计算数据集上的总成本
        convert表示是否将标签从索引转换为one-hot向量
        即训练集(常见情景)不需要convert'''
        total_cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(y)
            total_cost += self.cost.forward(a, y)
        total_cost += 0.5 * (lmbda / len(data)) * sum(
            np.linalg.norm(w) ** 2 for w in self.weights) # L2正则化项
        return total_cost
    
    
    # 用于选择超参数
    # 使用tolist将numpy数组转换为Python原生列表以便JSON序列化
    def save(self, filename):
        '''将网络的结构和参数保存到文件中'''
        data = {
            "sizes": self.sizes,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
            "activator": self.activator.__name__,
            "cost": self.cost.__name__
        }
        with open(filename, "w") as f:
            json.dump(data, f)
            
    
    def load(filename):
        '''从文件中加载网络的结构和参数'''
        with open(filename, "r") as f:
            data = json.load(f)
        activator = globals()[data["activator"]]
        cost = globals()[data["cost"]]
        net = Network(data["sizes"], activator, cost)
        net.weights = [np.array(w) for w in data["weights"]]
        net.biases = [np.array(b) for b in data["biases"]]
        return net
    
    