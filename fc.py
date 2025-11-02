from functools import reduce
import numpy as np
from activators import SigmoidActivator, ReluActivator

# 所有的偏导数按分母求导法则计算

# 全连接层实现类
class FullConnectedLayer(object):
    def __init__(self, input_size, output_size,
                 activator):
        '''
        构造函数
        input_size: 本层输入向量的维度
        output_size: 本层输出向量的维度
        activator: 激活函数
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # 权重数组W
        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        # 偏置项b（拿出单独作为一个向量而不是通过恒为1的节点放在权重矩阵中）
        self.b = np.zeros((output_size, 1))
        # 输出向量
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        '''
        前向计算
        input_array: 输入向量，维度必须等于input_size
        '''
        self.input = input_array
        self.output = self.activator.forward(np.dot(self.W, input_array) + self.b)

    def backward(self, delta_array):
        '''
        反向计算W和b的梯度
        delta_array: 本层的误差项，由上一层给出（输出层的误差项直接计算）
        self.delta：下一层的误差项
        '''
        self.delta = self.activator.backward(self.input) * np.dot(self.W.T, delta_array) 
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array # x关于b的偏导数是1

    def update(self, learning_rate):
        '''
        使用梯度下降算法更新权重
        '''
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad

    # 打印本层权重和偏置项
    def dump(self):
        print('W: %s\nb:%s' % (self.W, self.b))


# 神经网络类
class Network(object):
    def __init__(self, layers):
        '''
        构造函数
        '''
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], layers[i + 1],
                    SigmoidActivator() #选择激活函数
                )
            )

    def predict(self, sample):
        '''
        使用神经网络实现预测
        sample: 输入样本
        '''
        output = sample
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self, labels, data_set, rate, epoch):
        '''
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数
        '''
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self, label):
        delta = self.layers[-1].activator.backward(self.layers[-1].output) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta # 返回输入层上一层的delta（好像没意义）

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)

    def dump(self):
        for layer in self.layers:
            layer.dump()

    def loss(self, output, label):
        return 0.5 * ((label - output) * (label - output)).sum()

    def gradient_check(self, sample_feature, sample_label):
        '''
        梯度检查
        network: 神经网络对象
        sample_feature: 样本的特征
        sample_label: 样本的标签
        '''
        # 获取网络在当前样本下每个连接的梯度
        self.predict(sample_feature)
        self.calc_gradient(sample_label)

        # 检查梯度
        epsilon = 10e-4
        for fc in self.layers:
            for i in range(fc.W.shape[0]):
                for j in range(fc.W.shape[1]):
                    fc.W[i, j] += epsilon
                    output = self.predict(sample_feature)
                    err1 = self.loss(sample_label, output)
                    fc.W[i, j] -= 2 * epsilon
                    output = self.predict(sample_feature)
                    err2 = self.loss(sample_label, output)
                    expect_grad = (err1 - err2) / (2 * epsilon)
                    fc.W[i, j] += epsilon
                    print('weights(%d,%d): expected - actual %.4e - %.4e' % (i, j, expect_grad, fc.W_grad[i, j]))


from bp import train_data_set


# 同时转置多个向量/矩阵
def transpose(args):
    return list(map(
        lambda arg: list(map(
            lambda line: np.array(line).reshape(len(line), 1)
            , arg))
        , args
    ))


# 编码器
class Normalizer(object):
    def __init__(self):
        self.mask = [
            0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80
        ]

    def norm(self, number):
        data = list(map(lambda m: 0.9 if number & m else 0.1, self.mask))
        return np.array(data).reshape(8, 1) # 将行向量转置为列向量

    def denorm(self, vec):
        binary = list(map(lambda i: 1 if i > 0.5 else 0, vec[:, 0]))
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x, y: x + y, binary)


def correct_ratio(network):
    normalizer = Normalizer()
    correct = 0
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print('correct_ratio: %.2f%%' % (correct / 256 * 100))


if __name__ == '__main__':
    normalizer = Normalizer()
    data_set = []
    labels = []
    for i in range(0, 256):
        s = normalizer.norm(i)
        l = normalizer.norm(i)  
        data_set.append(s)
        labels.append(l)
        
    """    for i in range(len(data_set)):
        print('data: %s\tlabel: %s' % (normalizer.denorm(data_set[i]), normalizer.denorm(labels[i])))"""    

    labels, data_set = transpose(train_data_set()) # 转置为列向量
    net = Network([8, 7, 8])
    rate = 0.5
    mini_batch = 20
    epoch = 20
    for i in range(epoch):
        net.train(labels, data_set, rate, mini_batch)
        print('after epoch %d loss: %f' % (
            (i + 1),
            net.loss(labels[-1], net.predict(data_set[-1])) # 对最后一个样本计算误差
        ))
        rate /= 1.2 # 学习率衰减
        
    correct_ratio(net)  
    
    for i in range(15):
        print(normalizer.denorm(net.predict(normalizer.norm(i))))

