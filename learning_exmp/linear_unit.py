from perceptron import Perceptron

#定义激活函数f
f = lambda x: x

# 继承感知器作为父类
class LinearUnit(Perceptron):
    def __init__(self, input_num):
        '''初始化线性单元，设置输入参数的个数(调用父类的初始化方法，线性单元的激活函数固定为线性函数)'''
        Perceptron.__init__(self, input_num, f)


def get_training_dataset():
    '''
    捏造5个人的收入数据
    '''
    # 构建训练数据
    # 输入向量列表，每一项是工作年限
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    # 期望的输出列表，月薪，注意要与输入一一对应
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels    


def train_linear_unit():
    '''
    使用数据训练线性单元
    '''
    # 创建感知器，输入参数的特征数为1（工作年限）
    lu = LinearUnit(1)
    # 训练，迭代10轮, 学习速率为0.01
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 10, 0.01)
    #返回训练好的线性单元
    return lu


def plot(linear_unit):
    import matplotlib.pyplot as plt # 函数内导入，避免污染全局命名空间；引入别名，简化命名，避免重名并提高可读性
    input_vecs, labels = get_training_dataset()
    fig = plt.figure()
    ax = fig.add_subplot(111) # 创建1*1网格的第一个子图
    ax.scatter([x[0] for x in input_vecs], labels) # 实际散点图
    weights = linear_unit.weights
    bias = linear_unit.bias
    x = range(0,12,1)
    y = [weights[0] * xi + bias for xi in x] # 训练出的线性拟合图
    ax.plot(x, y)
    plt.show()


if __name__ == '__main__': 
    '''训练线性单元'''
    linear_unit = train_linear_unit()
    # 打印训练获得的权重
    print (linear_unit)
    # 测试
    print ('Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4]))
    print ('Work 15 years, monthly salary = %.2f' % linear_unit.predict([15]))
    print ('Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5]))
    print ('Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3]))
    plot(linear_unit)