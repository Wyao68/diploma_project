"""
运行网络进行训练和验证的主程序
"""

# Standard library
import os

# My library
import FC_model

# Third-party libraries
import torch
import random
import numpy as np

def set_random_seed(seed=33):
    """设置所有随机种子，以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

RANDOM_SEED = set_random_seed()

if __name__ == "__main__":
    # 加载数据
    base = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base, 'saved_models')

    training_ds = torch.load(os.path.join(data_path, 'training_data.pt'), weights_only=False)
    test_ds = torch.load(os.path.join(data_path, 'test_data.pt'), weights_only=False)

    net = FC_model.FullyConnectedNet([5, 39, 116, 233, 2], dropout_p=0.0)
    
    net.running(training_ds, test_ds, epochs=100, batch_size=64, lr = 2.7e-3, weight_decay = 1e-4)

    