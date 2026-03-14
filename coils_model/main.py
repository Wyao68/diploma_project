"""
运行网络进行训练和验证的主程序
"""

# Standard library
import os
import torch
import random
import numpy as np
import json

# My library
import FC_model


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

    # 从best_hyperparams.json中加载模型和训练参数
    with open(os.path.join(data_path, 'best_hyperparams.json'), "r") as f:
        hyperparams = json.load(f)
    
    net_dims = [7]
    for i in range(hyperparams['n_hidden']):
        net_dims.append(hyperparams[f'n_units_layer{i}'])
    net_dims.append(2)
    
    net = FC_model.FullyConnectedNet(net_dims, dropout_p=hyperparams['dropout_p'])
    
    net.running(training_ds, test_ds, epochs=150, batch_size=64, lr =hyperparams['lr'], weight_decay =hyperparams['weight_decay'])

    