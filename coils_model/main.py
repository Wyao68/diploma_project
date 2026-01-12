"""
运行网络进行训练和验证的主程序
"""

# Standard library
import json

# My library
import data_processor
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
    training_data, validation_data, test_data, meta = data_processor.load_data(val_ratio = 0.3, test_ratio = 0.0, random_seed = RANDOM_SEED)
    # 保存 meta 以供后续调用模型
    with open("saved_models\\meta.json", "w") as f:json.dump(meta, f)

    net = FC_model.FullyConnectedNet([6, 32, 64, 64, 32, 2], dropout_p=0.0) 
    
    net.running(training_data, validation_data, training_data_size=4000 ,epochs=100, batch_size=64)

    