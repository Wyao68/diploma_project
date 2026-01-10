"""全连接神经网络基本模型

主要功能：
- 定义一个全连接神经网络类 FullyConnectedNet
- 实现训练和验证流程，并保存训练好的模型参数
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import data_processor


class FullyConnectedNet(nn.Module):
    def __init__(self, net_dims, dropout_p=0.0):
        super().__init__()
        self.dropout_p = dropout_p
        layers = []

        # 构建隐藏层(带激活与 dropout)，输出层不加激活与dropout
        in_dim = net_dims[0]
        for i, out_dim in enumerate(net_dims[1:]):  # enumerate 同时返回索引和值
            is_last = (i == len(net_dims[1:]) - 1)
            layers.append(nn.Linear(in_dim, out_dim))
            if not is_last:
                layers.append(nn.ReLU())
                if self.dropout_p > 0.0:
                    layers.append(nn.Dropout(p=self.dropout_p))
            in_dim = out_dim

        self.model = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        # ReLU 更适合使用 Kaiming 初始化以保持前向方差稳定
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.model(x)
    
    def running(self, 
                train_ds: torch.utils.data.Dataset, 
                val_ds: torch.utils.data.Dataset, 
                training_data_size: int = 1000,
                epochs: int = 30, 
                batch_size: int = 20) -> tuple[list, list, list, list, list, list]:
        '''
        执行训练过程并返回相关信息以便训练过程可视化
        
        参数说明：
        - train_ds: 训练集
        - val_ds: 验证集
        - training_data_size: 指定训练数据集大小(当指定范围大于实际数据集时，自动调整为实际大小)
        - epochs: 训练轮次
        - batch_size: 批次大小
        
        返回：
        - tra_loss: 训练损失列表
        - val_Max_relevant_err: 测试集最大相对误差列表
        - val_Avg_relevant_err: 测试集平均相对误差列表        
        - val_loss: 测试损失列表
        - tra_Max_relevant_err: 训练集最大相对误差列表
        - tra_Avg_relevant_err: 训练集平均相对误差列表
        '''
        # 创建损失与误差列表
        training_loss = []
        val_Max_relevant_err = []
        val_Avg_relevant_err = [] 
        validate_loss = []
        tra_Max_relevant_err = []
        tra_Avg_relevant_err = []

        # 数据加载
        n_train = min(training_data_size, len(train_ds))
        train_subset = Subset(train_ds, list(range(n_train)))
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # 设置设备（GPU 优先）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        # 损失函数
        criterion = nn.MSELoss() 
        
        # 优化器
        optimizer = optim.AdamW(self.parameters(),
                       lr=1e-2,             # 学习率
                       betas=(0.9, 0.999),  # 一阶&二阶矩动量系数
                       weight_decay=1e-4,   # L2正则化系数
                       )    
           
        #学习率调度器
        scheduler = ReduceLROnPlateau(
                        optimizer,
                        mode='min',           # 监控损失最小化
                        factor=0.5,           # 学习率减半
                        patience=5,           # 5个epoch无改善就调整
                        threshold=1e-4,       # 改善阈值
                        threshold_mode='rel', # 相对变化
                        cooldown=0,           # 调整后冷却轮数
                        min_lr=1e-6,          # 最小学习率
                        )
        
        # 训练与验证循环(这里的相对误差计算的都不对，要修改)
        for epoch in range(1, epochs + 1):
            self.train()
            
            tra_loss = 0.0 
            tra_Max_rel_err = 0.0 
            tra_Avg_rel_err = 0.0 
            tra_total = 0 
            
            for batch_x, batch_y in train_loader:
                # 将数据也转移到与模型所在的设备上
                batch_x = batch_x.to(device) 
                batch_y = batch_y.to(device)

                optimizer.zero_grad()                # 每轮训练开始时梯度清零
                outputs = self(batch_x)              # 前向传播
                t_loss = criterion(outputs, batch_y) # 计算损失函数
                t_loss.backward()                    # 反向传播
                optimizer.step()                     # 参数更新

                tra_loss += t_loss.item() * batch_x.size(0) # item将单个元素的tensor转化为Python的基础数据格式
                # 每个样本每个维度的相对误差，形状 [batch_size, output_dim]
                t_rel_err = (outputs - batch_y).abs() / batch_y.abs()
                # 对每个样本沿输出维度求平均，得到形状 [batch_size]
                t_per_sample_rel = t_rel_err.mean(dim=1)
                tra_Max_rel_err = max(tra_Max_rel_err, t_per_sample_rel.max().item())
                # 累加每个样本的平均相对误差用于计算 epoch 平均
                tra_Avg_rel_err += t_per_sample_rel.sum().item()
                tra_total += batch_x.size(0)

            tra_loss = tra_loss / tra_total
            tra_Avg_rel_err = tra_Avg_rel_err / tra_total
            
            training_loss.append(tra_loss)
            tra_Max_relevant_err.append(tra_Max_rel_err)
            tra_Avg_relevant_err.append(tra_Avg_rel_err)

            # 在验证集上评估
            self.eval() # 验证模式
            
            val_loss = 0.0 
            val_Max_rel_err = 0.0 
            val_Avg_rel_err = 0.0 
            val_total = 0 
            
            with torch.no_grad(): # 禁用梯度计算，用于推理和评估阶段，以节省内存和计算资源
                for vx, vy in val_loader:
                    vx = vx.to(device)
                    vy = vy.to(device)

                    v_outputs = self(vx)
                    v_loss = criterion(v_outputs, vy)

                    val_loss += v_loss.item() * vx.size(0)
                    v_rel_err = (v_outputs - vy).abs() / vy.abs()
                    v_per_sample_rel = v_rel_err.mean(dim=1)
                    val_Max_rel_err = max(val_Max_rel_err, v_per_sample_rel.max().item())
                    val_Avg_rel_err += v_per_sample_rel.sum().item()
                    val_total += vx.size(0)
                    
            val_loss = val_loss / val_total                    
            val_Avg_rel_err = val_Avg_rel_err / val_total
            
            validate_loss.append(val_loss)
            val_Max_relevant_err.append(val_Max_rel_err)
            val_Avg_relevant_err.append(val_Avg_rel_err)

            scheduler.step(val_loss) # 根据验证集的结果调整学习率

            # 打印本轮训练与验证结果
            print(f"Epoch {epoch:02d} - "
                  f"Training Loss: {tra_loss:.4f}, "
                  f"Max_relevant_error: {val_Max_rel_err*100:.2f}%, Average_relevant_error: {val_Avg_rel_err*100:.2f}% - ")

        # 保存模型参数字典
        torch.save(self.state_dict(), "saved_models\\coils_model_state_dict.pt")

        return training_loss, val_Max_relevant_err, val_Avg_relevant_err, validate_loss, tra_Max_relevant_err, tra_Avg_relevant_err


if __name__ == '__main__':
    # 作为脚本执行时进行简单测试
    # 加载数据集
    train_ds, val_ds, test_ds, meta = data_processor.load_data()
    
    net_dims = [6, 32, 64, 64, 32, 2]  # 网络层维度列表
    model = FullyConnectedNet(net_dims)
    model.running(train_ds, val_ds, training_data_size=4000, epochs=100, batch_size=64)
    