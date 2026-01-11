"""全连接神经网络基本模型

主要功能：
- 定义一个全连接神经网络类 FullyConnectedNet
- 实现训练和验证流程，并保存训练好的模型参数
"""
# Standard library
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau

# My library
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
                batch_size: int = 20):
        '''
        执行训练过程并保存训练过程信息以便可视化
        
        参数说明：
        - train_ds: 训练集
        - val_ds: 验证集
        - training_data_size: 指定训练数据集大小(当指定范围大于实际数据集时，自动调整为实际大小)
        - epochs: 训练轮次
        - batch_size: 批次大小
        
        保存信息：
        - tra_loss: 训练损失列表
        - val_L_Max_relevant_errs: 测试集电感最大相对误差列表
        - val_L_Avg_relevant_errs: 测试集电感平均相对误差列表 
        - val_R_Max_relevant_errs: 测试集电阻最大相对误差列表
        - val_R_Avg_relevant_errs: 测试集电阻平均相对误差列表       
        - val_loss: 测试损失列表
        - tra_L_Max_relevant_errs: 训练集电感最大相对误差列表
        - tra_L_Avg_relevant_errs: 训练集电感平均相对误差列表
        - tra_R_Max_relevant_errs: 训练集电阻最大相对误差列表
        - tra_R_Avg_relevant_errs: 训练集电阻平均相对误差列表
        '''
        # 创建损失与误差列表
        training_loss = []
        val_L_Max_relevant_errs = []
        val_L_Avg_relevant_errs = [] 
        val_R_Max_relevant_errs = []
        val_R_Avg_relevant_errs = []
        validate_loss = []
        tra_L_Max_relevant_errs = []
        tra_L_Avg_relevant_errs = []
        tra_R_Max_relevant_errs = []
        tra_R_Avg_relevant_errs = []

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
        
        # 训练与验证循环
        for epoch in range(1, epochs + 1):
            self.train()
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device) 
                batch_y = batch_y.to(device)

                optimizer.zero_grad()                # 每轮训练开始时梯度清零
                outputs = self(batch_x)              # 前向传播
                t_loss = criterion(outputs, batch_y) # 计算损失函数
                t_loss.backward()                    # 反向传播
                optimizer.step()                     # 参数更新

            # 在分别在训练集和验证集上评估这一轮的表现
            self.eval()
            
            tra_loss = 0.0 
            tra_L_Max_rel_err = 0.0 
            tra_L_Avg_rel_err = 0.0 
            tra_R_Max_rel_err = 0.0 
            tra_R_Avg_rel_err = 0.0 
            tra_total = 0 
            
            val_loss = 0.0 
            val_L_Max_rel_err = 0.0 
            val_L_Avg_rel_err = 0.0 
            val_R_Max_rel_err = 0.0 
            val_R_Avg_rel_err = 0.0
            val_total = 0 
            
            with torch.no_grad(): # 禁用梯度计算，以节省内存和计算资源
                for tx, ty in train_loader:
                    tx = tx.to(device) 
                    ty = ty.to(device)

                    t_outputs = self(tx)
                    t_loss = criterion(t_outputs, ty)
                    tra_loss += t_loss.item() * tx.size(0) 
                    # 每个样本每个维度的相对误差，形状 [batch_size, output_dim]
                    t_rel_err = (t_outputs - ty).abs() / ty.abs()

                    t_L_rel_err = t_rel_err[:, 0] # 电感相对误差
                    t_R_rel_err = t_rel_err[:, 1] # 电阻相对误差
                    
                    tra_L_Max_rel_err = max(tra_L_Max_rel_err, t_L_rel_err.max().item())
                    tra_R_Max_rel_err = max(tra_R_Max_rel_err, t_R_rel_err.max().item())

                    tra_L_Avg_rel_err += t_L_rel_err.sum().item()
                    tra_R_Avg_rel_err += t_R_rel_err.sum().item()
                    
                    tra_total += tx.size(0)

                tra_loss = tra_loss / tra_total 
                tra_L_Avg_rel_err = tra_L_Avg_rel_err / tra_total 
                tra_R_Avg_rel_err = tra_R_Avg_rel_err / tra_total 
                
                training_loss.append(tra_loss)
                tra_L_Max_relevant_errs.append(tra_L_Max_rel_err)
                tra_R_Max_relevant_errs.append(tra_R_Max_rel_err)
                tra_L_Avg_relevant_errs.append(tra_L_Avg_rel_err)
                tra_R_Avg_relevant_errs.append(tra_R_Avg_rel_err)

            with torch.no_grad():
                for vx, vy in val_loader:
                    vx = vx.to(device)
                    vy = vy.to(device)

                    v_outputs = self(vx)
                    v_loss = criterion(v_outputs, vy)

                    val_loss += v_loss.item() * vx.size(0)
                    v_rel_err = (v_outputs - vy).abs() / vy.abs()
                    v_L_rel_err = v_rel_err[:, 0]
                    v_R_rel_err = v_rel_err[:, 1]

                    val_L_Max_rel_err = max(val_L_Max_rel_err, v_L_rel_err.max().item())
                    val_R_Max_rel_err = max(val_R_Max_rel_err, v_R_rel_err.max().item())

                    val_L_Avg_rel_err += v_L_rel_err.sum().item()
                    val_R_Avg_rel_err += v_R_rel_err.sum().item()
                    
                    val_total += vx.size(0)

                val_loss = val_loss / val_total 
                val_L_Avg_rel_err = val_L_Avg_rel_err / val_total 
                val_R_Avg_rel_err = val_R_Avg_rel_err / val_total 
                
                validate_loss.append(val_loss)
                val_L_Max_relevant_errs.append(val_L_Max_rel_err)
                val_R_Max_relevant_errs.append(val_R_Max_rel_err)
                val_L_Avg_relevant_errs.append(val_L_Avg_rel_err)
                val_R_Avg_relevant_errs.append(val_R_Avg_rel_err)

            scheduler.step(val_loss) # 根据验证集的结果调整学习率

            # 打印本轮训练结果
            print(f"Epoch {epoch:02d} - "
                  f"Training Loss: {tra_loss:.4f}, ")

        # 保存模型参数字典
        torch.save(self.state_dict(), "saved_models\\coils_model_state_dict.pt")
        
            # 保存训练过程数据以供可视化
        with open("saved_models\\training_progress.json", "w") as f:
            json.dump([training_loss, \
                        val_L_Max_relevant_errs, val_L_Avg_relevant_errs, \
                        val_R_Max_relevant_errs, val_R_Avg_relevant_errs, \
                        validate_loss, \
                        tra_L_Max_relevant_errs, tra_L_Avg_relevant_errs, \
                        tra_R_Max_relevant_errs, tra_R_Avg_relevant_errs,], f)


if __name__ == '__main__':
    # 作为脚本执行时进行简单测试
    # 加载数据集
    train_ds, val_ds, test_ds, meta = data_processor.load_data()
    
    net_dims = [6, 32, 64, 64, 32, 2]  # 网络层维度列表
    model = FullyConnectedNet(net_dims)
    model.running(train_ds, val_ds, training_data_size=4000, epochs=100, batch_size=64)
    