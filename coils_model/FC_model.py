"""全连接神经网络基本模型

主要功能：
- 定义一个全连接神经网络类 FullyConnectedNet
- 实现训练和验证流程，并保存训练好的模型参数与训练过程数据
"""
# Standard library
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau


base = os.path.dirname(os.path.dirname(__file__))
dict_path = os.path.join(base, "saved_models", "coils_model_state_dict.pt")


class FullyConnectedNet(nn.Module):
    def __init__(self, net_dims, dropout_p=0.0):
        super().__init__()
        self.dropout_p = dropout_p
        layers = []
        
        # layers.append(nn.Dropout(p=0.0)) # 输入层前的 dropout 层

        # 构建隐藏层(带激活与 dropout)，输出层不加激活与dropout
        in_dim = net_dims[0]
        for i, out_dim in enumerate(net_dims[1:]):  # enumerate 同时返回索引和值
            is_last = (i == len(net_dims[:]) - 2)  # 判断是否为隐藏层的最后一层
            layers.append(nn.Linear(in_dim, out_dim))
            if not is_last:
                layers.append(nn.ReLU())    # 添加激活函数
                if self.dropout_p > 0.0:
                    layers.append(nn.Dropout(p=self.dropout_p)) # 添加 dropout 层
            in_dim = out_dim

        self.model = nn.Sequential(*layers)

        self._init_weights()

    def forward(self, x):
        return self.model(x)

    def _init_weights(self):
        # 对 ReLU 使用 Kaiming 初始化以保持前向方差稳定
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def running(self, 
                train_ds: torch.utils.data.Dataset, 
                val_ds: torch.utils.data.Dataset, 
                training_data_size: int | None = None,
                epochs: int = 100, 
                batch_size: int = 64,
                lr = 1e-2,
                weight_decay = 1e-4):
        '''
        执行完整训练过程并保存训练过程信息以便可视化
        
        参数说明：
        - train_ds: 训练集
        - val_ds: 验证集
        - training_data_size: 指定训练数据集大小(当指定范围大于实际数据集时，自动调整为实际大小，不指定则默认使用全部数据)
        - epochs: 训练轮次
        - batch_size: 批次大小
        
        保存信息：
        - tra_loss: 训练损失列表
        - val_L_Max_relevant_errs: 验证集电感最大相对误差列表
        - val_L_Avg_relevant_errs: 验证集电感平均相对误差列表 
        - val_R_Max_relevant_errs: 验证集电阻最大相对误差列表
        - val_R_Avg_relevant_errs: 验证集电阻平均相对误差列表       
        - val_loss: 验证集损失列表
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
        if training_data_size is not None:
            n_train = min(training_data_size, len(train_ds))
            train_subset = Subset(train_ds, list(range(n_train)))
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # 设置设备（GPU 优先）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        # 损失函数（均方差）
        criterion = nn.MSELoss() 
        
        # 优化器
        optimizer = optim.AdamW(
                       self.parameters(),
                       lr=lr,           
                       weight_decay=weight_decay,  
                       )    
           
        #学习率调度器
        scheduler = ReduceLROnPlateau(
                        optimizer,
                        mode='min',     
                        factor=0.5,       
                        patience=8, 
                        threshold=1e-3,
                        threshold_mode='rel'          
                        )
        
        # 早停机制参数
        # best_val_loss = float('inf')
        # patience = float('inf') # 不启用早停机制
        # trigger_times = 0
        # 训练与验证循环
        for epoch in range(1, epochs + 1):
            self.train()
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device) 
                batch_y = batch_y.to(device)

                optimizer.zero_grad()                # 每个batch训练开始时梯度清零
                outputs = self.model(batch_x)        # 前向传播
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

                    t_outputs = self.model(tx)
                    t_loss = criterion(t_outputs, ty)
                    tra_loss += t_loss.item() * tx.size(0) # 平均值乘以样本数量得到总损失，以便后续计算平均损失
                    # 每个样本每个维度的相对误差，形状 [batch_size, output_dim]
                    t_rel_err = (t_outputs - ty).abs() / ty.abs()

                    t_L_rel_err = t_rel_err[:, 0] # 电感相对误差
                    t_R_rel_err = t_rel_err[:, 1] # 电阻相对误差
                    
                    # # 50轮训练后，相对误差大于3.0的样本被认为是异常值，不计入后续统计
                    # if epoch >= 50:                    
                    #     valid_mask = (t_L_rel_err < 3.0) & (t_R_rel_err < 3.0)
                    #     t_L_rel_err = t_L_rel_err[valid_mask]
                    #     t_R_rel_err = t_R_rel_err[valid_mask]
                    
                    tra_L_Max_rel_err = max(tra_L_Max_rel_err, t_L_rel_err.max().item()) if t_L_rel_err.numel() > 0 else 0
                    tra_R_Max_rel_err = max(tra_R_Max_rel_err, t_R_rel_err.max().item()) if t_R_rel_err.numel() > 0 else 0

                    tra_L_Avg_rel_err += t_L_rel_err.sum().item()
                    tra_R_Avg_rel_err += t_R_rel_err.sum().item()
                    
                    tra_total += t_L_rel_err.size(0)

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

                    v_outputs = self.model(vx)
                    v_loss = criterion(v_outputs, vy)

                    val_loss += v_loss.item() * vx.size(0)
                    v_rel_err = (v_outputs - vy).abs() / vy.abs()
                    
                    v_L_rel_err = v_rel_err[:, 0]
                    v_R_rel_err = v_rel_err[:, 1]
                    
                    # if epoch >= 50:                    
                    #     valid_mask = (v_L_rel_err < 3.0) & (v_R_rel_err < 3.0)
                    #     v_L_rel_err = v_L_rel_err[valid_mask]
                    #     v_R_rel_err = v_R_rel_err[valid_mask]

                    val_L_Max_rel_err = max(val_L_Max_rel_err, v_L_rel_err.max().item()) if v_L_rel_err.numel() > 0 else 0
                    val_R_Max_rel_err = max(val_R_Max_rel_err, v_R_rel_err.max().item()) if v_R_rel_err.numel() > 0 else 0

                    val_L_Avg_rel_err += v_L_rel_err.sum().item()
                    val_R_Avg_rel_err += v_R_rel_err.sum().item()
                    
                    val_total += v_L_rel_err.size(0)

                val_loss = val_loss / val_total 
                val_L_Avg_rel_err = val_L_Avg_rel_err / val_total 
                val_R_Avg_rel_err = val_R_Avg_rel_err / val_total 
                
                validate_loss.append(val_loss)
                val_L_Max_relevant_errs.append(val_L_Max_rel_err)
                val_R_Max_relevant_errs.append(val_R_Max_rel_err)
                val_L_Avg_relevant_errs.append(val_L_Avg_rel_err)
                val_R_Avg_relevant_errs.append(val_R_Avg_rel_err)
                
            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     trigger_times = 0
            #     torch.save(self.state_dict(), dict_path) # 保存当前最优模型权重
            # else:
            #     trigger_times += 1
            #     if trigger_times >= patience:
            #         print("Early stopping!")
            #         break

            # 学习率调度器根据验证集的结果调整学习率
            scheduler.step(val_loss) 

            # 打印本轮训练结果
            print(f"Epoch {epoch:02d} - "
                  f"Training Loss: {tra_loss:.4f}")

        # 全部训练过程结束后保存最终模型权重
        torch.save(self.state_dict(), dict_path)
        
        # 保存训练过程数据以供可视化
        with open(os.path.join(base, "saved_models", "training_progress.json"), "w") as f:
            json.dump([training_loss, \
                        val_L_Max_relevant_errs, val_L_Avg_relevant_errs, \
                        val_R_Max_relevant_errs, val_R_Avg_relevant_errs, \
                        validate_loss, \
                        tra_L_Max_relevant_errs, tra_L_Avg_relevant_errs, \
                        tra_R_Max_relevant_errs, tra_R_Avg_relevant_errs], f)

    