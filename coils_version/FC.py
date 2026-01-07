import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import data_loader


# 定义全连接网络类，继承自 nn.Module(torch库的核心类，所有神经网络模型都应继承自它)
class FullyConnectedNet(nn.Module):
    def __init__(self, net_dims, dropout_p=0.4):
        super().__init__()
        self.dropout_p = dropout_p
        layers = []

        # 构建隐藏层(带激活与 dropout)，输出层不加激活与dropout
        in_dim = net_dims[0]
        for i, out_dim in enumerate(net_dims[1:]):
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
    
    def running(self, train_ds, val_ds, training_data_size ,epochs, batch_size):
        # 返回val_cost, val_accuracy, training_cost, training_accuracy以便训练过程可视化
        val_cost, val_accuracy = [],[]
        training_cost, training_accuracy = [],[]

        # 数据加载
        n_train = min(training_data_size, len(train_ds))
        train_subset = Subset(train_ds, list(range(n_train)))
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # 设置设备（GPU 优先）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        criterion = nn.CrossEntropyLoss() # 损失函数
        # adam结合了动量法和自适应学习率，为每个参数计算适合的学习率
        optimizer = optim.AdamW(self.parameters(),
                       lr=1e-3,             # 学习率
                       betas=(0.9, 0.999),  # 一阶&二阶矩动量系数
                       eps=1e-08,           # 数值稳定项(作为除数时的最小值)
                       weight_decay=1e-2,   # L2正则化系数
                       amsgrad=False)       # 优化器设置

        # 训练与验证循环
        for epoch in range(1, epochs + 1):
            self.train() # 使模型进入训练模式
            running_loss = 0.0 # 损失值
            correct = 0 # 正确预测数
            total = 0 # 总样本数

            for batch_x, batch_y in train_loader:
                # 每次遍历时，进行shuffle
                batch_x = batch_x.to(device) # 将数据也转移到与模型相同的设备上
                batch_y = batch_y.to(device)

                optimizer.zero_grad()              # 每轮训练开始时梯度清零
                outputs = self(batch_x)            # 前向传播，得到 logits(batch_size, num_labels)
                loss = criterion(outputs, batch_y) # 计算平均交叉熵损失
                loss.backward()                    # 触发自动微分进行反向传播
                optimizer.step()                   # 参数更新

                running_loss += loss.item() * batch_x.size(0) # item将单个元素的tensor转化为Python的基础数据格式
                preds = outputs.argmax(dim=1) # 针对分类问题
                correct += (preds == batch_y).sum().item()
                total += batch_x.size(0)

            epoch_loss = running_loss / total 
            epoch_acc = correct / total 
        
            training_cost.append(epoch_loss)
            training_accuracy.append(epoch_acc)

            # 在验证集上评估
            self.eval() # 进入验证模式
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            with torch.no_grad(): # 禁用梯度计算，用于推理和评估阶段，以节省内存和计算资源
                for vx, vy in val_loader:
                    vx = vx.to(device)
                    vy = vy.to(device)
                    vout = self(vx)
                    vloss = criterion(vout, vy)
                    val_loss += vloss.item() * vx.size(0)
                    vpreds = vout.argmax(dim=1)
                    val_correct += (vpreds == vy).sum().item()
                    val_total += vx.size(0)

            val_loss = val_loss / val_total 
            val_acc = val_correct / val_total 
        
            val_cost.append(val_loss)
            val_accuracy.append(val_acc)
        

            # 打印本轮训练与验证结果
            print(f"Epoch {epoch:02d}/{epochs} - "
                f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 保存整个模型（包含结构与参数）
        torch.save(self, "fc_model_full.pt")
        # 或仅保存参数字典（更推荐的做法）
        torch.save(self.state_dict(), "fc_model_state_dict.pt")
        
        return val_cost, val_accuracy, training_cost, training_accuracy


if __name__ == '__main__':
    # 加载数据集
    train_ds, val_ds, test_ds = data_loader.load_data_wrapper()
    
    net_dims = [784, 30, 10]  # 网络层维度列表
    model = FullyConnectedNet(net_dims)
    model.running(train_ds, val_ds, training_data_size=2000, epochs=20, batch_size=200)
    