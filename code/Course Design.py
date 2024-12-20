import numpy as np #数值计算
import pandas as pd #数据处理
from sklearn.preprocessing import StandardScaler,OneHotEncoder #数据预处理
#构建和训练神经网络
import torch
import torch.nn as nn
import torch.optim as optim

#使用pandas读取训练集和测试集的CSV文件。
df_train = pd.read_csv(r"E:\Engineering Software\pythonProject\my_pythonProject\Deep Learning\Course Design\train.csv")
df_test = pd.read_csv(r"E:\Engineering Software\pythonProject\my_pythonProject\Deep Learning\Course Design\testA.csv")

floats_train = [] # 用于存储训练集的心跳信号数据
y_train_filtered = []  # 用于存储与特征匹配的标签

# 逐行处理心跳信号数据，同时保留对应的标签
for signal, label in zip(df_train['heartbeat_signals'], df_train['label']):
    try:
        # 将字符串按逗号分割为多个数值
        signal_list_1 = signal.split(',')
        # 将每个分割的数值字符串转换为浮点数，并加入到 floats 列表中
        floats_train.append([float(x) for x in signal_list_1])
        # 保留对应的标签
        y_train_filtered.append(label)
    except ValueError:
        continue  # 跳过无法转换的值

# 将 floats_train 列表转换为 NumPy 数组
floats_train_array = np.array(floats_train)

# 对测试集进行同样的处理
floats_test = []
for signal in df_test['heartbeat_signals']:
    try:
        signal_list_2 = signal.split(',')
        floats_test.append([float(x) for x in signal_list_2])
    except ValueError:
        continue
floats_test_array = np.array(floats_test)
# 初始化 StandardScaler
scaler = StandardScaler()
# 对训练集数据进行标准化
x_train_scaled = scaler.fit_transform(floats_train_array)
# 对测试集数据使用相同的 scaler 进行标准化
x_test_scaled = scaler.transform(floats_test_array)
# 使用处理后的标签数据
y_train_filtered = np.array(y_train_filtered)
# 独热编码
encoder = OneHotEncoder(sparse_output=False)
y_train_one_hot = encoder.fit_transform(y_train_filtered.reshape(-1, 1))

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 全连接层的定义会根据前向传播过程中计算出的特征维度动态调整
        self.fc1 = None
        self.fc2 = nn.Linear(128, 4)  # 输出有 4 个类别
    def forward(self, x):
        # 卷积层
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        # 动态展平，x.size(0) 是 batch_size
        x = x.view(x.size(0), -1)  # 展平后 shape: (batch_size, 特征数)
        # 如果 fc1 还没有定义，则根据展平后的特征维度动态定义
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 128)
        # 全连接层
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型、损失函数、优化器
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 转换数据格式为 PyTorch 张量
X_train_tensor = torch.from_numpy(x_train_scaled).float()
X_train_tensor = X_train_tensor.unsqueeze(1)  # 添加 channel 维度，形状变为 (batch_size, 1, n_features)
y_train_tensor = torch.from_numpy(y_train_one_hot.argmax(axis=1)).long()

# 训练循环
n_epochs = 10  # 假设训练 200 个 epoch
for epoch in range(n_epochs):
    model.train()  # 确保模型处于训练模式
    optimizer.zero_grad()  # 清除梯度
    outputs = model(X_train_tensor)  # 传入数据
    loss = criterion(outputs, y_train_tensor)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权重

# 转换测试集数据为 PyTorch 张量
X_test_tensor = torch.from_numpy(x_test_scaled).float()
X_test_tensor = X_test_tensor.unsqueeze(1)  # 添加 channel 维度，形状变为 (batch_size, 1, n_features)

# 设置模型为评估模式，不需要计算梯度
model.eval()

# 关闭梯度计算，因为在预测时不需要反向传播
with torch.no_grad():
    # 模型对测试集进行预测
    test_outputs = model(X_test_tensor)
    # 通过 softmax 获取每个类别的预测概率
    test_probabilities = torch.softmax(test_outputs, dim=1).numpy()

import pandas as pd
# 加载提交文件模板
sample_submit = pd.read_csv(r"E:\Engineering Software\pythonProject\my_pythonProject\Deep Learning\Course Design\sample_submit.csv")
# 将预测的概率插入到相应的列中
# 假设 test_probabilities 是形状为 (n_samples, 4) 的概率分布
sample_submit[['label_0', 'label_1', 'label_2', 'label_3']] = test_probabilities
# 保存提交文件
sample_submit.to_csv('sample_submit.csv', index=False)
print("预测结果已保存到 sample_submit.csv")
