import numpy as np  # 导入NumPy库，用于数值计算
import pandas as pd  # 导入Pandas库，用于数据处理
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # 导入数据预处理工具，用于数据标准化和标签编码

# 导入构建和训练神经网络的库
import torch  # 导入PyTorch库，用于构建神经网络
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
import matplotlib.pyplot as plt  # 导入Matplotlib库，用于数据可视化
from torch.utils.data import DataLoader

from tqdm import tqdm

# 检查 GPU 是否可用
is_cuda_available = torch.cuda.is_available()
print(f"CUDA 可用: {is_cuda_available}")

if is_cuda_available:
    # 获取 GPU 数量
    gpu_count = torch.cuda.device_count()
    print(f"GPU 数量: {gpu_count}")

    # 获取每个 GPU 的名称
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i} 名称: {gpu_name}")

    # 获取当前 GPU 的索引
    current_device = torch.cuda.current_device()
    print(f"当前使用的 GPU 索引: {current_device}")

    # 获取当前 GPU 的名称
    current_gpu_name = torch.cuda.get_device_name(current_device)
    print(f"当前 GPU 名称: {current_gpu_name}")
else:
    print("没有可用的 GPU。")


# 检查是否有可用的GPU，如果有就用第一个GPU，否则用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 使用Pandas读取训练集和测试集的CSV文件
df_train = pd.read_csv(r"E:\Engineering Software\PythonProject\Deep Learning\Course Design\train.csv")
df_test = pd.read_csv(r"E:\Engineering Software\PythonProject\Deep Learning\Course Design\testA.csv")

floats_train = []  # 初始化列表，用于存储训练集的心跳信号数据
y_train_filtered = []  # 初始化列表，用于存储与特征匹配的标签

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
        continue  # 如果转换失败，则跳过当前循环

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

# 初始化 OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
# 对标签进行独热编码
y_train_one_hot = encoder.fit_transform(y_train_filtered.reshape(-1, 1))

# 定义卷积神经网络模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # 定义第一个卷积层，输入通道1，输出通道16，卷积核大小3，填充1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # 定义第二个卷积层，输入通道16，输出通道32，卷积核大小3，填充1
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # 定义第三个卷积层，输入通道32，输出通道64，卷积核大小3，填充1
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 初始化全连接层为None，稍后根据特征维度动态调整
        self.fc1 = None
        # 定义第二个全连接层，输入特征128，输出类别4
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        # 通过第一个卷积层，并应用ReLU激活函数
        x = torch.relu(self.conv1(x))
        # 通过第二个卷积层，并应用ReLU激活函数
        x = torch.relu(self.conv2(x))
        # 通过第三个卷积层，并应用ReLU激活函数
        x = torch.relu(self.conv3(x))
        # 动态展平，x.size(0) 是 batch_size
        x = x.view(x.size(0), -1)  # 展平后 shape: (batch_size, 特征数)
        # 如果 fc1 还没有定义，则根据展平后的特征维度动态定义
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.shape[1], 128).to(device)  # 在GPU上定义全连接层
        # 通过第一个全连接层，并应用ReLU激活函数
        x = torch.relu(self.fc1(x))
        # 通过第二个全连接层
        x = self.fc2(x)
        return x

# 初始化模型、损失函数、优化器
model = CNNModel()  # 实例化CNN模型
#model.half() #将模型转换为FP16精度

model = model.to(device)

criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.01)  # 定义Adam优化器，学习率为0.005
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 定义学习率调整策略，每30个epoch衰减到原来的0.1倍

# 转换数据格式为 PyTorch 张量
X_train_tensor = torch.from_numpy(x_train_scaled).float().to(device) # 将训练集数据转换为浮点型张量,输入数据转换为FP16精度
X_train_tensor = X_train_tensor.unsqueeze(1)  # 增加一个维度，变为 (batch_size, 1, n_features)
y_train_tensor = torch.from_numpy(y_train_one_hot.argmax(axis=1)).long().to(device)  # 将独热编码的标签转换为类别索引

# 初始化用于记录训练损失的列表
train_epoch_losses = []
train_epoch_accuracies = []

# 创建 DataLoader
batch_size = 64  # 定义批处理大小
train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 训练循环
n_epochs = 10  # 设置训练的epoch数量为250
for epoch in range(n_epochs):

    # 初始化用于记录当前epoch的损失和准确率的变量
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0

    # 使用tqdm来创建进度条，total是这一轮训练中总的批次数量
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f'Epoch {epoch + 1}')
    for batch_index, (X_batch, y_batch) in progress_bar:
        X_batch = X_batch.to(device)  # 确保数据在正确的设备上
        y_batch = y_batch.to(device)  # 确保标签在正确的设备上

        model.train()  # 将模型设置为训练模式
        optimizer.zero_grad()  # 清除优化器的梯度
        outputs = model(X_batch)  # 将训练数据传入模型得到输出

        loss = criterion(outputs, y_batch)  # 计算损失

        _, predicted = torch.max(outputs, 1)  # 获取预测结果
        correct = (predicted == y_batch).sum().item()  # 计算正确预测的数量
        accuracy = correct / len(y_batch)  # 计算准确率

        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 根据梯度更新模型参数
        # scheduler.step()  # 更新学习率

        # 累加当前批次的损失和正确预测的数量
        epoch_loss += loss.item()
        epoch_correct += correct
        epoch_total += len(y_batch)

        # 更新进度条信息
        progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy)

    # 计算当前epoch的平均损失和准确率
    epoch_loss /= len(data_loader)
    epoch_accuracy = epoch_correct / epoch_total

    # 将当前epoch的平均损失和准确率添加到列表中
    train_epoch_losses.append(epoch_loss)
    train_epoch_accuracies.append(epoch_accuracy)


# 绘制训练损失折线图
plt.figure(figsize=(10, 5))  # 设置图像大小
plt.subplot(1, 2, 1)  # 创建一个1行2列的子图，并定位到第一个
plt.plot(train_epoch_losses, label='Training Loss')  # 绘制训练损失
plt.title('Training Loss Over Epochs')  # 设置标题
plt.xlabel('Epoch')  # 设置x轴标签
plt.ylabel('Loss')  # 设置y轴标签
plt.legend()  # 显示图例

# 绘制训练准确率折线图
plt.subplot(1, 2, 2)  # 创建一个1行2列的子图，并定位到第二个
plt.plot(train_epoch_accuracies, label='Training Accuracy')  # 绘制训练准确率
plt.title('Training Accuracy Over Epochs')  # 设置标题
plt.xlabel('Epoch')  # 设置x轴标签
plt.ylabel('Accuracy')  # 设置y轴标签
plt.legend()  # 显示图例
plt.show()  # 显示图像

# 转换测试集数据为 PyTorch 张量
X_test_tensor = torch.from_numpy(x_test_scaled).float().to(device)  # 将测试集数据转换为浮点型张量
X_test_tensor = X_test_tensor.unsqueeze(1)  # 增加一个维度，变为 (batch_size, 1, n_features)

# 设置模型为评估模式，不需要计算梯度
model.eval()

# 初始化用于记录测试损失的列表
test_losses = []
test_accuracies = []

# 关闭梯度计算，因为在预测时不需要反向传播
with torch.no_grad():
    # 模型对测试集进行预测
    test_outputs = model(X_test_tensor).to(device)
    # 通过 softmax 获取每个类别的预测概率
    test_probabilities = torch.softmax(test_outputs, dim=1).cpu().numpy()

torch.cuda.empty_cache()  # 手动清理显存

# 导入Pandas库，用于处理数据
import pandas as pd
# 加载提交文件模板
sample_submit = pd.read_csv(r"E:\Engineering Software\PythonProject\Deep Learning\Course Design\sample_submit.csv")
# 将预测的概率插入到相应的列中
# 假设 test_probabilities 是形状为 (n_samples, 4) 的概率分布
sample_submit[['label_0', 'label_1', 'label_2', 'label_3']] = test_probabilities
# 保存提交文件
sample_submit.to_csv('sample_submit.csv', index=False)
print("预测结果保存至 sample_submit.csv")  # 打印信息，表示预测结果已保存
