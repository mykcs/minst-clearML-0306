import math

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 定义训练的设备
device = torch.device("cpu")

# 定义模型参数
input_dim = 784  # 输入图像的特征维度，例如MNIST数据集中的图像大小为28x28，所以输入维度为28x28=784
hidden_dim = 256  # Transformer模型中隐藏层的维度
num_classes = 10  # 图像分类任务的类别数量，例如MNIST数据集有10个类别（0-9）


# 准备数据集
train_data = torchvision.datasets.MNIST(
    "./data", train=True, transform=torchvision.transforms.ToTensor(), download=True
)
test_data = torchvision.datasets.MNIST(
    "./data", train=False, transform=torchvision.transforms.ToTensor(), download=True
)

# 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 构建Transformer模型

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.transformer_encoder = TransformerEncoder(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        embedded = self.embedding(x)  # 嵌入层处理输入
        encoded = self.positional_encoding(embedded)  # 位置编码
        transformed = self.transformer_encoder(encoded)  # Transformer编码器处理
        output: object = self.fc(transformed)  # 全连接层输出
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim):
        super(PositionalEncoding, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batch_size, seq_len = x.size()[:2]  # 获取输入张量的形状
        positional_encoding = self.generate_positional_encoding(seq_len, self.hidden_dim)  # 生成位置编码
        positional_encoding = positional_encoding.to(x.device)  # 将位置编码移动到与输入张量相同的设备
        x = x + positional_encoding.unsqueeze(0).expand_as(x)  # 将位置编码加到输入张量上
        return x

    def generate_positional_encoding(self, seq_len, hidden_dim):
        positional_encoding = torch.zeros(seq_len, hidden_dim)
        pos = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        positional_encoding[:, 0::2] = torch.sin(pos * div_term)
        positional_encoding[:, 1::2] = torch.cos(pos * div_term)
        return positional_encoding

class TransformerEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super(TransformerEncoder, self).__init__()

        self.attention = SelfAttention(hidden_dim)
        self.feed_forward = FeedForward(hidden_dim)

    def forward(self, x):
        x = self.attention(x)
        x = self.feed_forward(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        attention_weights = self.softmax(scores)
        attended_values = torch.matmul(attention_weights, value)
        x = x + attended_values
        return x

class FeedForward(nn.Module):
    def __init__(self, hidden_dim):
        super(FeedForward, self).__init__()

        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = x + x
        return x

# 创建网络模型
transformer_model = TransformerModel(input_dim, hidden_dim, num_classes)
transformer_model = transformer_model.to(device)

# 损失函数：交叉熵损失
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器：随机梯度下降
learning_rate = 0.01  # 学习速率
optimizer = torch.optim.SGD(transformer_model.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
total_train_num = 0
total_test_num = 0
epoch = 40
writer = SummaryWriter("transformer_logs")

for i in range(epoch):
    print("——————第{}轮训练开始—————".format(i + 1))

    # 开始训练
    transformer_model.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = transformer_model(imgs)
        loss: object = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_num += 1  # 记录训练次数
        if total_train_num % 500 == 0:
            print("训练次数：{}，LOSS：{}".format(total_train_num, loss))

    # 在测试集上进行评估
    transformer_model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = transformer_model(imgs)
            test_loss += loss_fn(outputs, targets).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == targets).sum().item()

    test_loss /= len(test_dataloader.dataset)
    accuracy = correct / len(test_dataloader.dataset)
    print("测试集上的平均损失：{:.4f}，准确率：{:.2f}%".format(test_loss, accuracy * 100))

    # 将训练和测试结果写入TensorBoard
    writer.add_scalar("Loss/train", loss, total_train_num)
    writer.add_scalar("Loss/test", test_loss, total_train_num)
    writer.add_scalar("Accuracy/test", accuracy, total_train_num)

# 关闭TensorBoard写入器
writer.close()