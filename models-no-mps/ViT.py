import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from einops.layers.torch import Rearrange
from einops import rearrange

# 定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 准备数据集
train_data = torchvision.datasets.MNIST("./data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST("./data", train=False, transform=torchvision.transforms.ToTensor(), download=True)

# 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

# 构建ViT模型
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, hidden_dim, num_heads, num_layers):
        super(ViT, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 1 * patch_size ** 2

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c')
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads),
            num_layers=num_layers
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = rearrange(x, 'b n d -> n b d')  # 将序列维度放在第一维
        x = self.transformer(x)
        x = x.mean(dim=0)  # 取序列维度的平均值
        x = self.classifier(x)
        return x

# 创建ViT模型实例
image_size = 28
patch_size = 7
num_classes = 10
hidden_dim = 64
num_heads = 4
num_layers = 4

vit = ViT(image_size, patch_size, num_classes, hidden_dim, num_heads, num_layers)
vit = vit.to(device)

# 损失函数：交叉熵损失
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器：随机梯度下降
learning_rate = 0.01  # 学习速率
optimizer = torch.optim.SGD(vit.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
total_train_num = 0
total_test_num = 0
num_epochs = 40
writer = SummaryWriter("vit_logs")

for epoch in range(num_epochs):
    print("——————第{}轮训练开始—————".format(epoch + 1))

    # 开始训练
    vit.train()
    for images, targets in tqdm(train_dataloader):
        images = images.to(device)
        targets = targets.to(device)
        outputs = vit(images)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_num += 1  # 记录训练次数
        if total_train_num % 500 == 0:
            print("训练次数：{}，LOSS：{}".format(total_train_num, loss.item()))

    # 测试步骤开始：
    vit.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for images, targets in test_dataloader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = vit(images)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)
            correct = predicted.eq(targets).sum().item()
            total_accuracy += correct

    average_test_loss = total_test_loss / len(test_dataloader)
    accuracy = total_accuracy / len(test_data)
    print("测试集上的总共的LOSS：{}".format(total_test_loss))
    print("测试集上的准确率：{}".format(accuracy))
    writer.add_scalar("test_loss", average_test_loss, total_test_num)
    writer.add_scalar("test_accuracy", accuracy, total_test_num)
    total_test_num += 1

writer.flush()
writer.close()