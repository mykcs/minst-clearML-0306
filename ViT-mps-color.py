import os

import psutil
import torch
import torchvision
from colorama import Fore, Style
from einops import rearrange
from einops.layers.torch import Rearrange
from prettytable import PrettyTable
from rich import print
from rich.console import Console
from torch import nn
from torch.nn import TransformerEncoder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

console = Console()
console.print("This is text formatted with [bold magenta]rich[/bold magenta]!", style="bold red")

# 定义训练的设备
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS is available. ")
else:
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled. ")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine. ")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 构建ViT模型
class ViT(nn.Module):
    def __init__(self,
                 patch_size=16,
                 embed_dim=768,
                 depths=(1, 1, 8, 12),
                 num_heads=(12, 12, 24, 36),
                 num_classes=10):
        super(ViT, self).__init__()

        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, 1, int(1000 / patch_size) ** 2, embed_dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))

        self.dropout = nn.Dropout(0.1)

        self.blocks = nn.ModuleList([
            TransformerEncoder(embed_dim, depth, num_heads) for depth, num_heads in zip(depths, num_heads)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # 将图像分割成 patch 并编码成向量
        x = self.patch_embed(x)

        # 添加位置编码
        b, c, h, w = x.shape
        x = x.flatten(2)
        x += self.pos_embed[:, :, :h * w]

        # 添加 CLS token
        cls_tokens = self.cls_token.expand(b, -1, -1, -1)
        x = torch.cat((cls_tokens, x), dim=2)

        # Transformer 编码器
        for block in self.blocks:
            x = block(x)

        # 取 CLS token 的输出并进行分类
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)

        return x



# 准备数据集
train_data = torchvision.datasets.MNIST("./data", train=True, transform=torchvision.transforms.ToTensor(),
                                        download=True)
test_data = torchvision.datasets.MNIST("./data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

# 加载数据集
train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

model = ViT().to(device)

# 损失函数：交叉熵损失
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器：随机梯度下降
learning_rate = 0.01  # 学习速率
optimizer = torch.optim.SGD(vit.parameters(), lr=learning_rate)

# 将 vit-mps 模型的日志写入 logs/vit-mps 目录
writer = SummaryWriter(log_dir="runs-train-logs/logs_vit-mps")

# 设置训练网络的一些参数
total_train_num = 0
total_test_num = 0
num_epochs = 20


# 要显示的指标
metrics = ["Epoch", "Iter", "GPU Memory", "Loss", "Accuracy"]
metrics_epochEnd = ["Epoch", "Iter", "GPU Memory", "Loss_epochEnd", "Accuracy_epochEnd"]
# 创建表格
table = PrettyTable(metrics)
table_epochEnd = PrettyTable(metrics_epochEnd)

for epoch in range(num_epochs):
    print(Fore.YELLOW + "————————————Train Round {}———————————".format(epoch + 1) + Style.RESET_ALL)  # 黄色标题

    running_loss = 0.0
    running_accuracy = 0.0
    total_correct = 0  # 记录正确预测的总数

    # 开始训练
    model.train()
    for i, (images, targets) in enumerate(tqdm(train_dataloader)):
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, dim=1)
        running_accuracy += predicted.eq(targets).sum().item()
        total_correct += predicted.eq(targets).sum().item()

        total_train_num += 1  # 记录训练次数

        if total_train_num % 500 == 0:
            process = psutil.Process(os.getpid())  # 获取当前进程
            gpu_memory = process.memory_info().rss / 1024 ** 3  # GB
            table.add_row([epoch + 1, i + 1, gpu_memory, running_loss / 500, running_accuracy / 500])
            print()
            print(table)
            table.clear_rows()  # 清除旧行，准备下一次更新

            # 增强训练信息打印
            print(Fore.GREEN + "训练次数：{}，LOSS：{}".format(total_train_num, loss.item()) + Style.RESET_ALL)

    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    model.eval()
    with torch.no_grad():
        for images, targets in test_dataloader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            running_accuracy += predicted.eq(targets).sum().item() / targets.size(0)
            correct = predicted.eq(targets).sum().item()
            total_accuracy += correct

    print("epoch end")
    average_test_loss = total_test_loss / len(test_dataloader)
    accuracy = total_accuracy / len(test_data)

    process = psutil.Process(os.getpid())  # 获取当前进程
    gpu_memory = process.memory_info().rss / 1024 ** 3  # GB
    table_epochEnd.add_row([epoch + 1, i + 1, gpu_memory, total_test_loss, accuracy])
    print(table_epochEnd)
    # table_epochEnd.clear_rows()  # 清除旧行，准备下一次更新

    print("test set all LOSS：{}".format(total_test_loss))
    print("test set上的准确率acc：{}".format(accuracy))
    print()
    writer.add_scalar("test_loss", average_test_loss, total_test_num)
    writer.add_scalar("test_accuracy", accuracy, total_test_num)
    total_test_num += 1

    # 重置统计值
    running_loss = loss.item()
    running_accuracy = 0.0

writer.flush()
writer.close()
