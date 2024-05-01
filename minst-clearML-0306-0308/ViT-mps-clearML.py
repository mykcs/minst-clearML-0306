import os

import clearml
import psutil
import torch
import torchvision
from clearml import Task
from colorama import Fore, Style
from einops import rearrange
from einops.layers.torch import Rearrange
from prettytable import PrettyTable
from rich import print
from rich.console import Console
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

clearml.browser_login()

# Always initialize ClearML before anything else. Automatic hooks will track as much as possible for you!
task = Task.init(
    project_name="pyPjct-minst-0306",
    task_name="Minst Training 06 -clearML",
    output_uri=True  # IMPORTANT: setting this to True will upload the model
    # If not set the local path of the model will be saved instead!
)

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

# 准备数据集
train_data = torchvision.datasets.MNIST("./data", train=True, transform=torchvision.transforms.ToTensor(),
                                        download=True)
test_data = torchvision.datasets.MNIST("./data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

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

'''
# Make sure ClearML knows these parameters are our hyperparameters!
task.connect(vit.parameters())
'''
# 设置训练网络的一些参数
total_train_num = 0
total_test_num = 0
num_epochs = 20
writer = SummaryWriter("runs-train-logs/logs_vit-mps-clearML")

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
    vit.train()
    for i, (images, targets) in enumerate(tqdm(train_dataloader)):
        images = images.to(device)
        targets = targets.to(device)
        outputs = vit(images)
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
    vit.eval()
    with torch.no_grad():
        for images, targets in test_dataloader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = vit(images)
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

# Save the model, saving the model will automatically also register it to
# ClearML thanks to the automagic hooks
vit.train.save_model("best_model")

# When a python script ends, the ClearML task is closed automatically. But in
# a notebook (that never ends), we need to manually close the task.
task.close()

writer.flush()
writer.close()
