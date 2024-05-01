import torch
import torchvision
from clearml import Task
from clearml.automation import HyperParameterOptimizer
from clearml.automation import UniformParameterRange, UniformIntegerParameterRange
from clearml.automation.optuna import OptimizerOptuna
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ------------------  Prepare Template Task  ------------------
# If you don't have one, create a new Task in ClearML for initial configuration:

# ------------------ Start Hyperparameter Optimization ------------------
# Always initialize ClearML before anything else. Automatic hooks will track as much as possible for you!
task = Task.init(
    project_name="pyPjct-minst-0306",
    task_name="Minst Training 05",
    output_uri=True,  # IMPORTANT: setting this to True will upload the model
    # If not set the local path of the model will be saved instead!
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False
)
# Get the ID of the template task
TEMPLATE_TASK_ID = task.id

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
train_dataloader = DataLoader(train_data, batch_size=256, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=256, shuffle=False)


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
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True),
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

task.connect(vit.parameters())

# 损失函数：交叉熵损失
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器：随机梯度下降
learning_rate = 0.01  # 学习速率
optimizer = torch.optim.SGD(vit.parameters(), lr=learning_rate)
optimizer = HyperParameterOptimizer(
    # specifying the task to be optimized, task must be in system already so it can be cloned
    base_task_id=TEMPLATE_TASK_ID,
    # setting the hyperparameters to optimize
    hyper_parameters=[
        UniformIntegerParameterRange('number_of_epochs', min_value=2, max_value=12, step_size=2),
        UniformIntegerParameterRange('batch_size', min_value=2, max_value=16, step_size=2),
        UniformParameterRange('dropout', min_value=0, max_value=0.5, step_size=0.05),
        UniformParameterRange('base_lr', min_value=0.00025, max_value=0.01, step_size=0.00025),
    ],
    # setting the objective metric we want to maximize/minimize
    objective_metric_title='accuracy',
    objective_metric_series='total',
    objective_metric_sign='max',

    # setting optimizer
    optimizer_class=OptimizerOptuna,

    # configuring optimization parameters
    execution_queue='default',
    max_number_of_concurrent_tasks=2,
    optimization_time_limit=60.,
    compute_time_limit=120,
    total_max_jobs=20,
    min_iteration_per_job=15000,
    max_iteration_per_job=150000,
)

# 设置训练网络的一些参数
total_train_num = 0
total_test_num = 0
num_epochs = 20
writer = SummaryWriter("runs-train-logs/logs_vit-mps-clearML")

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
    print()

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
    print("test set上 LOSS：{}".format(total_test_loss))
    print("test set上的准确率acc：{}".format(accuracy))
    print()
    writer.add_scalar("test_loss", average_test_loss, total_test_num)
    writer.add_scalar("test_accuracy", accuracy, total_test_num)
    total_test_num += 1

writer.flush()
writer.close()
