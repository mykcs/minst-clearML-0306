import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
#定义训练的设备
device = torch.device("cpu")

#准备数据集
train_data = torchvision.datasets.MNIST("./data",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.MNIST("./data",train=False,transform=torchvision.transforms.ToTensor(),download=True)

#加载数据集
train_dataloader = DataLoader(train_data,batch_size=256)
test_dataloeader = DataLoader(test_data,batch_size=256)
#构建神经网络
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),  # padding=2保证输入输出尺寸相同
            nn.ReLU(),  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),  # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)

#创建网络模型
lenet = LeNet()
lenet = lenet.to(device)
#损失函数：交叉熵损失
loss_fn = nn.NLLLoss()
loss_fn = loss_fn.to(device)
#优化器：随机梯度下降
learning_rate = 0.01 #学习速率
optimizer = torch.optim.SGD(lenet.parameters(),lr=learning_rate)

#设置训练网络的一些参数
total_train_num =0
total_test_num=0
epoch =40
writer = SummaryWriter("../lenet_logs")

for i in range(epoch):
    print("——————第{}轮训练开始—————".format(i+1))

    #开始训练
    lenet.train()
    for data in train_dataloader:
        imgs,targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = lenet(imgs)
        loss = loss_fn(outputs,targets)
        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_num+=1#记录训练次数
        if total_train_num %500==0:
            print("训练次数：{}，LOSS：{}".format(total_train_num,loss.item()))

    #测试步骤开始：
    lenet.eval()
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_dataloeader:
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs =lenet(imgs)
            loss= loss_fn(outputs,targets)
            total_test_loss=total_test_loss+loss.item()
            accuracy = (outputs.argmax(1)==targets).sum()
            total_accuracy = total_accuracy +accuracy
    print("测试集上的总共的LOSS：{}".format(total_test_loss))
    print("测试集上的准确率：{}".format(total_accuracy/len(test_data)))
    writer.add_scalar("test_loss",total_test_loss,total_test_num)
    writer.add_scalar("test_accurary", total_accuracy/len(test_data), total_test_num)
    total_test_num = total_test_num +1

torch.save(lenet, "../lenet_pth")
print("模型已保存")
writer.close()