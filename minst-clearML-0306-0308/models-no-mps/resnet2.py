import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
#定义训练的设备
device = torch.device("cuda")

#准备数据集
train_data = torchvision.datasets.MNIST("./data",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.MNIST("./data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
# print("size_of_train_data:{}".format(len(train_data)))   60000
# print("size_of_test_data:{}".format(len(test_data)))     10000

#加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloeader = DataLoader(test_data,batch_size=64)
#构建神经网络
class ResidualBlock(nn.Module):
    """
    每一个ResidualBlock,需要保证输入和输出的维度不变
    所以卷积核的通道数都设置成一样
    """
    def __init__(self, channel):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size=3, padding=1)

    def forward(self, x):
        """
        ResidualBlock中有跳跃连接;
        在得到第二次卷积结果时,需要加上该残差块的输入,
        再将结果进行激活,实现跳跃连接 ==> 可以避免梯度消失
        在求导时,因为有加上原始的输入x,所以梯度为: dy + 1,在1附近
        """
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.res_block_1 = ResidualBlock(16)
        self.res_block_2 = ResidualBlock(32)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = self.res_block_1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.res_block_2(x)
        x = x.view(in_size, -1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)

#创建网络模型
resnet2 = ResNet()
resnet2  = resnet2 .to(device)
#损失函数：交叉熵损失
loss_fn = nn.NLLLoss()
loss_fn = loss_fn.to(device)
#优化器：随机梯度下降
learning_rate = 0.01 #学习速率
optimizer = torch.optim.SGD(resnet2.parameters(),lr=learning_rate)

#设置训练网络的一些参数
total_train_num =0
total_test_num=0
epoch =10
writer = SummaryWriter("resnet2_logs")

for i in range(epoch):
    print("——————第{}轮训练开始—————".format(i+1))

    #开始训练
    resnet2.train()
    for data in train_dataloader:
        imgs,targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = resnet2(imgs)
        loss = loss_fn(outputs,targets)

        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_num+=1#记录训练次数
        if total_train_num %500==0:
            print("训练次数：{}，LOSS：{}".format(total_train_num,loss.item()))

    #测试步骤开始：
    resnet2.eval()
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_dataloeader:
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs =resnet2(imgs)
            loss= loss_fn(outputs,targets)
            total_test_loss=total_test_loss+loss.item()
            accuracy = (outputs.argmax(1)==targets).sum()
            total_accuracy = total_accuracy +accuracy
    print("测试集上的总共的LOSS：{}".format(total_test_loss))
    print("测试集上的准确率：{}".format(total_accuracy/len(test_data)))
    writer.add_scalar("test_loss",total_test_loss,total_test_num)
    writer.add_scalar("test_accurary", total_accuracy/len(test_data), total_test_num)
    total_test_num = total_test_num +1

torch.save(resnet2,"resnet2_pth")
print("模型已保存")
writer.close()