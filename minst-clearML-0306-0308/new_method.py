import torch
import torchvision
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
class Mynn(nn.Module):
    def __init__(self):
        super(Mynn, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

#测试所创建的神经网络
# if __name__ == '__main__':
#     mynn = Mynn()
#     input=torch.ones(64,1,28,28)
#     output = mynn(input)
#     print(output.shape)

#创建网络模型
mynn = Mynn()
mynn = mynn.to(device)
#损失函数：交叉熵损失
#loss_fn = nn.CrossEntropyLoss()   与nullloss对比后，发现CrossEntropyLoss性能在此数据集上性能较差
loss_fn = nn.NLLLoss()
loss_fn = loss_fn.to(device)
#优化器：随机梯度下降
learning_rate = 0.01 #学习速率
optimizer = torch.optim.SGD(mynn.parameters(),lr=learning_rate)

#设置训练网络的一些参数
total_train_num =0
total_test_num=0
epoch =10
writer = SummaryWriter("my_logs")

for i in range(epoch):
    print("——————第{}轮训练开始—————".format(i+1))

    #开始训练
    mynn.train()
    for data in train_dataloader:
        imgs,targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = mynn(imgs)
        loss = loss_fn(outputs,targets)
        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_num+=1#记录训练次数
        if total_train_num %500==0:
            print("训练次数：{}，LOSS：{}".format(total_train_num,loss.item()))

    #测试步骤开始：
    mynn.eval()
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_dataloeader:
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs =mynn(imgs)
            loss= loss_fn(outputs,targets)
            total_test_loss=total_test_loss+loss.item()
            accuracy = (outputs.argmax(1)==targets).sum()
            total_accuracy = total_accuracy +accuracy
    print("测试集上的总共的LOSS：{}".format(total_test_loss))
    print("测试集上的准确率：{}".format(total_accuracy/len(test_data)))
    writer.add_scalar("test_loss",total_test_loss,total_test_num)
    writer.add_scalar("test_accurary", total_accuracy/len(test_data), total_test_num)
    total_test_num = total_test_num +1

torch.save(mynn,"mynn.pth")
print("模型已保存")
writer.close()