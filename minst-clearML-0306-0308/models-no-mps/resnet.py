import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#定义训练的设备
device = torch.device("cpu")
#准备数据集
train_data = torchvision.datasets.MNIST("./data",train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.MNIST("./data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
#加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloeader = DataLoader(test_data,batch_size=64)
#构建神经网络
resnet = torchvision.models.resnet18()
resnet.conv1 = nn.Conv2d(1,64,7,2,3,bias=False)
resnet.fc = nn.Linear(512,10)

#创建网络模型
resnet =resnet.to(device)
#损失函数：交叉熵损失
#loss_fn = nn.CrossEntropyLoss()   与nullloss对比后，发现CrossEntropyLoss性能在此数据集上性能较差
loss_fn = nn.NLLLoss()
loss_fn = loss_fn.to(device)
#优化器：随机梯度下降
learning_rate = 0.001 #学习速率
optimizer = torch.optim.SGD(resnet.parameters(),lr=learning_rate)

#设置训练网络的一些参数
total_train_num =0
total_test_num=0
epoch =20
writer = SummaryWriter("resnet_logs")

for i in range(epoch):
    print("——————第{}轮训练开始—————".format(i+1))

    #开始训练
    resnet.train()
    for data in train_dataloader:
        imgs,targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = resnet(imgs)
        loss = loss_fn(outputs,targets)
        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_num+=1#记录训练次数
        if total_train_num %100==0:
            print("训练次数：{}，LOSS：{}".format(total_train_num,loss.item()))

    #测试步骤开始：
    resnet.eval()
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_dataloeader:
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs =resnet(imgs)
            loss= loss_fn(outputs,targets)
            total_test_loss=total_test_loss+loss.item()
            accuracy = (outputs.argmax(1)==targets).sum()
            total_accuracy = total_accuracy +accuracy
    print("测试集上的总共的LOSS：{}".format(total_test_loss))
    print("测试集上的准确率：{}".format(total_accuracy/len(test_data)))
    writer.add_scalar("test_loss",total_test_loss,total_test_num)
    writer.add_scalar("test_accurary", total_accuracy/len(test_data), total_test_num)
    total_test_num = total_test_num +1

torch.save(resnet,"resent_pth")
print("模型已保存")
writer.close()