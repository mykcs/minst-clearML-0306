#vgg_2.py
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

#定义训练的设备：用GPU来运行
device = torch.device("cpu")
#对数据集进行变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])
#准备数据集
train_data=torchvision.datasets.MNIST("./data",train=True,transform=transform,download=True)
test_data=torchvision.datasets.MNIST("./data",train=False,transform=transform,download=True)
# print("size_of_train_data:{}".format(len(train_data)))   60000张图片用来训练
# print("size_of_test_data:{}".format(len(test_data)))     10000张图片用来测试

#加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloeader = DataLoader(test_data,batch_size=64)
#构建神经网络
class Vgg_2(nn.Module):
    def __init__(self):
        super(Vgg_2, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 128, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(tensor=m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(tensor=m.weight, val=1)
                nn.init.constant_(tensor=m.bias, val=0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(tensor=m.weight, mean=0, std=0.01)
                nn.init.constant_(tensor=m.bias, val=0)

#创建网络模型
vgg_2 = Vgg_2()
vgg_2 = vgg_2.to(device)
#损失函数：交叉熵损失
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
#优化器：随机梯度下降
learning_rate = 0.01 #学习速率
optimizer = torch.optim.SGD(vgg_2.parameters(),lr=learning_rate)

#设置训练网络的一些参数
total_train_num =0
total_test_num=0
epoch =10
writer = SummaryWriter("vgg2_logs")#tensorboard用来展示结果

for i in range(epoch):
    print("——————第{}轮训练开始—————".format(i+1))
    #开始训练
    vgg_2.train()
    for data in train_dataloader:
        imgs,targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs =vgg_2(imgs)
        loss = loss_fn(outputs,targets)#计算loss
        optimizer.zero_grad() #优化器优化模型
        loss.backward() #根据loss改变参数
        optimizer.step()
        total_train_num+=1#记录训练次数
        if total_train_num %500==0:#每500次训练后输出LOSS值
            print("训练次数：{}，LOSS：{}".format(total_train_num,loss.item()))

    #测试步骤开始：
    vgg_2.eval()
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_dataloeader:
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs =vgg_2(imgs)
            loss= loss_fn(outputs,targets)#计算loss
            total_test_loss=total_test_loss+loss.item()
            accuracy = (outputs.argmax(1)==targets).sum()#计算预测准确的数据的个数
            total_accuracy = total_accuracy +accuracy#统计总共的预测准确的数据的个数
    #打印本次训练的结果：
    print("测试集上的LOSS：{}".format(total_test_loss))
    print("测试集上的准确率：{}".format(total_accuracy/len(test_data)))#total_accuracy/len(test_data)是准确率
    writer.add_scalar("test_loss",total_test_loss,total_test_num)
    writer.add_scalar("test_accurary", total_accuracy/len(test_data), total_test_num)
    total_test_num = total_test_num +1
#保存模型
torch.save(vgg_2, "../vgg2_pth")
print("模型已保存")
writer.close()