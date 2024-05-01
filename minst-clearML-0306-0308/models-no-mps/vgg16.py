import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#定义训练的设备
device = torch.device("cpu")

resize = transforms.Resize((64,64))
tensor = transforms.ToTensor()
trans_compose = transforms.Compose([
    tensor,
    resize
])

#准备数据集
train_data = torchvision.datasets.MNIST("./data",train=True,transform=trans_compose,download=True)
test_data = torchvision.datasets.MNIST("./data",train=False,transform=trans_compose,download=True)
# print("size_of_train_data:{}".format(len(train_data)))   60000
# print("size_of_test_data:{}".format(len(test_data)))     10000

#加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloeader = DataLoader(test_data,batch_size=64)
#构建神经网络
vgg16 = torchvision.models.vgg16(weights = None)
vgg16.features[0] = nn.Conv2d(1,64,kernel_size=3)
vgg16.classifier[6]=nn.Linear(4096,10)

#测试所创建的神经网络
# if __name__ == '__main__':
#     img = torch.ones(64,1,28,28)
#     resize = transforms.Resize((64,64))
#     new_img = resize(img)
#     output = vgg16(new_img)
#     print("vgg16:",output.shape)

#创建网络模型
vgg16 =vgg16.to(device)
#损失函数：交叉熵损失
#loss_fn = nn.CrossEntropyLoss()   与nullloss对比后，发现CrossEntropyLoss性能在此数据集上性能较差
loss_fn = nn.NLLLoss()
loss_fn = loss_fn.to(device)
#优化器：随机梯度下降
learning_rate = 0.000001 #学习速率
optimizer = torch.optim.SGD(vgg16.parameters(),lr=learning_rate)

#设置训练网络的一些参数
total_train_num =0
total_test_num=0
epoch =10
writer = SummaryWriter("my_logs")

for i in tqdm(range(epoch)):
    print("——————第{}轮训练开始—————".format(i+1))

    #开始训练
    vgg16.train()
    for data in train_dataloader:
        imgs,targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = vgg16(imgs)
        loss = loss_fn(outputs,targets)
        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_num+=1#记录训练次数
        if total_train_num %500==0:
            print("训练次数：{}，LOSS：{}".format(total_train_num,loss.item()))

    #测试步骤开始：
    vgg16.eval()
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():
        for data in test_dataloeader:
            imgs,targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs =vgg16(imgs)
            loss= loss_fn(outputs,targets)
            total_test_loss=total_test_loss+loss.item()
            accuracy = (outputs.argmax(1)==targets).sum()
            total_accuracy = total_accuracy +accuracy
    print("测试集上的总共的LOSS：{}".format(total_test_loss))
    print("测试集上的准确率：{}".format(total_accuracy/len(test_data)))
    writer.add_scalar("test_loss",total_test_loss,total_test_num)
    writer.add_scalar("test_accurary", total_accuracy/len(test_data), total_test_num)
    total_test_num = total_test_num +1

torch.save(vgg16,"vgg16_pth")
print("模型已保存")
writer.close()