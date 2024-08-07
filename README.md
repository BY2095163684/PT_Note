# 调包侠的赛博笔记本
无教学意义

## 训练的一般流程
1.准备数据集
```python
import torchvision
train_data = torchvision.datasets.CIFAR10(root="./data/train",train=True,transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root="./data/train",train=False,transform=torchvision.transforms.ToTensor())
# print(len(train_data))
# print(len(test_data))
```
2.加载数据集
```python
from torch.utiles.data import DataLoader
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)
```
3.搭建CNN
```python
from torch import nn
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024,64),
            nn.Linear(64,10)
        )

    def forward(self,x):
        x = self.model(x)
        return x

mycnn = MyCNN()
```
4.损失函数
```python
loss_fn = nn.CrossEntropyLoss()
```
5.优化器
```python
import torch
optimizer = torch.optim.SGD(mycnn.parameters(),lr=0.001)
```
6.训练网络
```python
train_step = 0  # 记录训练次数
test_step = 0  # 记录测试次数
epoch = 10  # 训练总次数

for i in range(epoch):
    # 训练
    for data in train_dataloader:
        imgs,targets = data
        ouputs = mycnn(imgs)

        loss = loss_fn(outputs,targets) # 损失值
        optimizer.zero_grad()   # 梯度清零
        loss.backward() # 反向传播
        optimizer.step()    # 参数改进

        train_step += 1
        print(f"{train_step}, Loss: {loss}")
    # 测试
    total_test_loss = 0 # 测试损失总值
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            outputs = mycnn(imgs)

            loss = loss_fn(outputs,targets)
            total_test_loss += loss
    print(f"Test loss: {total_test_loss}")

    torch.save(mycnn,f"./mycnn_{i}")
    print("Model is saved")
```