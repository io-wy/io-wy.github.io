---
title: Pytorch深度神经网络
date: 2025-02-06T10:54:27.000Z
tags: [dl]
category: 自用
comments: true
draft: false
---

## Pytorch深度神经网络

不懂就上官网，这篇没啥理论，纯上手

部分代码来自ai，当然我自己有跑过，ds学ai上手特别快

[torch — PyTorch 2.6 文档 - PyTorch 深度学习库](https://pytorch.ac.cn/docs/stable/torch.html)

### 模型构造

构造模型有几种方式，当然这些是可以组合起来用的

#### nn.Module

```python
import torch.nn as nn
import torch.nn.functional as F
class Mymodule(nn.Module):
    def __inti__(self):
        super(Mymodule,self).__init__()
        self.fc1=nn.Linear(784,256)
        self.fc2=nn.Linear(256,10)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        x=self.fc2(x)
        return x
model=Mymodule()
model(x)
```

灵活性高，可以自定义构建复杂模型，需要自定义前向传播逻辑

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x += residual  # 跳跃连接
        return F.relu(x)
```

自定义跳跃连接

#### nn.Sequential

简单易用，但是不能处理复杂的网络分支

```python
import torch.nn as nn
Model=nn.Sequential(
	nn.Linear(784,256),
	nn.ReLU(),
	nn.Linear(256,10)
)
```

#### nn.ModuleList/nn.ModuleDict

动态网络结构，更灵活

```python
class DynamicModel(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(128, 128) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x
```

#### 预定义模型

```python
import torchvision.module as models
model=models.resnet18(pretrained=True)
model.fc=nn.Linear(model.fc.in_features,10)
```

#### 函数式API

```python
import torch.nn.functional as F
class MyModel(nn.Module):
	def __init__(self):
		super(MyModel,self).__init__()
		self.fc1=nn.Linear(784,256)
		self.fc2=nn.Linear(256,10)
	def forward(self,x):
		x=F.relu(self.fc1(x))
		x=self.fc2(x)
		return x
model=MyModel
```

#### 混合模型

```python
class CNNRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.rnn = nn.RNN(32 * 13 * 13, 128, batch_first=True)

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x, _ = self.rnn(x)
        return x
```

### 模型训练

在神经网络中参数有两种，一种是超参数，需要手动调节的，如学习率，另一种参数则是在通过模型训练，自动寻找最优解，可以理解为权重

既然是最优解，那我们第一时间可以反映的自然是微分/求导，这也是使用pytorch的原因，不然可能就需要numpy+sympy+...求偏导，求jacobi矩阵...

训练神经网络，最常见的就是反向传播算法，参数（权重）会根据**损失函数**相对于给定参数的梯度进行调整，我的理解大概是把损失函数当成因变量，而权重作为自变量，求得梯度后就相当于找到了改变自变量的方向

#### 自动求导（梯度计算+传播

这个时候就需要介绍到**pytorch自动求导**了，torch.autograd，任何计算图进行梯度的自动计算[使用 torch.autograd 进行自动微分 — PyTorch Tutorials 2.6.0+cu124 文档 - PyTorch 深度学习库](https://pytorch.ac.cn/tutorials/beginner/basics/autogradqs_tutorial.html)

```python
loss.backward()
w.grad
b.grad#这两是参数
```

当我们已经完成模型训练后要使用模型，也就是只进行前向训练，这个时候就不需要梯度的

```python
with torch.no_grad:
	z=torch.matmul(x,w)+b
z = torch.matmul(x, w)+b
z_det = z.detach()#两种写法差不多
```

计算图是什么呢？这是一个由**张量+算子**构成的”流程图“，可以简单理解从左到右一步步进行状态转移

张量是什么应该没有人不知道，所以这里就分析一下算子

算子分为张量操作（张量运算），神经网络（特征提取，激活/损失函数），数据流（数据载入（shuffle随机乱序，batch_size分批载入），数据处理（参考机器学习的数据预处理））或许这些应该放在模型构造那里说？至于其他的（如算子之间的依赖和独立其实并不需要太过深究

在这一步完成了前向传播，梯度计算，反向传播，要实现对参数/权重的优化我们就需要**优化器**，经过优化器+梯度auto modify之后，就可以得到一个比较好的神经网络模型了，之后就给他喂训练集就好了

#### 损失函数

```python
criterion = nn.CrossEntropyLoss()
```

需要使用再临时查找吧

#### 优化

常见的优化算法不少，对这些算法的分析不会出现在这，这里只有优化器的常见使用方法

```python
import torch.optim as optim
model = MyModel()  # 假设 MyModel 是自定义的神经网络
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 设置学习率
for epoch in range(num_epochs):  # 遍历数据集多次
    for data, target in dataloader:  # 遍历数据加载器
        optimizer.zero_grad()  # 清零梯度
        output = model(data)   # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()        # 反向传播
        optimizer.step()       # 更新参数
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)
```

```python
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99)
optimizer = optim.Adagrad(model.parameters(), lr=0.01)
optimizer = optim.Adadelta(model.parameters(), rho=0.9)
```

之后是调参的问题了，**Adam**可以适应大部分深度学习任务

至于**保存和加载模型**此处不再赘述

### 手搓MLP

纯参考，方便理清思路，代码跑过，没问题

```python
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x#到这里模型构造就完成了

# 初始化模型(超参数)
input_size = X_train_leaves.shape[1]
hidden_size = 64
output_size = 1
model = Model(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()#到这里上面提到的就都完成了

    # 使用模型（验证集
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            val_loss += criterion(outputs, batch_y).item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
```

CNN就只需要多池化层和卷积核的知识就好了，主体框架其实也是一致的

### 补充

**数据预处理**在ml部分有提到，这里不多赘述，就写一写转换成tensor的一些步骤

```python
# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train_leaves, dtype=torch.float32)
y_train_tensor = torch.tensor(y1_train.values, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val_leaves, dtype=torch.float32)
y_val_tensor = torch.tensor(y1_test.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_leaves, dtype=torch.float32)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

关于**dataloader**的用法就去看官方文档吧

### 调参

### 背后的最优化（
