---
title: 多任务学习MTL
date: 2025-03-24T10:54:27.000Z
tags: [ai]
category: 教程
comments: true
draft: false
---

## 多任务学习(MTL)

multy-task learning 同时学习不同领域的任务，通过特定领域的信息提高泛化能力，这里的research gap似乎有点大，所有....（手动狗头）

一个好的多任务学习工作，必然会在任务间的联系下很大的功夫，这也就是创新点所在。相对于单任务学习，多任务学习的创新点在于网络架构和loss的设计，即使前段时间我们仍旧不死心的试图寻找新的办法来做，但并未成功...

对于很多提到的多任务学习的特点，**关联任务**，**正则化**(防止过拟合)，其实我个人觉得没有太大的必要多说，前者作为科研创新显而易见，后者在实验的过程中水到渠成QAQ，还有**共享架构**节约内存加快计算也是不违反直觉的，所以都很容易接受

### 网络架构

多任务学习意味着要多个网络并行或者拼接，就会存在一些问题(尤其是size上的)，既然大部分blog主要阐述的是软参数共享和硬参数共享，稍微贴心一点，这里也说明一下

#### 硬参数共享

这是最原始的MTL方法，多个任务共享底层网络，然后顶层多个任务头分别输出，这样做的好处很明显，就是计算高效，还能避免过拟合，不过由实验得知不同任务的效果不一样，有的时候性能可能会下降

```python
class eg(nn.Module):
	def __init__(self):
		self.shared_encoder=ResNet50()
        self.task1=nn.Linear(xx)
        self.task2=nn.conv2d(xx)
    def forward(self,x):
        features=self.shared_encoder(x)
        out1=self.task1(features)
        out2=self.task2(features)
        return out1,out2
```

没有代码补全真的是一种折磨

#### 软参数共享

每个任务都有独立的网络，通过约束(loss)联系在一起，既然放到硬参数共享后面那自然优点就是它的缺点的solution了，不过参数量太大了，比如我现在做的mtl，5个epoch跑了一个小时。。。

```python
class semantic_map():
	pass
class object_dection():
    pass
def loss_semantic():
    pass
def loss_dection():
    pass
loss=w1*loss_semantic+w2*loss_dection
```

主要还是loss之间的约束，如果想要w1,w2可学习的话，可以用random+归一化，让模型自己去学

#### personal trick

就是记录一下自己遇到的坑

因为我的网络比较复杂，把cv/nlp模型对接了，所以出现了非常非常多的特征不对齐的issue，为此我花费了整整两天的时间来debug

用到了一些层

##### 自适应池化层

```python
self.pool = nn.AdaptiveAvgPool2d((4, 4))#这是平均池化，把Avg改成Max就是最大池化
aligned_feat = self.pool(feat)  # 输入 [B, C, H, W] → 输出 [B, C, 4, 4]
```

##### 特征变换层

```python
self.channel_trans=nn.Conv2d(256,512,kernel_size=1)
```

256通道变成512通道，用于通道对齐

##### 拼接问题

```python
self.fusion = nn.Sequential(
    nn.Conv2d(256+128, 256, kernel_size=1),  # 拼接后降维
    nn.ReLU()
)
fused_feat = self.fusion(torch.cat([feat1, feat2], dim=1))
```

直接拼接后降维，这也是我初步做的东西，后面可能还有再设置的高级一点concat+Conv

##### 注意力机制

自注意力

```python
self.attention=nn.MultiheadAttention(embed_dim=256,num_head=8)
aligned_feat,_=self.attention(feat.feat,feat)
```

如果是B,C,W,H需要reshape成词长度的那几个特征，我也网络

cross attention

```python
# 计算图像特征（Q）和文本特征（K/V）的对齐
scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, H*W, L]
attn_weights = F.softmax(scores, dim=-1)
aligned_feat = torch.matmul(attn_weights, V)  # [B, H*W, C]
```

多模态那一套感觉没啥好多说的

#### 约束设计

这部分不太适合说太多，因为本身多任务的loss就是**单个任务+动态权重+任务互促**，科研的创新点基本都是在互促上面，这也是insignt所在，我很难把insight的内容讲清楚，至少我自己对此理解不多，知道上面的基本内容大概也就足够了

忘了还想说什么了，好像有些有意思的想法忘记写了，算了就这样吧
