---
title: 'Multi-Task Learning (MTL)'
description: 'Multi-task learning network architecture and loss design guide'
pubDate: '2025-03-24'
heroImage: '/img/6.png'
tags:
  - ai
  - machine-learning
---

Multi-task Learning 同时学习不同领域的任务，通过特定领域的信息提高泛化能力。

一个好的多任务学习工作，必然会在任务间的联系下很大的功夫，这也就是创新点所在。相对于单任务学习，多任务学习的创新点在于网络架构和 loss 的设计。

对于很多提到的多任务学习的特点，**关联任务**，**正则化**（防止过拟合），其实我个人觉得没有太大的必要多说，前者作为科研创新显而易见，后者在实验的过程中水到渠成。**共享架构** 节约内存加快计算也是不违反直觉的。

## 网络架构

多任务学习意味着要多个网络并行或者拼接，就会存在一些问题（尤其是 size 上的）。

### 硬参数共享

这是最原始的 MTL 方法，多个任务共享底层网络，然后顶层多个任务头分别输出。好处很明显，就是计算高效，还能避免过拟合，不过由实验得知不同任务的效果不一样，有的时候性能可能会下降。

```python
class Eg(nn.Module):
    def __init__(self):
        self.shared_encoder = ResNet50()
        self.task1 = nn.Linear(2048, ...)
        self.task2 = nn.Conv2d(2048, ...)

    def forward(self, x):
        features = self.shared_encoder(x)
        out1 = self.task1(features)
        out2 = self.task2(features)
        return out1, out2
```

### 软参数共享

每个任务都有独立的网络，通过约束（loss）联系在一起。优点是灵活，但参数量太大了。

```python
class SemanticMap:
    pass

class ObjectDetection:
    pass

def loss_semantic():
    pass

def loss_detection():
    pass

loss = w1 * loss_semantic + w2 * loss_detection
```

主要还是 loss 之间的约束，如果想要 w1, w2 可学习的话，可以用 random + 归一化，让模型自己去学。

### Personal Trick

因为网络比较复杂，把 CV/NLP 模型对接了，出现了非常多的特征不对齐的 issue。

#### 自适应池化层

```python
self.pool = nn.AdaptiveAvgPool2d((4, 4))  # 平均池化，把 Avg 改成 Max 就是最大池化
aligned_feat = self.pool(feat)  # 输入 [B, C, H, W] → 输出 [B, C, 4, 4]
```

#### 特征变换层

```python
self.channel_trans = nn.Conv2d(256, 512, kernel_size=1)
```

256 通道变成 512 通道，用于通道对齐。

#### 拼接问题

```python
self.fusion = nn.Sequential(
    nn.Conv2d(256 + 128, 256, kernel_size=1),  # 拼接后降维
    nn.ReLU()
)
fused_feat = self.fusion(torch.cat([feat1, feat2], dim=1))
```

直接拼接后降维。

#### 注意力机制

自注意力：

```python
self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)
aligned_feat, _ = self.attention(feat, feat, feat)
```

Cross Attention：

```python
# 计算图像特征（Q）和文本特征（K/V）的对齐
scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, H*W, L]
attn_weights = F.softmax(scores, dim=-1)
aligned_feat = torch.matmul(attn_weights, V)  # [B, H*W, C]
```

## 约束设计

多任务的 loss 就是**单个任务 + 动态权重 + 任务互促**，科研的创新点基本都是在互促上面，这是 insight 所在。