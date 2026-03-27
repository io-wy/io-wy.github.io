---
title: 'Neural Network'
description: 'Understanding neural networks from CNN to Transformer to Diffusion'
pubDate: '2025-03-27'
heroImage: '/img/1.png'
tags:
  - ai
  - deep-learning
---

# Neural Network



最近研究了扩散模型，对神经网络又有了比较新的理解，于是回过头去把思路理了一遍，结合现有的工作，做了一点自己的构建



## 局部敏感 && 概念对齐

#### gradient

最早的一类方法把解释定义为输出对输入的局部敏感性。以最为trivial的分来举例子，对于类别 c 的 score `s_c(x)`，saliency map 可以写作

$$ A_i(x) = \frac{\partial S_c(x)}{\partial x_i} $$

非常的简单直接，而且可以在不改变模型的前提下计算；然而，它非常严格依赖局部的线性近似，所以对噪声，梯度饱和，重参数化都很敏感。有相关Integrated Gradients 把局部导数沿着路径积分，以降低单点梯度的不稳定性

#### feature visualization

目前更经常做的还是视觉相关内容，因此一个更加自然的问题可以想到：某个中间单元究竟偏好怎么样的模型输入输出呢

一定程度上可以表述为优化问题

$$ x^\star = \arg \max_x a_j(x) - \lambda R(x)$$

$a_j(x)$ 是第 j 个单元/通道激活，$R(x)$是图像先验/正则；这类方法可以让单元偏好具象化；但它仍然是描述性的，并不能保证因果作用



平时也做过类似的实验，进行可视化的方法来分析idea的合理性，这里采用mnist的方法做一下代替（仅展示部分code）

*  `t-SNE`降维可视化mnist每个数字，达到了抱团的效果
* 可视化卷积特征图，得到中间层的高关联性

```python
tsne = TSNE(n_components=2, random_state=0, init='pca')
features_2d = tsne.fit_transform(features[:2000])// t-SNE

def visualize_conv1_features(model, image):
    features = get_conv1_features(model, image)
    num_filters = features.size(0)
    fig, axes = plt.subplots(4, 8, figsize=(15, 8))
    axes = axes.ravel()
```



实际上对此我们可以简单的归结为，在MLP/CNN的时代，建立的解释对象大多是 **输出归因 神经元偏好 **，在视觉中效果和卷积提供的空间局部性和层级组合性，以及稳定的语义先验；对于解释性来说，无法离开这些归纳偏置，对于实际实验来说，这确实是非常优雅的验证方式



## 机制 



致敬一下 attention

$$ \mathrm{Attn}(Q,K,V) = \mathrm{softmax}(\frac{QK^\top}{\sqrt{d_k}})V$$

直觉上来，大家都非常的清楚attention的权重就是”模型把注意力放在哪里“，但高attention权重和更严格的重要性度量不总一致（Jain & Wallace 指出），甚至于，在不改变预测的情况，可以构造完全不同的attention分布，attention对于充分解释还是很遥远，不过作为信息路由的观测窗口还是相对优雅



#### 模块交互

如果把 transformer 的残差流视为计算的主载体

$$h_{\ell + 1} = h_\ell + \sum_{m \in \mathcal{H}_\ell} \mathrm{Head}_m(h_\ell) + \mathrm{MLP}_\ell(h_\ell).$$

在这个视角下，attention head 是路由器，MLP 常被解释为某种 key-value memory [9]，而多个头与 MLP 可以组合成完成特定子任务的电路 `circuit`（？），这似乎是有更意思的机制层解释，它承认真正的计算分布在模块交互，而非单点激活



从可视化转向“机制科学”，似乎实验验证的过程更加困难了，不过似乎更接近真相了



## Diffusion



进来研究的diffusion，解释问题带入动态视角，前向噪声如下

$$q(x_t \mid x_0) = \mathcal{N}\!\left(\sqrt{\bar{\alpha}_t}\, x_0,\; (1 - \bar{\alpha}_t)I\right),$$

训练目标，大多是预测网络噪声

$$\mathcal{L}_{\mathrm{simple}} =
\mathbb{E}_{x_0,\epsilon,t}
\left[\lVert \epsilon - \epsilon_\theta(x_t, t, c) \rVert_2^2\right].$$

这意味着模型的行为不是一个简单的静态传播决定的，而做成了多步去噪轨迹共同决定；因此解释就必须有两个问题

* 语义何时进入图像
* 通过什么模块进入图像

因此我简单看了`DAAM`，文本到图像生成中，使用`cross-attention`对词语与图像区域的对应关系，做了时间与层上的聚合；似乎可以做一个归因图

$$M_w = \sum_{t,\ell h} \mathrm{Up}(\alpha^{(t,\ell ,h)}_w)$$

不再把解释限定为最终像素的敏感性，而是把条件控制看作一条跨时间传播的因果链

进一步的，近年的research开始把diffusion理解为前空间中逐步形成可分解，可组合结构的过程，而不只是”反复去噪“，这意味着diffusion的解释不能只看单个时间部分热土，而需要研究整条轨迹上的统计规律：不同噪声区间承载何种语义，哪些模块负责全局布局，哪些模块负责局部纹理，以及文本条件是在早期还是晚期真正生效。



## Final



于此从CNN到transformer，再到diffusion，算是简单的了解了一下很多人对神经网络的认知，可解释性算不上，但用来指导idea判断验证，实验的做法还是不错的，这块也是我个人觉得比较有意思的分析，有用不一定，但感觉还是挺有趣的

