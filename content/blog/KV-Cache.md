---
title: 'KV Cache Explained'
description: 'KV Cache mechanism in LLM inference'
pubDate: '2025-12-20'
heroImage: '/img/8.png'
tags:
  - ai
  - llm
---

## Quick Start

KV Cache：让 LLM 在生成的时候不要忘了刚才说过的话再读一遍。

## LLM Foundation

**预填充（prefill）**：读所有输入 + 生成第一个 token 词，attention 计算全都是并行的（处理输入的所有 token）

**解码（decode）**：逐 token（词）生成（自回归生成，完全无法并行）

要提高吞吐量，要同时处理多个请求，但是每次 decode 一个新词都要重新计算之前的 token。

解码时需要记住之前生成的词，eg：为了生成第一百个词，要重新生成前九十九个词。

## 解码阶段复用计算：KV Cache 的出现 && 用处

复用解码时的计算：

前面的词怎么影响后面的词 -> 通过改变对后面的词的注意力（attention）

-> KV Cache：存储之前的词在注意力阶段产生的中间结果

把计算量换成了显存占用

**用处 1：单请求加速**，在显存常驻，被解码阶段反复使用直到请求完成

**用处 2：多请求复用**，前缀复用（Prefix Cache），把一个请求的前面的词的计算给另一个请求用

（eg：多轮对话，长文本问答 and RAG）

## KV Cache：空间换时间 && 显存优化

然而 KV Cache 太大了，1w 个 token 有好几个 G，多请求后解决手段：

1. **KV Cache 分页**（vLLM PageAttention）
2. **卸载**：把很久之前的 KV Cache 放到内存/硬盘（GPU -> CPU RAM or SSD）（DeepSpeed Inference）
3. **缩小**：eg 基于滑动窗口的注意力机制，动态的丢掉 KV Cache，推理时做量化

---

## Detailed

### Why KV

$$h_4 = attention(Q_4, [k_1, k_2, k_3, k_4], [v_1, v_2, v_3, v_4])$$

此处 *h4* 是 hidden state

### KV Cache 很占显存

$$size = layers \times tokens \times hidden \times 2$$

Layers = 32, hidden = 4096, token = 10k, dtype = fp16, KV = 5GB+

### Batch 推理困难

每个请求的长度，生成速度，结束时间不同，KV Cache 长度都不同，很难（并行）batch 处理多个请求，详解：continuous batching（连续批处理）

### KV Cache 显存碎片

类似 OS virtual memory，处理多个不同长度的 KV Cache（GPU allocator 碎片化）

### Prefix Cache 复用问题

hash(prompt) -> 找 KV cache -> 直接复用