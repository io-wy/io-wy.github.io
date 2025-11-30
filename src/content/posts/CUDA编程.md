---
title: CUDA编程
date: 2025-03-24T10:54:27.000Z
tags: [ai]
category: 自用
comments: true
draft: false
---

CUDA是由NVIDIA开发的GPU编程模型，提供了GPU编程的简易接口，基于CUDA编程可以构建基于GPU计算的应用程序，主要支持C++，python的语言接口，值得一提的是，CUDA编程是一个异构模型，需要GPU和CPU协同工作，在CUDA中 **host**指代CPU及其内存，而用**device**指代GPU及其内存，更重要的是，我们需要host和device之间进行通信，也就是我们非常想解决的**数据拷贝**问题

### CUDA编程模型基础

典型的CUDA程序执行流程如下

- 分配host内存，并进行数据初始化
- 分配device内存，并从host把数据拷贝到device上
- 调用CUDA的和函数在device上完成指定的运算
- 把device的运算结果考虑到host上
- 释放device和host上分配的内存

最重要的过程就是调用CUDA的核函数来执行并行计算，kernel是CUDA中一个重要的概念，在device上线程中并行执行的函数，核函数用**global**声明，从代码层面上如下所示
