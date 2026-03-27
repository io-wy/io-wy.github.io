---
title: 'Kubernetes Quick Start'
description: 'K8s container orchestration and deployment guide'
pubDate: '2025-06-10'
heroImage: '/img/3.png'
tags:
  - devops
  - kubernetes
---

k8s是一个容器编排平台

## 容器的秘密

### Pod

* 最小调度单位
* 共享网络和存储
* Deployment 管理以及相关调度

### Deployment

* 副本管理
* 滚动更新
* 回滚能力
* 扩缩容能力

### Service && StatefulSet && ConfigMap/Secret

...

## 入门

### 容器化

随便写一个 dockerfile

```bash
docker build -t go-app:latest .
```

然后构建镜像+加载镜像到 kind 集群

```bash
kind load docker-image docker-app:latest --name gpu-cluster
```

## k8s 写部署文件

参考 [K8S：Yaml 文件详解及编写示例](https://blog.csdn.net/Katie_ff/article/details/132841454)

## 部署到 k8s

kubectl 是 k8s 的命令行工具

```bash
kubectl apply -f k8s/deployment.yml
```

可以参考 kubectl 的 handbook 或者相关文档查查常用 API

## client-go

```bash
go get k8s.io/client-go@v0.29.0
```

相关文档 [kubernetes/client-go - k8s 中文文档](https://www.k8src.cn/en/client-go.html)

## k8s Operator

Operator = k8s 资源 + 控制器，通过 CRD 拓展 k8s API，自动管理复杂应用

user（创建 myapp CR）-> API Server（处理 CR）-> Controller -> 创建/更新 k8s 资源

### Operator SDK

[operator-framework/operator-sdk - k8s 中文文档](https://www.k8src.cn/en/operator-sdk.html)

## Kubeflow

[文档 | Kubeflow 中文](https://kubeflow.cn/docs/)

## 深入

### k8s 网络机制

- [k8s 中的网络（较详细汇总）](https://www.cnblogs.com/jojoword/p/11214256.html)
- [Kubernetes 网络模型架构详解](https://cloud.tencent.com/developer/article/2588612)

### k8s 存储机制

- [Kubernetes 存储机制详解](https://www.oryoy.com/news/kubernetes-cun-chu-ji-zhi-xiang-jie-tan-suo-k8s-chi-jiu-hua-shu-ju-cun-chu-ce-lve-yu-shi-jian.html)
- [K8S 存储原理与 NFS 存储实践](https://zhuanlan.zhihu.com/p/714037674)

### k8s 调度机制

- [从零开始入门 K8s | 调度器的调度流程和算法介绍](https://zhuanlan.zhihu.com/p/101908480)