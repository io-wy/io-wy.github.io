---
title: Docker入门
date: 2025-05-19T10:54:27.000Z
tags: [后端]
category: 自用
comments: true
draft: false
---

当课题组要复现一个项目时，学妹不会配环境，于是，我用docker直接打包环境发给她，~~成功俘获学妹芳心~~发现学妹连docker都不会

首先安装docker，有手就行

### **编写dockerfile**

```dockerfile
# 使用官方 Py 基础镜像
FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 复制项目代码到容器中
COPY . /app

# 安装系统级依赖（如 CUDA 或其他库）
RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
RUN pip install --upgrade pip
RUN pip install -r requirements.txt  # 假设你已经在项目中包含了 requirements.txt 文件

# 设定默认命令，启动项目
CMD ["python", "main.py"]  # 请替换为你项目的启动命令
```

### **创建 requirements.txt**

`requirements.txt` 文件包含了你项目的所有 Python 依赖。可以通过以下命令生成该文件：

```bash
pip freeze > requirements.txt
```

确保 `requirements.txt` 文件包含了所有项目所需的库（如 `torch`, `tensorflow`, `numpy`, `pandas`, `matplotlib` 等）。

### **构建 Docker 镜像**

在项目目录中，打开终端并执行以下命令来构建 Docker 镜像：

```bash
docker build -t your_project_name .
```

这将使用 `Dockerfile` 和 `requirements.txt` 来构建镜像，`-t` 后面的 `your_project_name` 是你为镜像命名的标签。构建过程可能需要一些时间，具体取决于项目的大小和依赖。

### **运行 Docker 容器**

一旦镜像构建完成，你可以通过以下命令来运行容器：

```bash
docker run --gpus all -it your_project_name
```

- `--gpus all`：如果你希望容器使用 GPU 来加速深度学习任务，可以添加这个参数。学妹的电脑如果没有 GPU，可以去掉这个参数。
- `-it`：允许你进入交互式终端。

### **分享镜像**

如果学妹的机器上无法直接构建镜像，你可以将构建好的 Docker 镜像打包并发送给她。

#### 将镜像保存为文件：

```bash
docker save -o your_project_name.tar your_project_name
```

这将会把镜像保存为一个 `.tar` 文件。你可以通过邮件或其他方式将 `your_project_name.tar` 文件发送给学妹。

#### 学妹导入镜像：

学妹可以通过以下命令导入镜像：

```bash
docker load -i your_project_name.tar
```

然后学妹就可以使用之前的 `docker run` 命令运行容器了。

### **其他**

- **Jupyter Notebook**：如果你希望学妹使用 Jupyter Notebook 可以在 Dockerfile 中安装并配置好 Jupyter：

  ```dockerfile
  RUN pip install jupyter
  CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
  ```

  这样，她可以通过浏览器访问 `http://localhost:8888` 来使用 Notebook。

- **共享数据文件**：如果项目中有数据文件，可以通过 Docker 的 **volume** 功能来挂载本地文件夹到容器中，从而共享文件数据。

  ```bash
  run -v /path/to/data:/app/data your_project_name
  ```

这样，学妹就可以直接运行容器，而不需要配置任何环境。如果她不熟悉命令行操作，你也可以考虑将 Docker 容器与图形化的 Docker Desktop 配合使用，这样会更容易操作。

然后学妹用完你的环境完美的复现了项目和帅气多金的少爷跑咯。。。。。。。
