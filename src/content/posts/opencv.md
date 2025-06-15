---
title: opencv随便玩玩
date: 2025-06-14T10:54:27.000Z
tags: [可盐可甜]
category: 自用
comments: true
draft: false
---

## opencv

### 安装opencv

只有基本功能版

```python
pip install opencv-python
```

包含额外算法(eg:SIFT,SURF)

```python
pip install opencv-contrib-python
```

相关依赖

```python
pip install scikit-image
pip install numpy matplotlib
```

### 核心模块

cv2 : python接口主模块

imgproc : 图像处理 ，滤波，形态学，阈值，几何变换等

highgui : 界面与IO，窗口显示，鼠标交互，视频读写

feature2d : 特征检测与描述 ORB , BRIEF , SIFT等算法

calib3d : 三维重建

### IO基本操作

```python
import cv2
import numpy as np

img = cv2.imread(r'img.jpeg',cv2.IMREAD_COLOR)
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cv2.imwrite('gray.jpg',gray_img)
cv2.imshow('原图'，img)
cv2.waitkey(0)
cv2.destroyAllwindows()

h,w,_ = img.shape
px = img[100.200]
img[100:150,200:250] = (0,0,255) #把指定区域变成红色，BGR就是恶心
```

### 几何变换

#### 缩放与反转

```python
resized = cv2.resize(img,None,fx=0.5,fy=0.5,interpolation = cv2.INTER_LINEAR)
flipped = cv2.flip(img,1) # 0:垂直，1：水平，-1：水平+垂直
```

#### 旋转

```python
center = (W//2,H//2)
m=cv2.geRotationMatrix2D(center ,45 , 1.0) #按着center旋转45度
rotated = cv2.warpAffine(img,M,(w,h))
```

#### 透视与仿射变换

```python
pass
```

#### 图像增强与滤波

##### 平滑

```python
blur = cv2.GaussianBlur(img,(5,5),1.5)
median = cv2.medianBlur(img,5)
bilateral = cv2.bilateralFilter(img,9,75,75)
```

##### 锐化

```python
hist = cv2.calcHist([gray_img],[0],None,[256],[0,256])
eq = cv2.equalizeHist(gray_img)
```

##### 直方图与均衡化

```python
hist = cv2.calcHist([gray],[0],None,[256],[0,256])
eq = cv2.equalizeHist(gray)
```

### 函数详解

#### 图像 I/O 与基本函数详解

- **`cv2.imread(path, flags)`**
  - `path`：图像文件路径
  - `flags`：读取模式
    - `cv2.IMREAD_COLOR`（默认，忽略 alpha 通道，返回 BGR 彩色图）
    - `cv2.IMREAD_GRAYSCALE`（直接以灰度图形式读入）
    - `cv2.IMREAD_UNCHANGED`（包括 alpha 通道）
- **`cv2.imwrite(path, img, params=None)`**
  - `path`：保存路径
  - `img`：要保存的数组
  - `params`：可选压缩参数（如 JPEG 质量 `cv2.IMWRITE_JPEG_QUALITY`）
- **`cv2.imshow(winname, img)`** & **`cv2.waitKey(delay)`**
  - `winname`：窗口名
  - `delay`：等待时间（毫秒），`0` 表示无限期
- **`cv2.destroyAllWindows()`**：关闭所有窗口

#### 颜色空间转换

- **`cv2.cvtColor(src, code)`**
  - `src`：输入图像
  - `code`：转换方式
    - `cv2.COLOR_BGR2GRAY`：BGR → 灰度
    - `cv2.COLOR_BGRRGB`：BGR → RGB
    - `cv2.COLOR_RGB2HSV`：RGB → HSV

#### 几何变换

- **缩放：`cv2.resize(src, dsize, fx, fy, interpolation)`**

  - `dsize`：目标尺寸 `(width, height)`，可设 `None` 使用比例
  - `fx, fy`：缩放因子（水平方向、垂直方向），当 `dsize=None` 时生效
  - `interpolation`：插值方法
    - `cv2.INTER_NEAREST`：最邻近
    - `cv2.INTER_LINEAR`（默认）
    - `cv2.INTER_AREA`：对图像缩小时更佳
    - `cv2.INTER_CUBIC`：4×4 立方

- **翻转：`cv2.flip(src, flipCode)`**

  - `flipCode`：
    - `0`：垂直翻转
    - `1`：水平翻转
    - `-1`：水平+垂直

- **仿射变换：**

  ```python
  M = cv2.getAffineTransform(pts1, pts2)
  warped = cv2.warpAffine(src, M, (width, height))
  ```

  - `pts1`、`pts2`：各三个点坐标对应
  - `warpAffine` 参数与 `resize` 类似

- **透视变换：**

  ```python
  M = cv2.getPerspectiveTransform(src_points, dst_points)
  dst = cv2.warpPerspective(src, M, (w, h))
  ```

  - 需四个点对应

#### 滤波与锐化

- **`cv2.GaussianBlur(src, ksize, sigmaX, sigmaY=0)`**
  - `ksize`：卷积核尺寸，必须为奇数元组 `(k,k)`
  - `sigmaX, sigmaY`：X、Y 方向高斯核标准差；若 `sigmaY=0` 则同 `sigmaX`
- **`cv2.medianBlur(src, ksize)`**
  - `ksize`：核大小，必须为奇数
- **`cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)`**
  - `d`：领域直径
  - `sigmaColor`：颜色空间滤波标准差
  - `sigmaSpace`：坐标空间滤波标准差
- **自定义卷积：`cv2.filter2D(src, ddepth, kernel)`**
  - `ddepth`：目标图像深度，`-1` 表示与原图相同
  - `kernel`：自定义卷积核

#### 阈值与形态学

- **全局阈值：**

  ```python
  ret, dst = cv2.threshold(src, thresh, maxval, type)
  ```

  | 参数     | 含义                         |
  | -------- | ---------------------------- |
  | `thresh` | 阈值                         |
  | `maxval` | 大于阈值时赋予的最大值       |
  | `type`   | 阈值类型，如 `THRESH_BINARY` |

- **常用 `type`**

  - `cv2.THRESH_BINARY`：`dst = maxval if src>thresh else 0`
  - `cv2.THRESH_BINARY_INV`：反二值化
  - `cv2.THRESH_TRUNC`：大于阈值部分截断
  - `cv2.THRESH_TOZERO`：小于阈值部分归 0

- **自适应阈值：**

  ```python
  dst = cv2.adaptiveThreshold(src, maxval, adaptiveMethod, thresholdType, blockSize, C)
  ```

  - `adaptiveMethod`：`ADAPTIVE_THRESH_MEAN_C` 或 `ADAPTIVE_THRESH_GAUSSIAN_C`
  - `blockSize`：计算局部阈值的窗口大小（奇数）
  - `C`：从均值或加权和中减去常数

- **形态学操作：**

  ```python
  kernel = cv2.getStructuringElement(shape, ksize)
  dst = cv2.morphologyEx(src, op, kernel, iterations=1)
  ```

  - `shape`：`MORPH_RECT`（矩形）、`MORPH_ELLIPSE`、`MORPH_CROSS`
  - `op`：`MORPH_OPEN`、`MORPH_CLOSE`、`MORPH_GRADIENT` 等
  - `iterations`：迭代次数

#### 边缘检测与轮廓

- **Canny 边缘：**

  ```python
  edges = cv2.Canny(image, threshold1, threshold2, apertureSize=3, L2gradient=False)
  ```

  - `threshold1`、`threshold2`：低/高双阈值
  - `apertureSize`：Sobel 算子孔径，常用 3
  - `L2gradient`：是否使用更精确的 L2 范数

- **轮廓查找：**

  ```python
  contours, hierarchy = cv2.findContours(src, mode, method)
  ```

  - `mode`：`RETR_EXTERNAL`（仅外轮廓）、`RETR_TREE`（全层次）
  - `method`：`CHAIN_APPROX_SIMPLE`（压缩水平方向、垂直方向直线段）、`CHAIN_APPROX_NONE`

#### 特征检测与匹配

- **ORB:**

  ```python
  orb = cv2.ORB_create(nfeatures, scaleFactor, nlevels)
  kp, des = orb.detectAndCompute(image, mask)
  ```

  - `nfeatures`：检测的关键点数目上限
  - `scaleFactor`：每层图像金字塔缩放系数
  - `nlevels`：金字塔层数

- **BFMatcher:**

  ```python
  bf = cv2.BFMatcher(normType, crossCheck)
  matches = bf.match(des1, des2)
  ```

  - `normType`：`NORM_HAMMING`（ORB/BRIEF）或 `NORM_L2`（SIFT/SURF）
  - `crossCheck`：是否启用交叉匹配

#### 深度学习推理（DNN 模块）

- **加载模型：**

  ```python
  net = cv2.dnn.readNetFromONNX('model.onnx')
  ```

- **构造输入：**

  ```python
  blob = cv2.dnn.blobFromImage(image, scalefactor, size, mean, swapRB, crop)
  ```

  - `scalefactor`：缩放因子，常设 `1/255.0`
  - `size`：网络输入尺寸 `(w,h)`
  - `mean`：各通道均值，进行均值化
  - `swapRB`：是否交换 R、B 通道
  - `crop`：是否中心裁剪

- **推理：**

  ```python
  net.setInput(blob)
  output = net.forward([outputLayerNames])
  ```
