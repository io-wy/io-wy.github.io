---
title: sklearn 常用函数
date: 2025-01-20T10:54:27.000Z
tags: [ai]
category: data
comments: true
draft: false
---

## sklearn常用函数

（自用）--from 学技术的数学飞舞一枚~！

个人用sklearn主要就是用来做数据处理，当然有的时候会和具有sklearn API 的模型混用

### 1 数据清洗

缺失值处理

```python
from sklearn.imput import SimpleImputer
imputer=SimpleImputer(strategy='mean')
X_filled=imputer.fit_transform(X)
```

用均值(mean)填充缺失值，也可以用中位数/众数...

异常值处理

### 2 特征变换

特征变换主要分为数值型特征变换（这里的特征变换主要就是指这个），分类特征变换（实际上编码分类变量也属于特征变换），文本特征变换，时间特征变换（这些遇到再说）

#### 归一化（Normalization）

将数据缩放到指定范围（默认【0，1】）

```python
from sklearn.preprocessing import MinMaxScale
scaler=MinMaxScaler()
X_scaler=scaler.fit_transform(X)
```

#### 标准化（Standardization）

将数据转换成均值为0，方差为1的标准正态分布

```python
from sklearn.preprocessing import StandardScaler
scaler=Standardscaler
X_scaler=scaler.fit_transform(x)
```

#### 鲁棒缩放 (Robust Scaling)

```python
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
X_scaler=scaler.fit_transform(X)
```

#### 标签二值化(Label Binarization)

```python
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
y_bin = lb.fit_transform(y)
```

非sklearn

#### 对数变换（Log Transformation）

```python
import numpy as np
data['feature'] = np.log(data['feature'])
```

### 3 编码分类变量

将object换个类型（str,int,float,category...）

#### 整数编码（Ordinal Encoding）将**有序分类变量**转换为**整数**编码

```python
from sklearn.preprocessing import OrdinalEncoder
encoder=OrdinalEncoder()
X_encoded=encoder.fit_transform(X)
```

实例

```python
ordinal_encoder=OrdinalEncoder(
	dtype=np.int32,
	handle_unknown='use_encoded_value',
	unknown_value=-1,
	encoded_missing_value=-1
).set_output(transform="pandas")
ordinal_encoder.fit_transfor(Xy_all[cat_features])
```

一些参数的设置，参见官网

输出为pandas用到了一个set_output的API

#### 独热编码（One_Hot Encoding）**无序**分类变量转换为**二进制**

https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder

参数都在官网上需要再临时一个个实现

```python
from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder()
X_encoded=encoder.fit_transform(X)
```

#### 标签编码（Label Encoding）将**目标变量（标签）**转换为整数编码

```python
 from sklearn.preprocessing import LabelEncoder
  encoder = LabelEncoder()
  y_encoded = encoder.fit_transform(y)
```

官方文档还有一个Target Encoder没见过就不写了

大概完成到数据预处理的位置就好了，上面的三个步骤也是我自己的工作流，可能还有一些可视化的和pandas的处理并没有在这里展现，不过那并不重要

之后模型训练我应该会重新开一篇文章来写

### 关于Pipeline

这个好像就是一个简化代码的工具，能防止数据泄露，方便优化

由多个step组成，每个step是一个元组，包含“步骤名称（string）”（标识每个步骤）和“转换器/估计器”（数据处理和建模的对象）

常见的步骤就是

数据预处理：数据清洗，标准化，编码

模型训练：分类器，回归器

假设我们有一个数据集，并且需要对数据进行标准化后训练一个支持向量机（SVM）分类器。我们可以将标准化和模型训练步骤组合成一个管道（Pipeline）

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# 加载数据
data = load_iris()
X, y = data.data, data.target

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建 Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # 数据标准化
    ('svc', SVC())  # 支持向量机分类器
])

# 训练模型
pipeline.fit(X_train, y_train)

# 预测结果
y_pred = pipeline.predict(X_test)

# 打印模型精度
print(f"Model accuracy: {pipeline.score(X_test, y_test)}")
```

感觉这个就是做一些简单题的时候耦合一下用的....
