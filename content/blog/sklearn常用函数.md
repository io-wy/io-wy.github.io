---
title: 'sklearn Common Functions'
description: 'sklearn data preprocessing and feature engineering functions'
pubDate: '2025-01-20'
heroImage: '/img/4.png'
tags:
  - ai
  - sklearn
---

个人用 sklearn 主要就是用来做数据处理，当然有的时候会和具有 sklearn API 的模型混用。

## 1 数据清洗

### 缺失值处理

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_filled = imputer.fit_transform(X)
```

用均值（mean）填充缺失值，也可以用中位数/众数...

### 异常值处理

略

## 2 特征变换

特征变换主要分为：
- 数值型特征变换
- 分类特征变换（编码分类变量）
- 文本特征变换
- 时间特征变换（遇到再说）

### 归一化（Normalization）

将数据缩放到指定范围（默认【0，1】）

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

### 标准化（Standardization）

将数据转换成均值为 0，方差为 1 的标准正态分布

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 鲁棒缩放（Robust Scaling）

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

### 对数变换（Log Transformation）

非 sklearn

```python
import numpy as np
data['feature'] = np.log(data['feature'])
```

## 3 编码分类变量

将 object 换个类型（str, int, float, category...）

### 整数编码（Ordinal Encoding）

将**有序分类变量**转换为**整数**编码

```python
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)
```

实例：

```python
import numpy as np
ordinal_encoder = OrdinalEncoder(
    dtype=np.int32,
    handle_unknown='use_encoded_value',
    unknown_value=-1,
    encoded_missing_value=-1
).set_output(transform="pandas")
X_encoded = ordinal_encoder.fit_transform(X[cat_features])
```

输出为 pandas 用到了 `set_output` 的 API。

### 独热编码（One-Hot Encoding）

**无序**分类变量转换为**二进制**

```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)
```

参数都在官网上需要再临时一个个实现。详见 [scikit-learn 官方文档](https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.OneHotEncoder.html)

### 标签编码（Label Encoding）

将**目标变量（标签）**转换为整数编码

```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
```

官方文档还有一个 Target Encoder没见过就不写了。

## 关于 Pipeline

Pipeline 是一个简化代码的工具，能防止数据泄露，方便优化。

由多个 step 组成，每个 step 是一个元组，包含"步骤名称（string）"（标识每个步骤）和"转换器/估计器"（数据处理和建模的对象）。

常见的步骤就是：
- 数据预处理：数据清洗，标准化，编码
- 模型训练：分类器，回归器

假设我们有一个数据集，并且需要对数据进行标准化后训练一个支持向量机（SVM）分类器：

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

大概完成到数据预处理的位置就好了，上面的三个步骤也是我自己的工作流。之后模型训练应该会重新开一篇文章来写。