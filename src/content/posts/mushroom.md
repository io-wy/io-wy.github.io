---
title: mushroom hunter
date: 2025-1-24
tags: [ML]
category: 记录
comments: true
draft: false
---

## 蘑菇猎人

大概就是边做边学，然后一堆官方文档看不下去

### 第一版

读取

从csv读取数据

```python
::codepen{import pandas as dp
data=pd.read_csv('data.csv')}
```

数据预处理

工作量最大的一集

在这一步我的目的是成功训练一个模型，并让他能跑，因此并没有做很完备的优化

使用pandas将特征的‘object’类型转为lightgbm可以识别的‘category’类型

```python
::codepen{train_data['class']=train_data['class'].astype('categroy').cat.codes}
```

得到所有的特征名称

```python
::codepen{train_data.columns}
```

整理数据成x_train

```python
::codepen{x_train=train_data[['class', 'cap-diameter', 'cap-shape', 'cap-surface', 'cap-color',
       'does-bruise-or-bleed', 'gill-attachment', 'gill-spacing', 'gill-color',
       'stem-height', 'stem-width', 'stem-root', 'stem-surface', 'stem-color',
       'veil-type', 'veil-color', 'has-ring', 'ring-type', 'spore-print-color',
       'habitat', 'season']]
y_train=train_data['class']
x_test=test_data[['id', 'cap-diameter', 'cap-shape', 'cap-surface', 'cap-color',
       'does-bruise-or-bleed', 'gill-attachment', 'gill-spacing', 'gill-color',
       'stem-height', 'stem-width', 'stem-root', 'stem-surface', 'stem-color',
       'veil-type', 'veil-color', 'has-ring', 'ring-type', 'spore-print-color',
       'habitat', 'season']]}
```

可能之后会使用sklearn进一步pro

算法调用

此处就只用了lightgbm

```python
::codepen{train_data_lgb=lgb.Dataset(x_train,
                           label=y_train,
                           categorical_feature=['class', 'cap-diameter', 'cap-shape', 'cap-surface', 'cap-color',
       'does-bruise-or-bleed', 'gill-attachment', 'gill-spacing', 'gill-color',
       'stem-height', 'stem-width', 'stem-root', 'stem-surface', 'stem-color',
       'veil-type', 'veil-color', 'has-ring', 'ring-type', 'spore-print-color',
       'habitat', 'season'],
                          free_raw_data=False)
test_data_lgb=lgb.Dataset(x_test,label=x_test, categorical_feature=['cap-diameter', 'cap-shape', 'cap-surface', 'cap-color',
       'does-bruise-or-bleed', 'gill-attachment', 'gill-spacing', 'gill-color',
       'stem-height', 'stem-width', 'stem-root', 'stem-surface', 'stem-color',
       'veil-type', 'veil-color', 'has-ring', 'ring-type', 'spore-print-color',
       'habitat', 'season'],
                         free_raw_data=False)}
```

```python
::codepen{model = lgb.train(
    params,
    train_data_lgb,
    num_boost_round=100  # 迭代次数
)}
```

写入CSV

```python
::codepen{predictions = model.predict(x_test)}
```

```python
::codepen{results_df = pd.DataFrame({
    'id': test_data['id'],
    'predicted_class': predictions  # 预测结果
})}
```

```python
::codepen{results_df.to_csv('predictions.csv', index=False)}
```

第一版的得分非常差只有0.24

### 第二版

这版是我网上参考别人的代码，一整套工作流程可能更加标准

```python
::codepen{Xy_train=pd.read_csv('train.csv')
X_test=pd.read_csv('test.csv')
Xy_all=pd.concat([Xy_train,X_test],axis=0)
cat_features = Xy_all.columns[Xy_all.dtypes=='object']}
```

```python
::codepen{from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder(
    dtype=np.int32,
    handle_unknown='use_encoded_value',
    unknown_value=-1,
    encoded_missing_value=-1,
).set_output(transform="pandas")
ordinal_encoder.fit_transform(Xy_all[cat_features])}
```

```python
::codepen{Xy_all[cat_features]=ordinal_encoder.fit_transform(Xy_all[cat_features])
Xy_train=Xy_all[Xy_all["class"]!=0]
X_train=Xy_train.drop(columns=["class"])
y_train=Xy_train["class"]}
```

到此数据预处理就完成了，把所有的object都换成int32并且把测试集中的class删去了

之后就是选用模型他选的还是lightgbm如果还要优化的话我可能才需要更进一步

```python
::codepen{M=lgb.LGBMRegressor()
M.fit(X_train,y_train)
y_pred=M.predict(X_test)
threshold = 0.5  # 可以根据需求调整
y_pred_class = ['p' if pred > threshold else 'e' for pred in y_pred]
results_df=pd.DataFrame({
    "id":X_test["id"],
    "class":y_pred_class
})
results_df.to_csv('second.csv', index=False)}
```

这版的得分进步很大达到了0.973

之后还有第三版...等我想写的时候再来吧。
