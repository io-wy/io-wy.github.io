---
title: mushroom hunter
date: 2024-02-02T00:00:00Z
tags: [ML]
category: 记录
comments: true
draft: false
---

## 蘑菇猎人

大概就是边做边学

lightgbm有难以想象的高性能.....

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
::codepen{
M=lgb.LGBMRegressor()
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

### 第三版

这版里面就有进行调参+模型的评估

然后record当然可以当成log来用啦啦啦

![image-20250127002228596](D:\0 program\blog\io-wy.github.io\src\content\posts\image-20250127002228596.png)

初始参数

来来来，让我去找找lightgbm还有哪些参数

- **`n_estimators`:** 树的数量，增加可以提高模型性能，但也会增加计算成本。
- **`max_depth`:** 树的最大深度，控制模型的复杂度。
- **`learning_rate`:** 学习率，较小的学习率通常需要更多的树。
- **`num_leaves`:** 叶子节点数，控制树的复杂度。
- **`subsample`:** 样本采样比例，防止过拟合。
- **`colsample_bytree`:** 特征采样比例，防止过拟合。
- 上面这些是常用参数

![image-20250127003840506](D:\0 program\blog\io-wy.github.io\src\content\posts\image-20250127003840506.png)

这次修改：learning_rate=0.08, n_estimators=130,

然后我觉得如果再继续调参也没啥意义了（准确率都爆了....），所以第三版就这样吧...

```python
::codepen{
import pandas as pd
import numpy as np
import sklearn
import lightgbm as lgb}
```

读取数据，预处理

```python
::codepen{
Xy_train=pd.read_csv(r"D:\0 program\mushroom hunter first attempt\some data\train.csv")#数据预处理
X_test=pd.read_csv(r"D:\0 program\mushroom hunter first attempt\some data\test.csv")
Xy_all=pd.concat([Xy_train,X_test],axis=0)
cat_features = Xy_all.columns[Xy_all.dtypes=='object']
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder(
    dtype=np.int32,
    handle_unknown='use_encoded_value',
    unknown_value=-1,
    encoded_missing_value=-1,
).set_output(transform="pandas")#set_output的API接口
ordinal_encoder.fit_transform(Xy_all[cat_features])
Xy_all[cat_features]=ordinal_encoder.fit_transform(Xy_all[cat_features])}
```

拆分数据

```python
::codepen{
X_test=Xy_all[Xy_all["class"]==-1].drop(columns=["class"])#测试数据
Xy_train=Xy_all[Xy_all["class"]!=-1]
X_train=Xy_train.drop(columns=["class"])#特征
y_train=Xy_train["class"]#目标
from sklearn.model_selection import train_test_split#调参预处理
X1_train,X1_test,y1_train,y1_test=train_test_split(X_train,y_train,test_size=0.25,random_state=42,stratify=y_train)#X1_train是训练数据，X1_test是验证数据，X_test是测试数据
}
```

模型训练---验证集

```python
::codepen{
Model=lgb.LGBMClassifier(
    boosting_type='gbdt',
    num_leaves=31,
    max_depth=-1,
    learning_rate=0.08,
    n_estimators=130,
    subsample_for_bin=200000,
    objective=None,
    class_weight=None,
    min_split_gain=0.0,
    min_child_weight=0.001,
    min_child_samples=20,
    subsample=1.0,
    subsample_freq=0,
    colsample_bytree=1.0,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=None,
    n_jobs=None,
    importance_type='split',)#调参，sklearn API的lightgbm的所有参数
Model.fit(X1_train,y1_train)
y1_pred=Model.predict(X1_test)
threshold = 0.5
y_pred_class = ['p' if pred > threshold else 'e' for pred in y1_pred]
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
print("分类报告")#大模型教我评估模型
print(classification_report(y1_test, y1_pred, target_names=['e', 'p']))

print("\n混淆矩阵")
print(confusion_matrix(y1_test, y1_pred))

print("\n准确率")
print(accuracy_score(y1_test, y1_pred))

print("\nROC AUC 分数")
print(roc_auc_score(y1_test, y1_pred))#默认参数的分数就非常好了！
}
```

模型预测---测试集

```python
::codepen{
Model.fit(X_train,y_train)
y_pred=Model.predict(X_test)
threshold = 0.5
y_pred_class = ['p' if pred > threshold else 'e' for pred in y_pred]
results_df=pd.DataFrame({
    "id":X_test["id"],
    "class":y_pred_class
})
results_df.to_csv('lalala.csv', index=False)
}
```

所以第三版也就这样了0.978

事实上到现在我也不知道具体怎么调参，用随即搜索和网格搜索，貌似根本跑不完....反正没啥耐心

可能要学点分布式系统？

然后第四版可能要结合同学的代码进行K折交叉验证了？

预处理/模型还能优化吗？

### 第四版

emmm出乎意料的出现了提升有限的第四版

预处理照做

```python
::codepen{
Xy_train=pd.read_csv(r"D:\0 program\mushroom hunter first attempt\some data\train.csv")#数据预处理
X_test=pd.read_csv(r"D:\0 program\mushroom hunter first attempt\some data\test.csv")
Xy_all=pd.concat([Xy_train,X_test],axis=0)
cat_features = Xy_all.columns[Xy_all.dtypes=='object']
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder(
    dtype=np.int32,
    handle_unknown='use_encoded_value',
    unknown_value=-1,
    encoded_missing_value=-1,
).set_output(transform="pandas")#set_output的API接口
ordinal_encoder.fit_transform(Xy_all[cat_features])
Xy_all[cat_features]=ordinal_encoder.fit_transform(Xy_all[cat_features])
}
```

数据分割

```python
::codepen{
X_test=Xy_all[Xy_all["class"]==-1].drop(columns=["class"])#测试数据
Xy_train=Xy_all[Xy_all["class"]!=-1]
X_train=Xy_train.drop(columns=["class"])#特征
y_train=Xy_train["class"]#目标
from sklearn.model_selection import train_test_split#调参预处理
X1_train,X1_test,y1_train,y1_test=train_test_split(X_train,y_train,test_size=0.25,random_state=42,stratify=y_train)#X1_train是训练数据，X1_test是验证数据，X_test是测试数据
}
```

lightgbm提取特征

```python
::codepen{
def extract_features_with_lightgbm(X_train, y_train, X_val, X_test):
    # 初始化 LightGBM 模型
    lgbm = lgb.LGBMClassifier(
        boosting_type='gbdt',
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.08,
        n_estimators=130,
        subsample_for_bin=200000,
        objective=None,
        class_weight=None,
        min_split_gain=0.0,
        min_child_weight=0.001,
        min_child_samples=20,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=None,
        n_jobs=None,
        importance_type='split',)

    # 训练 LightGBM 模型
    lgbm.fit(X_train, y_train)

    # 提取叶子节点特征
    X_train_leaves = lgbm.predict(X_train, pred_leaf=True)
    X_val_leaves = lgbm.predict(X_val, pred_leaf=True)
    X_test_leaves = lgbm.predict(X_test, pred_leaf=True)

    return X_train_leaves, X_val_leaves, X_test_leaves
X_train_leaves, X_val_leaves, X_test_leaves = extract_features_with_lightgbm(X1_train, y1_train, X1_test, X_test)
}
```

化成张量

```python
::codepen{
# 标准化叶子节点特征
scaler = StandardScaler()
X_train_leaves = scaler.fit_transform(X_train_leaves)
X_val_leaves = scaler.transform(X_val_leaves)
X_test_leaves = scaler.transform(X_test_leaves)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train_leaves, dtype=torch.float32)
y_train_tensor = torch.tensor(y1_train.values, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val_leaves, dtype=torch.float32)
y_val_tensor = torch.tensor(y1_test.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_leaves, dtype=torch.float32)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
}
```

mlp模型训练

```python
::codepen{
#模型
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(HousePriceModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

# 初始化模型
input_size = X_train_leaves.shape[1]
hidden_size = 64
output_size = 1
model = HousePriceModel(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # 验证集上的损失
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            val_loss += criterion(outputs, batch_y).item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
}
```

预测任务

```python
::codepen{
# 预测
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).numpy()

# 保存预测结果
lala = pd.DataFrame({'id': X_test['id'], 'class': predictions.flatten()})
threshold = 0.5
lala['class'] = ['p' if pred > threshold else 'e' for pred in lala['class']]
lala.to_csv('lala.csv', index=False)
print("预测结果已保存到 lala.csv")
}
```

这里到了0.982其实这明显树模型更好，mlp的意义确实也不是很大，理论上的进步还是有限的，所以进步也是很小...或许还有第五版？或者我不感兴趣了也有可能
