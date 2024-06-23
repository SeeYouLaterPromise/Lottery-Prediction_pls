import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 数据预处理函数：分解中奖号码，添加日期特征
def preprocess_data(data):
    data['个位'] = data['中奖号码'] % 10
    data['十位'] = (data['中奖号码'] // 10) % 10
    data['百位'] = data['中奖号码'] // 100
    data['开奖日期'] = pd.to_datetime(data['开奖日期'])
    data['年'] = data['开奖日期'].dt.year
    data['月'] = data['开奖日期'].dt.month
    data['日'] = data['开奖日期'].dt.day
    return data


# 添加特征函数：总和、交互特征、滞后特征
def add_features(data):
    data['总和'] = data['个位'] + data['十位'] + data['百位']

    data['百十和'] = data['百位'] + data['十位']
    data['百个和'] = data['百位'] + data['个位']
    data['十个和'] = data['十位'] + data['个位']
    data['百十乘'] = data['百位'] * data['十位']
    data['百个乘'] = data['百位'] * data['个位']
    data['十个乘'] = data['十位'] * data['个位']

    data['百位_sqrt'] = np.sqrt(data['百位'])
    data['十位_sqrt'] = np.sqrt(data['十位'])
    data['个位_sqrt'] = np.sqrt(data['个位'])

    data['百十_sqrt和'] = data['百位_sqrt'] + data['十位_sqrt']
    data['百个_sqrt和'] = data['百位_sqrt'] + data['个位_sqrt']
    data['十个_sqrt和'] = data['十位_sqrt'] + data['个位_sqrt']

    # 添加滞后特征
    for lag in [1, 2, 3]:
        data[f'个位_lag{lag}'] = data['个位'].shift(lag)
        data[f'十位_lag{lag}'] = data['十位'].shift(lag)
        data[f'百位_lag{lag}'] = data['百位'].shift(lag)

    data.fillna(method='bfill', inplace=True)
    return data


# 加载数据
train_data_path = 'train.csv'
test_data_path = 'test.csv'
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# 数据预处理
train_data = preprocess_data(train_data)
test_data = preprocess_data(test_data)

# 特征工程
train_data = add_features(train_data)
test_data = add_features(test_data)

# 准备特征和目标变量
features = [col for col in train_data.columns if col not in ['期号', '中奖号码', '开奖日期', '个位', '十位', '百位']]
X_train = train_data[features]
y_train = train_data[['个位', '十位', '百位']]
X_test = test_data[features]
y_test = test_data[['个位', '十位', '百位']]

# 训练随机森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 计算准确率
accuracy = {
    '个位': accuracy_score(y_test['个位'], y_pred[:, 0]),
    '十位': accuracy_score(y_test['十位'], y_pred[:, 1]),
    '百位': accuracy_score(y_test['百位'], y_pred[:, 2])
}

print("Model accuracy:", accuracy)



