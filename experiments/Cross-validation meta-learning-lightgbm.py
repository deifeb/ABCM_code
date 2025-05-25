import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GridSearchCV
import time  # 导入 time 模块
start_time = time.time()
# 数据加载
X_features = pd.read_excel(r"C:\Users\123\Desktop\0.96F9_encoder_feature组合.xlsx", index_col=0)
model_error = pd.read_excel(r"C:\Users\123\Desktop\av_error_xin2组合.xlsx")

# 合并特征
features = X_features.merge(model_error, on='label', how='left')

# 编码目标变量
label_encoder = LabelEncoder()
features['y'] = features['y'].fillna('unknown')  # 防止缺失值问题
y = label_encoder.fit_transform(features["y"])
labels = np.array(y)

# 特征选择
feature = features.drop(columns=['y', "SBJ", "ETS", "DeepAR", "DeepRenewal Flat", "DeepRenewal Exact","DeepRenewal Hybrid", "ARIMA"])
# scaler = StandardScaler()
# feature = scaler.fit_transform(feature)

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(feature, labels, test_size=0.2, random_state=42)

param_grid = {'n_estimators': [50, 100, 200],
              'max_depth': [None, 10, 20],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4],
              'bootstrap': [True, False]}
meta_learner = GridSearchCV(RandomForestClassifier(random_state=42, class_weight="balanced"), param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)

meta_learner.fit(X_train, y_train)

y_pred = meta_learner.predict(X_test)

print(f"Best n_estimators: {meta_learner.best_params_['n_estimators']}")
print(f"Best Accuracy: {meta_learner.best_score_}")

# # 交叉验证
scores = cross_val_score(meta_learner, X_train, y_train, cv=5)
print(f'Cross-Validation Accuracy: {np.mean(scores)} ± {np.std(scores)}')
# 记录结束时间
end_time = time.time()
# 计算运行总时间
total_time = end_time - start_time
print(f"代码从运行至结束的总时间: {total_time:.2f} 秒")