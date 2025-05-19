from collections import Counter
import numpy as np
import pandas as pd
import tensorflow as tf
from kscorer.kscorer import KScorer
from scipy.stats import variation
import statsmodels.api as sm
import antropy as ant
import pickle
import os



os.environ["OMP_NUM_THREADS"] = '3'


def save_to_pickle(data, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)
    print(f"数据已保存到 {filename} 文件中。")


def load_from_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    print(f"数据已从 {filename} 文件中加载。")
    return data


def compute_features(time_series):
    features = {}

    # F1: 平均需求间隔 (ADI)
    demand_indices = np.where(time_series > 0)[0]
    if len(demand_indices) > 1:
        inter_demand_intervals = np.diff(demand_indices)
        features['F1'] = np.mean(inter_demand_intervals)
    else:
        features['F1'] = np.nan

    # F2: 方差系数平方 (CV^2)
    non_zero_demand = time_series[time_series > 0]
    if len(non_zero_demand) > 0:
        features['F2'] = variation(non_zero_demand) ** 2
    else:
        features['F2'] = np.nan

    # F3: 近似熵
    features['F3'] = ant.app_entropy(time_series, order=2)

    # F4: 零值百分比
    features['F4'] = np.sum(time_series == 0) / len(time_series)

    # F5: 超出 [mean - std, mean + std] 范围的值的百分比
    mean_y = np.mean(time_series)
    std_y = np.std(time_series)
    features['F5'] = np.sum((time_series < mean_y - std_y) | (time_series > mean_y + std_y)) / len(time_series)

    # F6: 线性最小二乘回归系数
    chunk_size = 12  # 对于月度数据
    chunks = [time_series[i:i + chunk_size] for i in range(0, len(time_series), chunk_size)]
    variances = [np.var(chunk) for chunk in chunks if len(chunk) == chunk_size]
    if len(variances) > 1:
        x = np.arange(len(variances))
        X = sm.add_constant(x)
        model = sm.OLS(variances, X).fit()
        features['F6'] = model.params[1]
    else:
        features['F6'] = np.nan

    # F7: 连续变化的平均绝对值
    consecutive_changes = np.diff(time_series)
    features['F7'] = np.mean(np.abs(consecutive_changes))

    # F8: 最后一个块的平方和占整个序列的比例
    k = 4
    chunk_length = len(time_series) // k
    last_chunk = time_series[-chunk_length:]
    features['F8'] = np.sum(last_chunk ** 2) / np.sum(time_series ** 2)

    # F9: 序列末尾连续零值的百分比
    consecutive_zero_at_end = 0
    for value in reversed(time_series):
        if value == 0:
            consecutive_zero_at_end += 1
        else:
            break
    features['F9'] = consecutive_zero_at_end / len(time_series)

    return features


# 设置随机种子以确保可重复性
np.random.seed(42)
tf.random.set_seed(42)
num_samples = 5000
sequence_length = 84
data = pd.read_excel(r'E:\简单间歇性需求\原始RAF_forecast.xlsx')
data.set_index(data.columns[0], inplace=True)
input_data = data.T
input_shape = (sequence_length,)
print(input_data.shape[1])
encoding_dim = 12
input_layer = tf.keras.layers.Input(shape=input_shape)
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
decoded = tf.keras.layers.Dense(sequence_length, activation='sigmoid')(encoded)
autoencoder = tf.keras.models.Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(input_data, input_data, epochs=50, batch_size=16)
encoder = tf.keras.models.Model(input_layer, encoded)
encoded_data = encoder.predict(input_data)
feature_autoencoder = pd.DataFrame(encoded_data, columns=[f'F{i}' for i in range(10, encoding_dim + 10)])
# plot_model(autoencoder, to_file='autoencoder_model.png', show_shapes=True, show_layer_names=True)
feature_9 = pd.DataFrame([compute_features(data[col]) for col in data.columns])
# x = pd.concat([feature_9, feature_autoencoder], axis=1)
# x.to_excel(r'E:\论文论文\迁移学习\程序\特征提取\0.90F9_encoder_feature_xin_2.xlsx')
x = pd.read_excel(r'E:\论文论文\迁移学习\程序\特征提取\0.96F9_encoder_feature.xlsx')
# x = input_data
X = x.iloc[:4000, :]
Y = x.iloc[4000:, :]
ks = KScorer()
labels, centroids, _ = ks.fit_predict(X, retall=True)
# save_to_pickle(ks, r'E:\论文论文\迁移学习\程序\特征提取\0.96F9_encoder_ks.pkl')
ks.show()  # 聚类点以及相应的得分高亮显示。这些带标签的点对应于所有指标的平均分数中的局部最大值，因此是选择最佳聚类数的最佳选项
K = ks.optimal_
kk = ks.ranked_
kk.to_excel(r'E:\论文论文\迁移学习\程序\特征提取\0.96F9_encoder_画图用.xlsx')
print(f'最佳聚类数为：{K}')
# kk.to_excel(r'E:\论文论文\迁移学习\程序\特征提取\0.96F9_encoder_画图用.xlsx')
label_count = Counter(labels)
for label, count in label_count.items():
    print(f"{label}: {count}")
print(ks.peak_scores_)
label = pd.DataFrame(labels, columns=['label'])
label.to_excel(r'E:\论文论文\迁移学习\程序\特征提取\0.96F9_encoder_xxx.xlsx')
y_label = ks.predict(Y, retall=False)
yy_label = pd.DataFrame(y_label, columns=['label'])
yy_label.to_excel(r'E:\论文论文\迁移学习\程序\特征提取\0.96F9_encoder_yyy.xlsx')

