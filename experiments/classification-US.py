from collections import Counter
import numpy as np
import pandas as pd
import tensorflow as tf
from kscorer.kscorer import KScorer
from scipy.stats import variation
import statsmodels.api as sm
import antropy as ant
import pickle
import warnings

warnings.filterwarnings("ignore")


def load_from_pickle(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    print(f"数据已从 {filename} 文件中加载。")
    return data


def compute_features(time_series):
    features = {}

    # F1: Average Demand Interval (ADI)
    demand_indices = np.where(time_series > 0)[0]
    if len(demand_indices) > 1:
        inter_demand_intervals = np.diff(demand_indices)
        features['F1'] = np.mean(inter_demand_intervals)
    else:
        features['F1'] = np.nan

    # F2: Square of Coefficient of Variation (CV^2)
    non_zero_demand = time_series[time_series > 0]
    if len(non_zero_demand) > 0:
        features['F2'] = variation(non_zero_demand) ** 2
    else:
        features['F2'] = np.nan

    # F3: Approximate Entropy
    features['F3'] = ant.app_entropy(time_series, order=2)

    # F4: Percentage of Zero Values
    features['F4'] = np.sum(time_series == 0) / len(time_series)

    # F5: Percentage of Values Outside [mean - std, mean + std] Range
    mean_y = np.mean(time_series)
    std_y = np.std(time_series)
    features['F5'] = np.sum((time_series < mean_y - std_y) | (time_series > mean_y + std_y)) / len(time_series)

    # F6: Linear Least Squares Regression Coefficient
    chunk_size = 12  # For monthly data
    chunks = [time_series[i:i + chunk_size] for i in range(0, len(time_series), chunk_size)]
    variances = [np.var(chunk) for chunk in chunks if len(chunk) == chunk_size]
    if len(variances) > 1:
        x = np.arange(len(variances))
        X = sm.add_constant(x)
        model = sm.OLS(variances, X).fit()
        features['F6'] = model.params[1]
    else:
        features['F6'] = np.nan

    # F7: Average Absolute Value of Consecutive Changes
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


if __name__ == '__main__':
    # 读取数据
    data = pd.read_excel(r"E:\论文论文\迁移学习\程序\特征提取\美国汽车零部件2674非0少5.xlsx", index_col=0)

    sequence_length = 51
    input_shape = (sequence_length,)
    encoding_dim = 12
    in_data = data.iloc[:, 1:]
    input_data = data.iloc[:, 1:].T
    input_layer = tf.keras.layers.Input(shape=input_shape)
    # encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
    # decoded = tf.keras.layers.Dense(sequence_length, activation='sigmoid')(encoded)
    # autoencoder = tf.keras.models.Model(input_layer, decoded)
    # # 编译模型
    # autoencoder.compile(optimizer='adam', loss='mse')
    # # 训练自动编码器
    # autoencoder.fit(input_data, input_data, epochs=50, batch_size=16)
    # # 提取特征
    # encoder = tf.keras.models.Model(input_layer, encoded)
    # encoded_data = encoder.predict(input_data)
    # feature_autoencoder = pd.DataFrame(encoded_data, columns=[f'F{i}' for i in range(10, encoding_dim + 10)])
    # # 手动提取9个特征
    # feature_9 = pd.DataFrame([compute_features(in_data[col]) for col in in_data.columns])
    # x = pd.concat([feature_9, feature_autoencoder], axis=1)

    # x.to_excel(r"美国汽车零部件26741.xlsx")
    x = pd.read_excel(r"美国汽车零部件26741.xlsx")
    x = x.dropna()
    ks = load_from_pickle("0.96F9_encoder_ks.pkl")
    y_label = ks.predict(x, retall=False)
    yy_label = pd.DataFrame(y_label, columns=['label'])
    yy_label.to_excel(r"美国汽车零部件2674_y1.xlsx")
    label_count = Counter(y_label)
    for label, count in label_count.items():
        print(f"{label}: {count}")