import pandas as pd
import numpy as np
from gluonts.dataset.common import ListDataset
from gluonts.model.npts import NPTSPredictor
from gluonts.model.deepar import DeepAREstimator
from gluonts.distribution.neg_binomial import NegativeBinomialOutput
from gluonts.distribution.piecewise_linear import PiecewiseLinearOutput
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.r_forecast import RForecastPredictor
from deeprenewal import DeepRenewalEstimator
from deeprenewal import CrostonForecastPredictor
import pickle
from deeprenewal import IntermittentEvaluator
import warnings
import random
from collections import Counter
import tensorflow as tf
from kscorer.kscorer import KScorer
from scipy.stats import variation
import statsmodels.api as sm
import antropy as ant
import pickle
import os

warnings.filterwarnings('ignore')


def save_pickle(data, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
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

class QLearningAgent:
    def __init__(self, categories, actions, error, max_iterations=5000, alpha=0.05, gamma=0.8, epsilon=0.9):
        self.categories = categories
        self.actions = actions
        self.q_table = pd.DataFrame({state: {action: 0.0 for action in actions.values()} for state in categories})
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.gamma = gamma
        self.error = pd.DataFrame(error)
        self.epsilon = epsilon
        self.r_value = self._initialize_r_value()

    def _initialize_r_value(self):
        error = self.error.T
        mean_values = error.mean()
        # 将均值插入到DataFrame的最后一行
        error.loc['mean'] = mean_values
        error_mean = error.loc['mean']
        error = error.drop('mean')
        for col in error.columns:
            min_val = error[col].min()
            error[col] = np.where(error[col] == min_val, 100,
                                  np.where((error[col] <= error_mean.loc[col]) & (error[col] > min_val), 0, -1))
        r_value = error.copy()
        return r_value

    def get_model_reward(self, current_state, action):
        return self.r_value.loc[action, current_state]

    def get_next_state(self, action):
        valid_indices = self.r_value.columns[self.r_value.loc[action] != -1].tolist()
        return random.choice(valid_indices) if valid_indices else None

    def q_learning(self):
        for iteration in range(self.max_iterations):
            current_state = random.choice(self.categories)
            while True:
                if random.uniform(0, 1) < self.epsilon:
                    max_action = random.choice(self.r_value[self.r_value[current_state] != -1].index.tolist())
                else:
                    max_action = self.q_table[current_state].idxmax()

                reward = self.get_model_reward(current_state, max_action)
                next_state = self.get_next_state(max_action)
                max_next_q_value = max(self.q_table[next_state]) if next_state in self.q_table else 0
                self.q_table[current_state][max_action] = (1 - self.alpha) * self.q_table[current_state][
                    max_action] + self.alpha * (
                                                                  reward + self.gamma * max_next_q_value)

                if next_state is None or reward == 100:
                    break
                current_state = next_state

            if iteration % 100 == 0:
                print(f"Iteration {iteration}:")
                print(self.q_table)

        print("Final Q-table:")
        print(self.q_table)
        return self.q_table


def format_trans(data):
    # 转化训练数据和测试数据的格式
    start_date = '1996-01-01'
    date_range = pd.date_range(start=start_date, periods=84, freq='M')  # 从开始日期生成84个月的时间序列
    data.insert(0, 'ds', date_range)  # 在第一列插入时间序列
    Qtrain_test_data = []
    Qtrain_train_data = []
    for i, col in enumerate(data.columns):
        if col != 'ds':
            Qtrain_test_series = {
                "start": data['ds'].iloc[0],
                "target": data[col].values.tolist(),
                "feat_static_cat": [i - 1],  # 添加静态特征，这里假设静态特征为每个序列的索引 i
            }
            Qtrain_train_series = {
                "start": data['ds'].iloc[0],
                "target": data[col].iloc[:78].values.tolist(),  # 仅取前78行作为target
                "feat_static_cat": [i - 1],
            }
            Qtrain_test_data.append(Qtrain_test_series)
            Qtrain_train_data.append(Qtrain_train_series)
    train_data = ListDataset(Qtrain_train_data, freq="M")  # 这里假设数据是每月一条记录，可以根据实际情况调整频率
    test_data = ListDataset(Qtrain_test_data, freq="M")
    # 生成tss
    tss = []
    for col in data.columns:
        df = pd.DataFrame(
            {'time': pd.date_range(start='1996-01-01', periods=84, freq='M'), col: data[col]})
        tss.append(df)
    if len(tss) > 0:
        del tss[0]
    for dff in tss:
        dff.set_index('time', inplace=True)
    return train_data, test_data, tss


def ARIMA_forecast(train_data, test_data, tss, freq, prediction_length, evaluator):
    print('ARIMA_forecast')
    arima_predictor = RForecastPredictor(freq=freq,
                                         prediction_length=prediction_length,
                                         method_name='arima',
                                         )
    arima_forecast = list(arima_predictor.predict(train_data))
    arima_agg_metrics, arima_item_metrics = evaluator(
        iter(tss), iter(arima_forecast), num_series=len(test_data)
    )
    return arima_agg_metrics


def deeprenewal_forecast(train_data, test_data, tss, prediction_length, learning_rate, freq, epochs, cell_type,
                         num_cells, dropout_rate, num_lags, evaluator, name):
    while True:
        try:
            trainer = Trainer(
                learning_rate=learning_rate,
                epochs=epochs,
                num_batches_per_epoch=100,
                clip_gradient=5.170127652392614,
                weight_decay=0.01,
                hybridize=True)  # hybridize false for development
            estimator = DeepRenewalEstimator(
                prediction_length=prediction_length,
                context_length=prediction_length * 2,
                num_layers=2,
                num_cells=num_cells,
                cell_type=cell_type,
                dropout_rate=dropout_rate,
                scaling=True,
                lags_seq=list(np.arange(1, num_lags + 1)),
                freq=freq,
                use_feat_dynamic_real=False,
                use_feat_static_cat=False,
                use_feat_static_real=False,
                cardinality=None,
                trainer=trainer,
            )
            predictor = estimator.train(train_data, test_data)
            break  # 如果成功训练模型，跳出循环
        except Exception as e:
            print(f"遇到错误: {e}")
            print("重新训练模型...")
            continue  # 继续下一次循环重新训练模型
    if name == 'DeepRenewal Flat':
        print('deep_renewal_flat_forecast')
        deep_renewal_flat_forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_data, predictor=predictor, num_samples=100
        )
        deep_renewal_flat_forecast = [prediction for prediction in deep_renewal_flat_forecast_it]
        print('评估')
        deep_renewal_flat_agg_metrics, deep_renewal_flat_item_metrics = evaluator(
            iter(tss), iter(deep_renewal_flat_forecast), num_series=len(test_data)
        )
        return deep_renewal_flat_agg_metrics
    elif name == 'DeepRenewal Exact':
        print('Deep Renewal Exact')
        predictor.forecast_generator.forecast_type = "exact"
        deep_renewal_exact_forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_data, predictor=predictor, num_samples=100
        )
        deep_renewal_exact_forecasts = [prediction for prediction in deep_renewal_exact_forecast_it]
        print('评估')
        deep_renewal_exact_agg_metrics, deep_renewal_exact_item_metrics = evaluator(
            iter(tss), iter(deep_renewal_exact_forecasts), num_series=len(test_data)
        )
        return deep_renewal_exact_agg_metrics
    elif name == 'DeepRenewal Hybrid':
        print('Deep Renewal Hybrid')
        predictor.forecast_generator.forecast_type = "hybrid"
        deep_renewal_hybrid_forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_data, predictor=predictor, num_samples=100
        )
        deep_renewal_hybrid_forecasts = [prediction for prediction in deep_renewal_hybrid_forecast_it]
        print('评估')
        deep_renewal_hybrid_agg_metrics, deep_renewal_hybrid_item_metrics = evaluator(
            iter(tss), iter(deep_renewal_hybrid_forecasts), num_series=len(test_data)
        )
        return deep_renewal_hybrid_agg_metrics


def deepar_forecast(train_data, test_data, tss, prediction_length, learning_rate, freq, epochs, cell_type, num_cells,
                    dropout_rate, num_layers, evaluator):
    distr = PiecewiseLinearOutput(7)
    deep_ar_trainer = Trainer(
        batch_size=128,
        learning_rate=learning_rate,
        epochs=epochs,
        num_batches_per_epoch=100,
        clip_gradient=5.48481845049343,
        weight_decay=0.001,
        hybridize=True)  # hybridize false for development

    deep_ar_estimator = DeepAREstimator(
        prediction_length=prediction_length,
        context_length=prediction_length * 2,
        num_layers=num_layers,
        num_cells=num_cells,
        cell_type=cell_type,
        dropout_rate=dropout_rate,
        scaling=True,
        lags_seq=list(np.arange(1, 1 + 1)),
        freq=freq,
        use_feat_dynamic_real=False,
        use_feat_static_cat=False,
        use_feat_static_real=False,
        distr_output=distr,
        cardinality=None,  # cardinality,
        trainer=deep_ar_trainer,
    )
    print('DeepAR')
    deep_ar_predictor = deep_ar_estimator.train(train_data, test_data)
    deep_ar_forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_data, predictor=deep_ar_predictor, num_samples=100
    )
    deep_ar_forecast = [prediction for prediction in deep_ar_forecast_it]
    deep_ar_agg_metrics, deep_ar_item_metrics = evaluator(
        iter(tss), iter(deep_ar_forecast), num_series=len(test_data)
    )
    return deep_ar_agg_metrics


def ETS_forecast(train_data, test_data, tss, freq, prediction_length, evaluator):
    print('ETS_forecast')
    ets_predictor = RForecastPredictor(freq=freq,
                                       prediction_length=prediction_length,
                                       method_name='ets',
                                       )
    ets_forecast = list(ets_predictor.predict(train_data))
    ets_agg_metrics, ets_item_metrics = evaluator(
        iter(tss), iter(ets_forecast), num_series=len(test_data)
    )
    return ets_agg_metrics


def SBJ_forecast(train_data, test_data, tss, freq, prediction_length, evaluator):
    print('SBJ_forecast')
    sbj_predictor = CrostonForecastPredictor(freq=freq,
                                             prediction_length=prediction_length,
                                             variant='sbj',
                                             no_of_params=2
                                             )
    sbj_forecast = list(sbj_predictor.predict(train_data))
    sbj_agg_metrics, sbj_item_metrics = evaluator(
        iter(tss), iter(sbj_forecast), num_series=len(test_data)
    )
    return sbj_agg_metrics


def NPTS_forecast(train_data, test_data, tss, freq, prediction_length, context_length, evaluator):
    print('NPTS_forecast')
    npts_predictor = NPTSPredictor(freq=freq, prediction_length=prediction_length, context_length=context_length,
                                   kernel_type="uniform", use_seasonal_model=False)
    npts_forecast = list(npts_predictor.predict(train_data))
    npts_agg_metrics, npts_item_metrics = evaluator(
        iter(tss), iter(npts_forecast), num_series=len(test_data)
    )
    return npts_agg_metrics


def calculate_model_acc(action, train_data, test_data, tss):
    evaluator = IntermittentEvaluator(quantiles=[0.25, 0.5, 0.75, 0.85], median=True, calculate_spec=False,
                                      round_integer=True)
    function_mapping = {
        'DeepAR': deepar_forecast,
        'SBJ': SBJ_forecast,
        'ARIMA': ARIMA_forecast,
        'ETS': ETS_forecast,
        'DeepRenewal Flat': deeprenewal_forecast,
        'DeepRenewal Exact': deeprenewal_forecast,
        'DeepRenewal Hybrid': deeprenewal_forecast,
    }
    if action in function_mapping:
        if action == 'Croston' or action == 'SBA' or action == 'SBJ' or action == 'ARIMA' or action == 'ETS':
            metric = function_mapping[action](train_data, test_data, tss, 'M', 6, evaluator)
            return metric
        elif action == 'DeepRenewal Flat' or action == 'DeepRenewal Exact' or action == 'DeepRenewal Hybrid':
            metric = function_mapping[action](train_data, test_data, tss, 6, 0.01, 'M', 10, 'lstm',
                                              64, 0.3, 1, evaluator, action)
            return metric
        elif action == 'DeepAR':
            metric = function_mapping[action](train_data, test_data, tss, 6, 0.01, 'M',
                                              10, 'gru', 128, 0.1, 2, evaluator)
            return metric
        else:
            print('模型不在已知范围内')


if __name__ == '__main__':
    df = pd.read_excel(r"D:\gluon_ts\新聚类方法\RAF_forecast2.xlsx")
    # 特征提取agent1
    np.random.seed(42)
    tf.random.set_seed(42)
    num_samples = 5000
    sequence_length = 84
    data = pd.read_excel(r'E:\简单间歇性需求\原始RAF_forecast.xlsx')
    data.set_index(data.columns[0], inplace=True)
    input_data = data.T
    input_shape = (sequence_length,)
    print(input_data.shape[1])
    encoding_dim = 12   # 可自定义
    input_layer = tf.keras.layers.Input(shape=input_shape)
    encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
    decoded = tf.keras.layers.Dense(sequence_length, activation='sigmoid')(encoded)
    autoencoder = tf.keras.models.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(input_data, input_data, epochs=50, batch_size=16)
    encoder = tf.keras.models.Model(input_layer, encoded)
    encoded_data = encoder.predict(input_data)
    feature_autoencoder = pd.DataFrame(encoded_data, columns=[f'F{i}' for i in range(10, encoding_dim + 10)])
    plot_model(autoencoder, to_file='autoencoder_model.png', show_shapes=True, show_layer_names=True)
    feature_9 = pd.DataFrame([compute_features(data[col]) for col in data.columns])
    x = pd.concat([feature_9, feature_autoencoder], axis=1)
    x.to_excel(r'E:\论文论文\迁移学习\程序\特征提取\0.96F9_encoder_feature_xin_2.xlsx')
    x = pd.read_excel(r'E:\论文论文\迁移学习\程序\特征提取\0.96F9_encoder_feature.xlsx')
    # 分类聚类agent2
    X = x.iloc[:4000, :]
    Y = x.iloc[4000:, :]
    ks = KScorer()
    labels, centroids, _ = ks.fit_predict(X, retall=True)
    save_to_pickle(ks, r'E:\论文论文\迁移学习\程序\特征提取\0.96F9_encoder_ks.pkl')
    ks.show()  # 聚类点以及相应的得分高亮显示。这些带标签的点对应于所有指标的平均分数中的局部最大值，因此是选择最佳聚类数的最佳选项
    K = ks.optimal_
    kk = ks.ranked_
    kk.to_excel(r'E:\论文论文\迁移学习\程序\特征提取\0.96F9_encoder.xlsx')
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
    a_data = df.iloc[:, :4001]
    b_data = pd.concat([df.iloc[:, 0], df.iloc[:, 4001:]], axis=1)
    labels = pd.DataFrame()
    data_names = a_data.columns.tolist()[1:]
    labels['data_names'] = data_names
    labels['category'] = pd.read_excel(r"E:\论文论文\迁移学习\程序\特征提取\0.96F9_encoder_x.xlsx", index_col=0)
    model_list = ['SBJ', 'ETS', 'DeepAR', 'DeepRenewal Flat', 'DeepRenewal Exact', "ARIMA",
                  'DeepRenewal Hybrid']
    # 按照category列进行分组，并将各个category的value放入不同位置的列表中
    # 训练和评估，为agent3输出特征-模型对，即训练数据
    grouped_data = labels.groupby('category')['data_names'].apply(list).reset_index()
    epochs = 10
    errors_cate = []
    # 训练和评估模型
    for epoch in range(epochs):
        for cate in range(grouped_data.shape[0]):
            print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~第{cate}类~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            data_a = a_data[grouped_data['data_names'][cate]]
            train_data_a, test_data_a, tss_a = format_trans(data_a)
            error = []
            for model in model_list:
                metric = calculate_model_acc(model, train_data_a, test_data_a, tss_a)
                error.append(metric)
            errors = pd.DataFrame(error, index=model_list)
            errors.to_excel(r'D:\gluon_ts\新聚类方法\结果{}-{}.xlsx'.format(epoch, cate))
            errors_cate.append(errors)
    new_error = []
    av_error = []
    error_name = ['MRAE', 'MASE', 'MAAPE', 'MAE']
    for i in range(grouped_data.shape[0]):
        avg_df = pd.DataFrame()
        for j in range(epochs):
            index = i + j * grouped_data.shape[0]
            avg_df = avg_df.add(errors_cate[index], fill_value=0)
        avg_df = avg_df / epochs  # 求均值
        new_error.append(avg_df)
        aa_data = avg_df[error_name]
        row_means = aa_data.mean(axis=1)
        av_error.append(row_means)
    av_error = pd.DataFrame(av_error)
    row_index = ['state0', 'state1', 'state2', 'state3', 'state4', 'state5', 'state6']
    av_error = av_error.rename(index=dict(zip(av_error.index, row_index)))
    # # # 生成Q表
    actions = {
        1: "SBJ",
        2: "ARIMA",
        3: "ETS",
        4: "DeepRenewal Flat",
        5: "DeepRenewal Exact",
        6: "DeepRenewal Hybrid",
        7: "DeepAR"
    }
    av_error.to_excel(r'D:\gluon_ts\新聚类方法\av_error_xin2.xlsx')
    # av_error = pd.read_excel(r'D:\gluon_ts\新聚类方法\av_error_xin2.xlsx', index_col=0)
    # av_error = load_pickle(r"D:\gluon_ts\新聚类方法\av_error.pkl")
    agent = QLearningAgent(row_index, actions, av_error)
    q_tables = agent.q_learning()
    q_tables.to_excel(r'D:\gluon_ts\新聚类方法\q_tables_xin2.xlsx')
    labels_b = pd.DataFrame()
    data_names = b_data.columns.tolist()[1:]
    labels_b['data_names'] = data_names
    labels_b['category'] = pd.read_excel(r"E:\论文论文\迁移学习\程序\特征提取\0.96F9_encoder_y.xlsx", index_col=0)
    # 按照category列进行分组，并将各个category的value放入不同位置的列表中
    grouped_data_test = labels_b.groupby('category')['data_names'].apply(list).reset_index()

    grouped_num = []
    for i in range(grouped_data_test.shape[0]):
        grouped_num.append(len(grouped_data_test['data_names'][i]))
    grouped = np.sum(grouped_num)
    if grouped != 1000:
        print('老铁有毛病了，快修改！')
    else:
        metrics_b = []
        for test_cate in range(grouped_data_test.shape[0]):
            data_b = b_data[grouped_data_test['data_names'][test_cate]]
            train_data_b, test_data_b, tss_b = format_trans(data_b)
            model_select = q_tables[row_index[test_cate]].idxmax()
            metric_b = calculate_model_acc(model_select, train_data_b, test_data_b, tss_b)
            metrics_b.append(metric_b)
        metrics_b = pd.DataFrame(metrics_b)
        metrics_b.to_excel(r"新聚类预测结果.xlsx")
