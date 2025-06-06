import time

import pandas as pd
import numpy as np
from gluonts.dataset.common import ListDataset
from gluonts.model.npts import NPTSPredictor
from gluonts.model.deepar import DeepAREstimator
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
import warnings
import os
from gluonts.transform import ExpectedNumInstanceSampler

warnings.filterwarnings("ignore")
ExpectedNumInstanceSampler(num_instances=0.9)

def format_trans(data):
    # 转化训练数据和测试数据的格式
    start_date = '1996-01-01'
    date_range = pd.date_range(start=start_date, periods=150, freq='D')  # 从开始日期生成84个月的时间序列
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
                "target": data[col].iloc[:138].values.tolist(),  # 仅取前78行作为target
                "feat_static_cat": [i - 1],
            }
            Qtrain_test_data.append(Qtrain_test_series)
            Qtrain_train_data.append(Qtrain_train_series)
    train_data = ListDataset(Qtrain_train_data, freq="D")  # 这里假设数据是每月一条记录，可以根据实际情况调整频率
    test_data = ListDataset(Qtrain_test_data, freq="D")
    # 生成tss
    tss = []
    for col in data.columns:
        df = pd.DataFrame(
            {'time': pd.date_range(start='1996-01-01', periods=150, freq='D'), col: data[col]})
        tss.append(df)
    if len(tss) > 0:
        del tss[0]
    for dff in tss:
        dff.set_index('time', inplace=True)
    return train_data, test_data, tss


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
                context_length=prediction_length,
                num_layers=2,
                num_cells=num_cells,
                cell_type=cell_type,
                dropout_rate=dropout_rate,
                scaling=True,
                lags_seq=list(np.arange(1, num_lags + 1)),
                # lags_seq=[1],
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


def Croston_forecast(train_data, test_data, tss, freq, prediction_length, evaluator):
    print('Croston_forecast')
    croston_predictor = CrostonForecastPredictor(freq=freq,
                                                 prediction_length=prediction_length,
                                                 variant='original',
                                                 no_of_params=2
                                                 )

    croston_forecast = list(croston_predictor.predict(train_data))
    croston_agg_metrics, croston_item_metrics = evaluator(
        iter(tss), iter(croston_forecast), num_series=len(test_data)
    )
    return croston_agg_metrics


def calculate_model_acc(action, train_data, test_data, tss):
    evaluator = IntermittentEvaluator(quantiles=[0.25, 0.5, 0.75, 0.85], median=True, calculate_spec=False,
                                      round_integer=True)
    function_mapping = {
        'DeepAR': deepar_forecast,
        'Croston': Croston_forecast,
        'ARIMA': ARIMA_forecast,
        'SBJ': SBJ_forecast,
        'ETS': ETS_forecast,
        'DeepRenewal Flat': deeprenewal_forecast,
        'DeepRenewal Exact': deeprenewal_forecast,
        'DeepRenewal Hybrid': deeprenewal_forecast,
    }
    if action in function_mapping:
        if action == 'Croston' or action == 'SBA' or action == 'SBJ' or action == 'ARIMA' or action == 'ETS':
            metric = function_mapping[action](train_data, test_data, tss, 'D', 12, evaluator)
            return metric
        elif action == 'DeepRenewal Flat' or action == 'DeepRenewal Exact' or action == 'DeepRenewal Hybrid':
            metric = function_mapping[action](train_data, test_data, tss, 12, 0.01, 'D', 10, 'lstm',
                                              64, 0.3, 1, evaluator, action)
            return metric
        elif action == 'DeepAR':
            metric = function_mapping[action](train_data, test_data, tss, 12, 0.01, 'D',
                                              10, 'gru', 128, 0.1, 2, evaluator)
            return metric
        else:
            print('模型不在已知范围内')


if __name__ == '__main__':
    data = pd.read_excel(r"E:\数据集\荷兰制造公司3451件商品销售情况\manufacturing_dataset_non_zero_filtered.xlsx", index_col=0)

    model_list = ['DeepRenewal Flat', 'DeepRenewal Exact', 'DeepRenewal Hybrid', 'ETS', 'DeepAR', 'SBJ']
    train_data_b, test_data_b, tss_b = format_trans(data)
    error = []
    for model in model_list:
        start_time = time.time()  # 记录开始时间
        metric = calculate_model_acc(model, train_data_b, test_data_b, tss_b)
        end_time = time.time()  # 记录结束时间
        print(f"{model} 总运行时间: {end_time - start_time:.2f} 秒")  # 输出总耗时
        error.append(metric)
    errors = pd.DataFrame(error, index=model_list)
    # errors.to_excel(r'E:\gluon_ts\新聚类方法\直接预测荷兰.xlsx')
