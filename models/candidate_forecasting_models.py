"""
候选预测模型 - Candidate Forecasting Models

该模块包含ABCM系统中使用的所有候选预测模型，专门用于间歇性需求预测
This module contains all candidate forecasting models used in the ABCM system,
specifically designed for intermittent demand forecasting

候选模型 Candidate Models:
1. DeepAR - 深度自回归模型 Deep Autoregressive Model
2. Deep Renewal (Flat/Exact/Hybrid) - 深度更新过程模型 Deep Renewal Process Models  
3. Croston - 克罗斯顿方法 Croston's Method
4. SBJ - Syntetos-Boylan-Jakobs方法 SBJ Method
5. ARIMA - 自回归综合移动平均模型 AutoRegressive Integrated Moving Average
6. ETS - 指数平滑状态空间模型 Exponential Smoothing State Space Model
7. NPTS - 非参数时间序列模型 Non-Parametric Time Series Model

作者 Author: ABCM Team
创建时间 Created: 2024
"""

import warnings
import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.model.npts import NPTSPredictor
from gluonts.model.deepar import DeepAREstimator
from gluonts.distribution.piecewise_linear import PiecewiseLinearOutput
from gluonts.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.r_forecast import RForecastPredictor
from deeprenewal import DeepRenewalEstimator, CrostonForecastPredictor
from deeprenewal import IntermittentEvaluator

warnings.filterwarnings("ignore")


class ForecastingModels:
    """
    预测模型容器类
    Container class for all forecasting models used in ABCM
    
    该类整合了9种专门用于间歇性需求预测的模型，每种模型都有其特定的适用场景：
    This class integrates 9 models specifically designed for intermittent demand forecasting,
    each with its specific use cases:
    
    深度学习模型 Deep Learning Models:
    - DeepAR: 适用于复杂模式和长期依赖 Suitable for complex patterns and long-term dependencies
    - Deep Renewal: 专门建模间歇性需求的到达过程 Specifically models arrival processes of intermittent demand
    
    传统统计模型 Traditional Statistical Models:
    - Croston & SBJ: 专门为间歇性需求设计 Specifically designed for intermittent demand
    - ARIMA: 适用于有趋势和季节性的数据 Suitable for data with trend and seasonality
    - ETS: 指数平滑方法的现代实现 Modern implementation of exponential smoothing
    - NPTS: 非参数方法，适用于复杂分布 Non-parametric method for complex distributions
    """
    
    def __init__(self, prediction_length=6, freq='M'):
        """
        初始化预测模型容器
        Initialize forecasting models container
        
        参数 Args:
            prediction_length (int): 预测时长(月数) Number of periods to forecast
            freq (str): 时间序列频率 Frequency of time series data
        """
        self.prediction_length = prediction_length
        self.freq = freq
        self.evaluator = IntermittentEvaluator(
            quantiles=[0.25, 0.5, 0.75, 0.85], 
            median=True, 
            calculate_spec=False,
            round_integer=True
        )
        
        print(f"预测模型初始化：预测长度={prediction_length}，频率={freq}")
        print(f"Forecasting models initialized: prediction_length={prediction_length}, freq={freq}")
    
    def format_data(self, data):
        """
        将数据转换为GluonTS格式
        Transform data to GluonTS format
        
        GluonTS是亚马逊开发的时间序列预测库，需要特定的数据格式
        GluonTS is a time series forecasting library developed by Amazon, requiring specific data format
        
        参数 Args:
            data (pd.DataFrame): 输入时间序列数据 Input time series data
            
        返回 Returns:
            tuple: (train_data, test_data, tss) 训练数据、测试数据和时间序列列表
        """
        print("格式化数据为GluonTS格式... Formatting data to GluonTS format...")
        
        start_date = '1996-01-01'
        n_periods = len(data)
        date_range = pd.date_range(start=start_date, periods=n_periods, freq=self.freq)
        data.insert(0, 'ds', date_range)
        
        train_test_data = []
        train_data_list = []
        
        for i, col in enumerate(data.columns):
            if col != 'ds':
                # 测试数据(完整序列) Test data (full series)
                test_series = {
                    "start": data['ds'].iloc[0],
                    "target": data[col].values.tolist(),
                    "feat_static_cat": [i - 1],
                }
                
                # 训练数据(排除最后prediction_length个周期) Training data (excluding last prediction_length periods)
                train_series = {
                    "start": data['ds'].iloc[0],
                    "target": data[col].iloc[:-self.prediction_length].values.tolist(),
                    "feat_static_cat": [i - 1],
                }
                
                train_test_data.append(test_series)
                train_data_list.append(train_series)
        
        train_data = ListDataset(train_data_list, freq=self.freq)
        test_data = ListDataset(train_test_data, freq=self.freq)
        
        # 生成用于评估的时间序列 Generate time series for evaluation
        tss = []
        for col in data.columns:
            if col != 'ds':
                df = pd.DataFrame({
                    'time': date_range, 
                    col: data[col]
                })
                df.set_index('time', inplace=True)
                tss.append(df)
        
        print(f"数据格式化完成：{len(train_data_list)}个时间序列")
        print(f"Data formatting completed: {len(train_data_list)} time series")
        
        return train_data, test_data, tss
    
    def deepar_forecast(self, train_data, test_data, tss, 
                       learning_rate=0.01, epochs=10, cell_type='gru', 
                       num_cells=128, dropout_rate=0.1, num_layers=2):
        """
        DeepAR预测模型
        DeepAR forecasting model
        
        DeepAR是一种基于RNN的概率预测模型，特别适合处理多个相关时间序列
        DeepAR is an RNN-based probabilistic forecasting model, particularly suitable for multiple related time series
        
        参数 Args:
            train_data: 训练数据 Training data
            test_data: 测试数据 Test data  
            tss: 时间序列列表 Time series list
            learning_rate (float): 学习率 Learning rate
            epochs (int): 训练轮数 Training epochs
            cell_type (str): RNN单元类型 RNN cell type
            num_cells (int): RNN单元数量 Number of RNN cells
            dropout_rate (float): Dropout比率 Dropout rate
            num_layers (int): 网络层数 Number of layers
            
        返回 Returns:
            tuple: (metrics, forecasts) 评估指标和预测结果
        """
        print('训练DeepAR模型... Training DeepAR...')
        
        # 配置分布输出 Configure distribution output
        distr = PiecewiseLinearOutput(7)
        
        # 配置训练器 Configure trainer
        trainer = Trainer(
            batch_size=128,
            learning_rate=learning_rate,
            epochs=epochs,
            num_batches_per_epoch=100,
            clip_gradient=5.48481845049343,
            weight_decay=0.001,
            hybridize=True
        )

        # 配置估计器 Configure estimator
        estimator = DeepAREstimator(
            prediction_length=self.prediction_length,
            context_length=self.prediction_length * 2,
            num_layers=num_layers,
            num_cells=num_cells,
            cell_type=cell_type,
            dropout_rate=dropout_rate,
            scaling=True,
            lags_seq=list(np.arange(1, 2)),
            freq=self.freq,
            use_feat_dynamic_real=False,
            use_feat_static_cat=False,
            use_feat_static_real=False,
            distr_output=distr,
            trainer=trainer,
        )
        
        # 训练模型 Train model
        predictor = estimator.train(train_data, test_data)
        
        # 生成预测 Generate forecasts
        forecast_it, _ = make_evaluation_predictions(
            dataset=test_data, predictor=predictor, num_samples=100
        )
        forecasts = list(forecast_it)
        
        # 评估模型 Evaluate model
        agg_metrics, _ = self.evaluator(
            iter(tss), iter(forecasts), num_series=len(test_data)
        )
        
        print(f"DeepAR训练完成，MASE: {agg_metrics.get('MASE', 'N/A'):.3f}")
        print(f"DeepAR training completed, MASE: {agg_metrics.get('MASE', 'N/A'):.3f}")
        
        return agg_metrics, forecasts
    
    def deeprenewal_forecast(self, train_data, test_data, tss,
                           learning_rate=0.01, epochs=10, cell_type='lstm',
                           num_cells=64, dropout_rate=0.3, num_lags=1,
                           forecast_type='flat'):
        """
        Deep Renewal预测模型
        Deep Renewal forecasting model
        
        Deep Renewal专门为间歇性需求建模，将需求过程分解为需求到达时间间隔和需求大小
        Deep Renewal is specifically designed for intermittent demand, decomposing the demand process 
        into demand arrival intervals and demand sizes
        
        参数 Args:
            forecast_type (str): 预测类型 Forecast type
                               'flat': 平坦预测 Flat forecast
                               'exact': 精确预测 Exact forecast  
                               'hybrid': 混合预测 Hybrid forecast
        """
        print(f'训练Deep Renewal模型 ({forecast_type})... Training Deep Renewal ({forecast_type})...')
        
        while True:
            try:
                trainer = Trainer(
                    learning_rate=learning_rate,
                    epochs=epochs,
                    num_batches_per_epoch=100,
                    clip_gradient=5.170127652392614,
                    weight_decay=0.01,
                    hybridize=True
                )
                
                estimator = DeepRenewalEstimator(
                    prediction_length=self.prediction_length,
                    context_length=self.prediction_length,
                    num_layers=2,
                    num_cells=num_cells,
                    cell_type=cell_type,
                    dropout_rate=dropout_rate,
                    scaling=True,
                    lags_seq=list(np.arange(1, num_lags + 1)),
                    freq=self.freq,
                    use_feat_dynamic_real=False,
                    use_feat_static_cat=False,
                    use_feat_static_real=False,
                    trainer=trainer,
                )
                
                predictor = estimator.train(train_data, test_data)
                break
                
            except Exception as e:
                print(f"训练出错: {e}. 重试... Training error: {e}. Retrying...")
                continue
        
        # 设置预测类型 Set forecast type
        if forecast_type.lower() == 'exact':
            predictor.forecast_generator.forecast_type = "exact"
        elif forecast_type.lower() == 'hybrid':
            predictor.forecast_generator.forecast_type = "hybrid"
        else:  # flat
            predictor.forecast_generator.forecast_type = "flat"
            
        forecast_it, _ = make_evaluation_predictions(
            dataset=test_data, predictor=predictor, num_samples=100
        )
        forecasts = list(forecast_it)
        
        agg_metrics, _ = self.evaluator(
            iter(tss), iter(forecasts), num_series=len(test_data)
        )
        
        print(f"Deep Renewal ({forecast_type})训练完成")
        print(f"Deep Renewal ({forecast_type}) training completed")
        
        return agg_metrics, forecasts
    
    def croston_forecast(self, train_data, test_data, tss, variant='original'):
        """
        Croston预测模型
        Croston forecasting model
        
        Croston方法是专门为间歇性需求设计的经典方法，分别预测需求间隔和需求大小
        Croston's method is a classic method specifically designed for intermittent demand,
        separately forecasting demand intervals and demand sizes
        """
        print(f'训练Croston模型 ({variant})... Training Croston ({variant})...')
        
        predictor = CrostonForecastPredictor(
            freq=self.freq,
            prediction_length=self.prediction_length,
            variant=variant,
            no_of_params=2
        )
        
        forecasts = list(predictor.predict(train_data))
        
        agg_metrics, _ = self.evaluator(
            iter(tss), iter(forecasts), num_series=len(test_data)
        )
        
        print(f"Croston ({variant})训练完成")
        print(f"Croston ({variant}) training completed")
        
        return agg_metrics, forecasts
    
    def sbj_forecast(self, train_data, test_data, tss):
        """
        SBJ (Syntetos-Boylan-Jakobs)预测模型
        SBJ (Syntetos-Boylan-Jakobs) forecasting model
        
        SBJ是Croston方法的改进版本，对偏差进行了修正
        SBJ is an improved version of Croston's method with bias correction
        """
        print('训练SBJ模型... Training SBJ...')
        
        predictor = CrostonForecastPredictor(
            freq=self.freq,
            prediction_length=self.prediction_length,
            variant='sbj',
            no_of_params=2
        )
        
        forecasts = list(predictor.predict(train_data))
        
        agg_metrics, _ = self.evaluator(
            iter(tss), iter(forecasts), num_series=len(test_data)
        )
        
        print("SBJ训练完成 SBJ training completed")
        
        return agg_metrics, forecasts
    
    def arima_forecast(self, train_data, test_data, tss):
        """
        ARIMA预测模型
        ARIMA forecasting model
        
        ARIMA是经典的时间序列预测方法，适用于有趋势和季节性的数据
        ARIMA is a classic time series forecasting method, suitable for data with trend and seasonality
        """
        print('训练ARIMA模型... Training ARIMA...')
        
        predictor = RForecastPredictor(
            freq=self.freq,
            prediction_length=self.prediction_length,
            method_name='arima',
        )
        
        forecasts = list(predictor.predict(train_data))
        
        agg_metrics, _ = self.evaluator(
            iter(tss), iter(forecasts), num_series=len(test_data)
        )
        
        print("ARIMA训练完成 ARIMA training completed")
        
        return agg_metrics, forecasts
    
    def ets_forecast(self, train_data, test_data, tss):
        """
        ETS (指数平滑)预测模型
        ETS (Exponential Smoothing) forecasting model
        
        ETS是指数平滑方法的现代状态空间形式，适用于各种时间序列模式
        ETS is the modern state space form of exponential smoothing, suitable for various time series patterns
        """
        print('训练ETS模型... Training ETS...')
        
        predictor = RForecastPredictor(
            freq=self.freq,
            prediction_length=self.prediction_length,
            method_name='ets',
        )
        
        forecasts = list(predictor.predict(train_data))
        
        agg_metrics, _ = self.evaluator(
            iter(tss), iter(forecasts), num_series=len(test_data)
        )
        
        print("ETS训练完成 ETS training completed")
        
        return agg_metrics, forecasts
    
    def npts_forecast(self, train_data, test_data, tss, context_length=None):
        """
        NPTS (非参数时间序列)预测模型
        NPTS (Non-Parametric Time Series) forecasting model
        
        NPTS是一种非参数方法，不需要假设特定的数据分布，适用于复杂的时间序列
        NPTS is a non-parametric method that doesn't assume specific data distributions, 
        suitable for complex time series
        """
        print('训练NPTS模型... Training NPTS...')
        
        if context_length is None:
            context_length = self.prediction_length * 2
        
        predictor = NPTSPredictor(
            freq=self.freq, 
            prediction_length=self.prediction_length, 
            context_length=context_length,
            kernel_type="uniform", 
            use_seasonal_model=False
        )
        
        forecasts = list(predictor.predict(train_data))
        
        agg_metrics, _ = self.evaluator(
            iter(tss), iter(forecasts), num_series=len(test_data)
        )
        
        print("NPTS训练完成 NPTS training completed")
        
        return agg_metrics, forecasts
    
    def run_all_models(self, data, models_to_run=None):
        """
        运行所有预测模型并返回结果
        Run all forecasting models and return results
        
        这是模型容器的主要方法，用于批量运行所有候选模型
        This is the main method of the model container for batch running all candidate models
        
        参数 Args:
            data (pd.DataFrame): 输入时间序列数据 Input time series data
            models_to_run (list): 要运行的模型列表(如果为None，运行所有模型) List of models to run (if None, run all)
            
        返回 Returns:
            dict: 所有模型的结果 Results from all models
        """
        if models_to_run is None:
            models_to_run = [
                'DeepAR', 'DeepRenewal Flat', 'DeepRenewal Exact', 
                'DeepRenewal Hybrid', 'Croston', 'SBJ', 'ARIMA', 'ETS', 'NPTS'
            ]
        
        print(f"开始运行{len(models_to_run)}个预测模型...")
        print(f"Starting to run {len(models_to_run)} forecasting models...")
        
        # 格式化数据 Format data
        train_data, test_data, tss = self.format_data(data)
        
        results = {}
        
        for model_name in models_to_run:
            print(f"\n{'='*50}")
            print(f"运行模型 Running {model_name}")
            print(f"{'='*50}")
            
            try:
                if model_name == 'DeepAR':
                    metric, forecasts = self.deepar_forecast(train_data, test_data, tss)
                    
                elif model_name == 'DeepRenewal Flat':
                    metric, forecasts = self.deeprenewal_forecast(
                        train_data, test_data, tss, forecast_type='flat'
                    )
                    
                elif model_name == 'DeepRenewal Exact':
                    metric, forecasts = self.deeprenewal_forecast(
                        train_data, test_data, tss, forecast_type='exact'
                    )
                    
                elif model_name == 'DeepRenewal Hybrid':
                    metric, forecasts = self.deeprenewal_forecast(
                        train_data, test_data, tss, forecast_type='hybrid'
                    )
                    
                elif model_name == 'Croston':
                    metric, forecasts = self.croston_forecast(train_data, test_data, tss)
                    
                elif model_name == 'SBJ':
                    metric, forecasts = self.sbj_forecast(train_data, test_data, tss)
                    
                elif model_name == 'ARIMA':
                    metric, forecasts = self.arima_forecast(train_data, test_data, tss)
                    
                elif model_name == 'ETS':
                    metric, forecasts = self.ets_forecast(train_data, test_data, tss)
                    
                elif model_name == 'NPTS':
                    metric, forecasts = self.npts_forecast(train_data, test_data, tss)
                    
                else:
                    print(f"未知模型: {model_name} Unknown model: {model_name}")
                    continue
                
                results[model_name] = {
                    'metrics': metric,
                    'forecasts': forecasts
                }
                
                print(f"{model_name} 完成成功 completed successfully")
                
            except Exception as e:
                print(f"运行{model_name}时出错: {e} Error running {model_name}: {e}")
                results[model_name] = {
                    'metrics': None,
                    'forecasts': None,
                    'error': str(e)
                }
        
        print(f"\n所有模型运行完成 All models completed running")
        
        return results 