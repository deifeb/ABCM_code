"""
特征提取代理 (Agent 1) - Feature Extraction Agent

该代理负责从时间序列数据中提取不同类型的特征，并实现多种特征提取策略
This agent is responsible for extracting different types of features from time series data 
and implementing multiple feature extraction strategies

六种特征提取策略 Six Feature Extraction Strategies:
(1) S_raw: 只使用原始需求特征 Only raw demand features
(2) S_expert: 只使用专家衍生特征 Only expert-derived features  
(3) S_auto_opt: 只使用最合适维度的自编码器特征 Only autoencoder features with optimal dimensions
(4) S_auto_opt+expert: 结合最合适维度的自编码器特征和专家衍生特征 Combination of optimal autoencoder + expert features
(5) S_auto_multi+expert: 结合不同维度的自编码器特征和专家衍生特征 Combination of multi-dimensional autoencoder + expert features

动态调整机制 Dynamic Adjustment Mechanism:
- 基于下游代理的闭环反馈 Closed-loop feedback from downstream agents
- 当聚类评估指标<0.9时触发策略调整 Strategy adjustment triggered when clustering metric < 0.9
- 动态调整自编码器维度或重新激活原始特征 Dynamic adjustment of autoencoder dimensions or reactivation of raw features

作者 Author: ABCM Team
创建时间 Created: 2024
"""

import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
from scipy.stats import variation
import statsmodels.api as sm
import antropy as ant
from typing import Dict, List, Tuple, Optional, Union
import pickle
import json
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["OMP_NUM_THREADS"] = '3'


class FeatureExtractionAgent:
    """
    代理1：特征提取代理 - 支持六种特征提取策略
    Agent 1: Feature Extraction Agent - Supporting Six Feature Extraction Strategies
    
    该代理实现了多种特征提取策略，并具备动态调整机制
    This agent implements multiple feature extraction strategies with dynamic adjustment mechanism
    """
    
    def __init__(self, 
                 encoding_dim: int = 12, 
                 autoencoder_epochs: int = 50, 
                 batch_size: int = 16,
                 multi_dims: List[int] = [8, 12, 16],
                 feedback_threshold: float = 0.9):
        """
        初始化特征提取代理
        Initialize the Feature Extraction Agent
        
        参数 Args:
            encoding_dim (int): 默认自编码器潜在空间维度 Default autoencoder latent space dimension
            autoencoder_epochs (int): 自编码器训练轮数 Training epochs for autoencoder
            batch_size (int): 批处理大小 Batch size for autoencoder training
            multi_dims (List[int]): 多维度自编码器维度列表 Multi-dimensional autoencoder dimensions
            feedback_threshold (float): 反馈调整阈值 Feedback adjustment threshold
        """
        self.encoding_dim = encoding_dim
        self.autoencoder_epochs = autoencoder_epochs
        self.batch_size = batch_size
        self.multi_dims = multi_dims
        self.feedback_threshold = feedback_threshold
        
        # 策略配置 Strategy configuration
        self.strategies = {
            'S_raw': 'raw_demand_only',
            'S_expert': 'expert_features_only', 
            'S_auto_opt': 'autoencoder_optimal_only',
            'S_auto_opt+expert': 'autoencoder_optimal_plus_expert',
            'S_auto_multi+expert': 'autoencoder_multi_plus_expert'
        }
        
        self.current_strategy = 'S_auto_opt+expert'  # 默认策略 Default strategy
        self.strategy_history = []  # 策略调整历史 Strategy adjustment history
        self.performance_history = {}  # 性能历史记录 Performance history
        
        # 模型存储 Model storage
        self.autoencoders = {}  # 不同维度的自编码器 Autoencoders with different dimensions
        self.encoders = {}     # 对应的编码器 Corresponding encoders
        
        # 反馈机制 Feedback mechanism
        self.feedback_signals = []
        self.adjustment_count = 0
        
        # 设置随机种子以确保可重现性 Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
        print("=" * 80)
        print("🚀 特征提取代理初始化完成 Feature Extraction Agent Initialized")
        print("=" * 80)
        print(f"默认策略 Default strategy: {self.current_strategy}")
        print(f"支持的维度 Supported dimensions: {self.multi_dims}")
        print(f"反馈阈值 Feedback threshold: {self.feedback_threshold}")
    
    def extract_raw_demand_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        策略 S_raw: 提取原始需求特征
        Strategy S_raw: Extract raw demand features
        
        这种策略直接使用原始需求数据，保持数据的简单性和直观性
        This strategy directly uses raw demand data, maintaining simplicity and intuitiveness
        
        参数 Args:
            data (pd.DataFrame): 输入时间序列数据 Input time series data
            
        返回 Returns:
            pd.DataFrame: 原始需求特征 Raw demand features
        """
        print("🔸 执行策略 S_raw: 提取原始需求特征")
        print("🔸 Executing Strategy S_raw: Extracting raw demand features")
        
        # 使用原始时间序列的统计特征 Use statistical features of raw time series
        raw_features = []
        
        for col in data.columns:
            series = data[col].values
            feature_dict = {
                'mean': np.mean(series),
                'std': np.std(series),
                'max': np.max(series),
                'min': np.min(series),
                'median': np.median(series),
                'skewness': pd.Series(series).skew(),
                'kurtosis': pd.Series(series).kurtosis(),
                'variance': np.var(series),
                'range': np.max(series) - np.min(series),
                'sum': np.sum(series),
                'non_zero_count': np.count_nonzero(series),
                'zero_count': np.sum(series == 0),
                'last_value': series[-1],
                'first_value': series[0],
                'trend_slope': np.polyfit(range(len(series)), series, 1)[0] if len(series) > 1 else 0
            }
            raw_features.append(feature_dict)
        
        raw_features_df = pd.DataFrame(raw_features)
        print(f"✅ 原始特征提取完成，特征数: {raw_features_df.shape[1]}")
        print(f"✅ Raw features extracted, feature count: {raw_features_df.shape[1]}")
        
        return raw_features_df
    
    def compute_expert_features(self, time_series: np.ndarray) -> Dict[str, float]:
        """
        计算专家知识特征(F1-F9)
        Compute expert knowledge features (F1-F9)
        
        这些特征是基于间歇性需求预测领域的专家知识设计的
        These features are designed based on expert knowledge in intermittent demand forecasting
        
        参数 Args:
            time_series (np.ndarray): 时间序列数据 Time series data
            
        返回 Returns:
            Dict[str, float]: 包含计算出的特征的字典 Dictionary containing computed features
        """
        features = {}

        # F1: 平均需求间隔 (ADI - Average Demand Interval)
        demand_indices = np.where(time_series > 0)[0]
        if len(demand_indices) > 1:
            inter_demand_intervals = np.diff(demand_indices)
            features['F1'] = np.mean(inter_demand_intervals)
        else:
            features['F1'] = np.nan

        # F2: 变异系数的平方 (CV² - Square of Coefficient of Variation)
        non_zero_demand = time_series[time_series > 0]
        if len(non_zero_demand) > 0:
            features['F2'] = variation(non_zero_demand) ** 2
        else:
            features['F2'] = np.nan

        # F3: 近似熵 (Approximate Entropy)
        try:
            features['F3'] = ant.app_entropy(time_series, order=2)
        except:
            features['F3'] = 0.0

        # F4: 零值百分比 (Percentage of Zero Values)
        features['F4'] = np.sum(time_series == 0) / len(time_series)

        # F5: 超出[均值±标准差]范围的值的百分比
        mean_y = np.mean(time_series)
        std_y = np.std(time_series)
        features['F5'] = np.sum((time_series < mean_y - std_y) | 
                               (time_series > mean_y + std_y)) / len(time_series)

        # F6: 线性最小二乘回归系数
        chunk_size = 12
        chunks = [time_series[i:i + chunk_size] 
                 for i in range(0, len(time_series), chunk_size)]
        variances = [np.var(chunk) for chunk in chunks if len(chunk) == chunk_size]
        
        if len(variances) > 1:
            try:
                x = np.arange(len(variances))
                X = sm.add_constant(x)
                model = sm.OLS(variances, X).fit()
                features['F6'] = model.params[1]
            except:
                features['F6'] = 0.0
        else:
            features['F6'] = np.nan

        # F7: 连续变化的平均绝对值
        consecutive_changes = np.diff(time_series)
        features['F7'] = np.mean(np.abs(consecutive_changes))

        # F8: 最后一个块的平方和占总序列的比例
        k = 4
        chunk_length = len(time_series) // k
        last_chunk = time_series[-chunk_length:]
        total_sum_squares = np.sum(time_series ** 2)
        features['F8'] = np.sum(last_chunk ** 2) / total_sum_squares if total_sum_squares > 0 else 0

        # F9: 序列末尾连续零值的百分比
        consecutive_zero_at_end = 0
        for value in reversed(time_series):
            if value == 0:
                consecutive_zero_at_end += 1
            else:
                break
        features['F9'] = consecutive_zero_at_end / len(time_series)

        return features
    
    def extract_expert_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        策略 S_expert: 提取专家衍生特征
        Strategy S_expert: Extract expert-derived features
        
        参数 Args:
            data (pd.DataFrame): 输入时间序列数据 Input time series data
            
        返回 Returns:
            pd.DataFrame: 专家特征数据框 Expert features DataFrame
        """
        print("🔹 执行策略 S_expert: 提取专家衍生特征")
        print("🔹 Executing Strategy S_expert: Extracting expert-derived features")
        
        expert_features = []
        for col in data.columns:
            features = self.compute_expert_features(data[col].values)
            expert_features.append(features)
        
        expert_features_df = pd.DataFrame(expert_features)
        print(f"✅ 专家特征提取完成，特征数: {expert_features_df.shape[1]}")
        print(f"✅ Expert features extracted, feature count: {expert_features_df.shape[1]}")
        
        return expert_features_df
    
    def build_autoencoder(self, input_shape: Tuple[int], encoding_dim: int) -> Tuple[tf.keras.Model, tf.keras.Model]:
        """
        构建指定维度的自编码器
        Build autoencoder with specified dimension
        
        参数 Args:
            input_shape (Tuple[int]): 输入数据的形状 Shape of input data
            encoding_dim (int): 编码维度 Encoding dimension
            
        返回 Returns:
            Tuple[tf.keras.Model, tf.keras.Model]: (autoencoder, encoder) 自编码器和编码器模型
        """
        input_layer = tf.keras.layers.Input(shape=input_shape)
        
        # 编码器部分 Encoder part
        encoded = tf.keras.layers.Dense(64, activation='relu')(input_layer)
        encoded = tf.keras.layers.Dense(32, activation='relu')(encoded)
        encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(encoded)
        
        # 解码器部分 Decoder part  
        decoded = tf.keras.layers.Dense(32, activation='relu')(encoded)
        decoded = tf.keras.layers.Dense(64, activation='relu')(decoded)
        decoded = tf.keras.layers.Dense(input_shape[0], activation='sigmoid')(decoded)
        
        autoencoder = tf.keras.models.Model(input_layer, decoded)
        encoder = tf.keras.models.Model(input_layer, encoded)
        
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder, encoder
    
    def extract_autoencoder_features(self, data: pd.DataFrame, encoding_dim: int) -> pd.DataFrame:
        """
        使用指定维度的自编码器提取特征
        Extract features using autoencoder with specified dimension
        
        参数 Args:
            data (pd.DataFrame): 输入时间序列数据 Input time series data
            encoding_dim (int): 编码维度 Encoding dimension
            
        返回 Returns:
            pd.DataFrame: 自编码器特征 Autoencoder features
        """
        input_data = data.T.values
        sequence_length = input_data.shape[1]
        input_shape = (sequence_length,)
        
        # 检查是否已有该维度的模型 Check if model with this dimension already exists
        dim_key = f"dim_{encoding_dim}"
        if dim_key not in self.autoencoders:
            print(f"🔧 构建 {encoding_dim} 维自编码器...")
            print(f"🔧 Building {encoding_dim}-dimension autoencoder...")
            
            autoencoder, encoder = self.build_autoencoder(input_shape, encoding_dim)
            
            print(f"🔄 训练 {encoding_dim} 维自编码器...")
            print(f"🔄 Training {encoding_dim}-dimension autoencoder...")
            
            autoencoder.fit(
                input_data, input_data,
                epochs=self.autoencoder_epochs,
                batch_size=self.batch_size,
                verbose=0
            )
            
            self.autoencoders[dim_key] = autoencoder
            self.encoders[dim_key] = encoder
        
        # 提取特征 Extract features
        encoder = self.encoders[dim_key]
        encoded_data = encoder.predict(input_data, verbose=0)
        
        feature_columns = [f'AutoEnc_{encoding_dim}D_F{i}' for i in range(1, encoding_dim + 1)]
        autoencoder_features = pd.DataFrame(encoded_data, columns=feature_columns)
        
        return autoencoder_features
    
    def extract_autoencoder_optimal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        策略 S_auto_opt: 提取最合适维度的自编码器特征
        Strategy S_auto_opt: Extract autoencoder features with optimal dimensions
        
        参数 Args:
            data (pd.DataFrame): 输入时间序列数据 Input time series data
            
        返回 Returns:
            pd.DataFrame: 最优维度自编码器特征 Optimal autoencoder features
        """
        print("🔹 执行策略 S_auto_opt: 提取最合适维度的自编码器特征")
        print("🔹 Executing Strategy S_auto_opt: Extracting optimal autoencoder features")
        
        return self.extract_autoencoder_features(data, self.encoding_dim)
    
    def extract_autoencoder_multi_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        提取多维度自编码器特征
        Extract multi-dimensional autoencoder features
        
        参数 Args:
            data (pd.DataFrame): 输入时间序列数据 Input time series data
            
        返回 Returns:
            pd.DataFrame: 多维度自编码器特征 Multi-dimensional autoencoder features
        """
        print("🔹 提取多维度自编码器特征...")
        print("🔹 Extracting multi-dimensional autoencoder features...")
        
        all_features = []
        for dim in self.multi_dims:
            features = self.extract_autoencoder_features(data, dim)
            all_features.append(features)
        
        multi_features = pd.concat(all_features, axis=1)
        print(f"✅ 多维度自编码器特征提取完成，总特征数: {multi_features.shape[1]}")
        print(f"✅ Multi-dimensional autoencoder features extracted, total features: {multi_features.shape[1]}")
        
        return multi_features
    
    def extract_features_by_strategy(self, data: pd.DataFrame, strategy: str = None) -> pd.DataFrame:
        """
        根据指定策略提取特征
        Extract features according to specified strategy
        
        参数 Args:
            data (pd.DataFrame): 输入时间序列数据 Input time series data
            strategy (str): 特征提取策略，如果为None则使用当前策略 Feature extraction strategy
            
        返回 Returns:
            pd.DataFrame: 提取的特征 Extracted features
        """
        if strategy is None:
            strategy = self.current_strategy
        
        print("=" * 80)
        print(f"🎯 执行特征提取策略: {strategy}")
        print(f"🎯 Executing feature extraction strategy: {strategy}")
        print("=" * 80)
        
        features_list = []
        
        if strategy == 'S_raw':
            # 策略1: 只使用原始需求特征
            features_list.append(self.extract_raw_demand_features(data))
            
        elif strategy == 'S_expert':
            # 策略2: 只使用专家衍生特征
            features_list.append(self.extract_expert_features(data))
            
        elif strategy == 'S_auto_opt':
            # 策略3: 只使用最合适维度的自编码器特征
            features_list.append(self.extract_autoencoder_optimal_features(data))
            
        elif strategy == 'S_auto_opt+expert':
            # 策略4: 结合最合适维度的自编码器特征和专家衍生特征
            features_list.append(self.extract_autoencoder_optimal_features(data))
            features_list.append(self.extract_expert_features(data))
            
        elif strategy == 'S_auto_multi+expert':
            # 策略5: 结合不同维度的自编码器特征和专家衍生特征
            features_list.append(self.extract_autoencoder_multi_features(data))
            features_list.append(self.extract_expert_features(data))
            
        else:
            raise ValueError(f"未知的策略: {strategy}. 支持的策略: {list(self.strategies.keys())}")
        
        # 合并特征 Combine features
        if len(features_list) == 1:
            final_features = features_list[0]
        else:
            final_features = pd.concat(features_list, axis=1)
        
        # 记录策略使用 Record strategy usage
        self.strategy_history.append({
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'feature_count': final_features.shape[1],
            'data_shape': data.shape
        })
        
        print("=" * 80)
        print(f"✅ 特征提取完成 Feature extraction completed")
        print(f"📊 使用策略: {strategy}")
        print(f"📊 Strategy used: {strategy}")
        print(f"📈 总特征数: {final_features.shape[1]}")
        print(f"📈 Total features: {final_features.shape[1]}")
        print(f"📋 数据形状: {data.shape} -> {final_features.shape}")
        print(f"📋 Data shape: {data.shape} -> {final_features.shape}")
        print("=" * 80)
        
        return final_features
    
    def receive_feedback(self, feedback_signal: Dict[str, Union[float, str, Dict]]):
        """
        接收来自下游代理的反馈信号
        Receive feedback signal from downstream agents
        
        参数 Args:
            feedback_signal (Dict): 反馈信号
                - clustering_score (float): 聚类评估分数
                - forecasting_errors (Dict): 各类别的预测误差
                - suggestion (str): 调整建议
                - category_performance (Dict): 各类别性能
        """
        print("\n" + "🔄" * 60)
        print("接收反馈信号 Receiving feedback signal")
        print("🔄" * 60)
        
        self.feedback_signals.append({
            'timestamp': datetime.now().isoformat(),
            'signal': feedback_signal,
            'current_strategy': self.current_strategy
        })
        
        clustering_score = feedback_signal.get('clustering_score', 1.0)
        forecasting_errors = feedback_signal.get('forecasting_errors', {})
        suggestion = feedback_signal.get('suggestion', '')
        
        print(f"📊 聚类评估分数: {clustering_score:.3f}")
        print(f"📊 Clustering score: {clustering_score:.3f}")
        print(f"💡 调整建议: {suggestion}")
        print(f"💡 Suggestion: {suggestion}")
        
        # 判断是否需要调整策略 Determine if strategy adjustment is needed
        if clustering_score < self.feedback_threshold:
            print(f"⚠️ 聚类分数 {clustering_score:.3f} 低于阈值 {self.feedback_threshold}")
            print(f"⚠️ Clustering score {clustering_score:.3f} below threshold {self.feedback_threshold}")
            self.adjust_strategy(feedback_signal)
        
        # 检查预测误差 Check forecasting errors
        if forecasting_errors:
            high_error_categories = [cat for cat, error in forecasting_errors.items() 
                                   if error > 5.0]  # 假设误差阈值为5.0
            if high_error_categories:
                print(f"⚠️ 高误差类别: {high_error_categories}")
                print(f"⚠️ High error categories: {high_error_categories}")
                self.adjust_for_high_errors(high_error_categories, forecasting_errors)
    
    def adjust_strategy(self, feedback_signal: Dict[str, Union[float, str, Dict]]):
        """
        根据反馈调整特征提取策略
        Adjust feature extraction strategy based on feedback
        
        参数 Args:
            feedback_signal (Dict): 反馈信号 Feedback signal
        """
        print("\n" + "🔧" * 50)
        print("执行策略调整 Executing strategy adjustment")
        print("🔧" * 50)
        
        old_strategy = self.current_strategy
        suggestion = feedback_signal.get('suggestion', '')
        
        # 策略调整逻辑 Strategy adjustment logic
        if suggestion == 'increase_expert_weight':
            # 增加专家知识特征的权重 Increase weight of expert knowledge features
            if self.current_strategy == 'S_auto_opt':
                self.current_strategy = 'S_auto_opt+expert'
            elif self.current_strategy == 'S_raw':
                self.current_strategy = 'S_expert'
            
        elif suggestion == 'try_multi_dimensional':
            # 尝试多维度自编码器 Try multi-dimensional autoencoder
            if 'multi' not in self.current_strategy:
                self.current_strategy = 'S_auto_multi+expert'
        
        elif suggestion == 'simplify_features':
            # 简化特征 Simplify features
            if 'multi' in self.current_strategy:
                self.current_strategy = 'S_auto_opt+expert'
            elif 'auto_opt+expert' in self.current_strategy:
                self.current_strategy = 'S_expert'
            
        elif suggestion == 'try_raw_features':
            # 尝试原始特征 Try raw features
            self.current_strategy = 'S_raw'
            
        else:
            # 默认策略序列 Default strategy sequence
            strategy_sequence = ['S_auto_opt+expert', 'S_auto_multi+expert', 'S_expert', 'S_auto_opt', 'S_raw']
            current_idx = strategy_sequence.index(self.current_strategy) if self.current_strategy in strategy_sequence else 0
            next_idx = (current_idx + 1) % len(strategy_sequence)
            self.current_strategy = strategy_sequence[next_idx]
        
        self.adjustment_count += 1
        
        print(f"🔄 策略调整: {old_strategy} -> {self.current_strategy}")
        print(f"🔄 Strategy adjustment: {old_strategy} -> {self.current_strategy}")
        print(f"📊 调整次数: {self.adjustment_count}")
        print(f"📊 Adjustment count: {self.adjustment_count}")
    
    def adjust_for_high_errors(self, high_error_categories: List[str], forecasting_errors: Dict[str, float]):
        """
        针对高误差类别调整参数
        Adjust parameters for high error categories
        
        参数 Args:
            high_error_categories (List[str]): 高误差类别列表 List of high error categories
            forecasting_errors (Dict[str, float]): 预测误差字典 Forecasting errors dictionary
        """
        print(f"\n🎯 针对高误差类别进行参数调整")
        print(f"🎯 Adjusting parameters for high error categories")
        
        # 动态调整自编码器维度 Dynamically adjust autoencoder dimensions
        if len(high_error_categories) > len(self.multi_dims) // 2:
            # 如果大部分类别都有高误差，增加编码维度
            self.encoding_dim = min(20, self.encoding_dim + 2)
            print(f"🔧 增加编码维度至: {self.encoding_dim}")
            print(f"🔧 Increased encoding dimension to: {self.encoding_dim}")
            
            # 重新构建自编码器 Rebuild autoencoder
            dim_key = f"dim_{self.encoding_dim}"
            if dim_key in self.autoencoders:
                del self.autoencoders[dim_key]
                del self.encoders[dim_key]
        
        # 考虑重新激活原始需求特征 Consider reactivating raw demand features
        avg_error = np.mean(list(forecasting_errors.values()))
        if avg_error > 10.0:  # 如果平均误差很高
            print("🔄 误差过高，考虑重新激活原始需求特征")
            print("🔄 Error too high, considering reactivation of raw demand features")
            self.current_strategy = 'S_raw'
    
    def get_strategy_performance_summary(self) -> Dict[str, any]:
        """
        获取策略性能摘要
        Get strategy performance summary
        
        返回 Returns:
            Dict[str, any]: 策略性能摘要 Strategy performance summary
        """
        summary = {
            'current_strategy': self.current_strategy,
            'total_adjustments': self.adjustment_count,
            'strategy_history': self.strategy_history[-10:],  # 最近10次记录
            'feedback_count': len(self.feedback_signals),
            'encoding_dim': self.encoding_dim,
            'multi_dims': self.multi_dims,
            'available_models': list(self.autoencoders.keys())
        }
        
        return summary
    
    def save_agent_state(self, filepath: str):
        """
        保存代理状态
        Save agent state
        
        参数 Args:
            filepath (str): 保存文件路径 Save file path
        """
        state = {
            'current_strategy': self.current_strategy,
            'encoding_dim': self.encoding_dim,
            'autoencoder_epochs': self.autoencoder_epochs,
            'batch_size': self.batch_size,
            'multi_dims': self.multi_dims,
            'feedback_threshold': self.feedback_threshold,
            'strategy_history': self.strategy_history,
            'feedback_signals': self.feedback_signals,
            'adjustment_count': self.adjustment_count,
            'performance_history': self.performance_history
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        print(f"💾 代理状态已保存至: {filepath}")
        print(f"💾 Agent state saved to: {filepath}")
    
    def load_agent_state(self, filepath: str):
        """
        加载代理状态
        Load agent state
        
        参数 Args:
            filepath (str): 加载文件路径 Load file path
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        self.current_strategy = state.get('current_strategy', 'S_auto_opt+expert')
        self.encoding_dim = state.get('encoding_dim', 12)
        self.autoencoder_epochs = state.get('autoencoder_epochs', 50)
        self.batch_size = state.get('batch_size', 16)
        self.multi_dims = state.get('multi_dims', [8, 12, 16])
        self.feedback_threshold = state.get('feedback_threshold', 0.9)
        self.strategy_history = state.get('strategy_history', [])
        self.feedback_signals = state.get('feedback_signals', [])
        self.adjustment_count = state.get('adjustment_count', 0)
        self.performance_history = state.get('performance_history', {})
        
        print(f"📁 代理状态已加载: {filepath}")
        print(f"📁 Agent state loaded from: {filepath}")
        print(f"🎯 当前策略: {self.current_strategy}")
        print(f"🎯 Current strategy: {self.current_strategy}")


def demo_feature_extraction_agent():
    """
    演示特征提取代理的功能
    Demonstrate the functionality of Feature Extraction Agent
    """
    print("🚀 特征提取代理演示 Feature Extraction Agent Demo")
    print("=" * 80)
    
    # 创建模拟数据 Create simulation data
    np.random.seed(42)
    n_series = 100
    n_periods = 84
    
    data = pd.DataFrame()
    for i in range(n_series):
        # 生成间歇性需求数据 Generate intermittent demand data
        demand = np.random.poisson(1.5, n_periods)
        demand = np.where(np.random.random(n_periods) > 0.7, demand, 0)
        data[f'item_{i:03d}'] = demand
    
    print(f"📊 生成模拟数据: {data.shape}")
    print(f"📊 Generated simulation data: {data.shape}")
    
    # 初始化代理 Initialize agent
    agent = FeatureExtractionAgent(
        encoding_dim=12,
        autoencoder_epochs=5,  # 减少训练时间用于演示
        multi_dims=[8, 12, 16]
    )
    
    # 测试所有策略 Test all strategies
    for strategy in agent.strategies.keys():
        print(f"\n{'='*60}")
        print(f"🧪 测试策略: {strategy}")
        print(f"🧪 Testing strategy: {strategy}")
        print(f"{'='*60}")
        
        features = agent.extract_features_by_strategy(data, strategy)
        print(f"📈 提取特征形状: {features.shape}")
        print(f"📈 Extracted features shape: {features.shape}")
    
    # 模拟反馈调整 Simulate feedback adjustment
    print(f"\n{'='*60}")
    print("🔄 模拟反馈调整 Simulating feedback adjustment")
    print(f"{'='*60}")
    
    # 模拟低聚类分数的反馈 Simulate low clustering score feedback
    feedback = {
        'clustering_score': 0.75,  # 低于阈值0.9
        'forecasting_errors': {'category_1': 7.5, 'category_2': 3.2},
        'suggestion': 'increase_expert_weight'
    }
    
    agent.receive_feedback(feedback)
    
    # 使用调整后的策略提取特征 Extract features with adjusted strategy
    adjusted_features = agent.extract_features_by_strategy(data)
    print(f"📊 调整后特征形状: {adjusted_features.shape}")
    print(f"📊 Adjusted features shape: {adjusted_features.shape}")
    
    # 显示性能摘要 Show performance summary
    summary = agent.get_strategy_performance_summary()
    print(f"\n📋 性能摘要 Performance Summary:")
    print(f"   当前策略 Current strategy: {summary['current_strategy']}")
    print(f"   调整次数 Total adjustments: {summary['total_adjustments']}")
    print(f"   反馈次数 Feedback count: {summary['feedback_count']}")
    
    # 保存代理状态 Save agent state
    agent.save_agent_state("feature_extraction_agent_demo_state.json")
    
    print("\n✅ 演示完成 Demo completed!")


if __name__ == "__main__":
    demo_feature_extraction_agent() 