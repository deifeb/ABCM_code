"""
ABCM主系统 - ABCM Main System

基于代理的协作模型主系统，协调三个代理进行间歇性需求预测
Agent-Based Collaborative Model main system that coordinates three agents 
for intermittent demand forecasting

系统架构 System Architecture:
┌─────────────────┐    特征    ┌─────────────────┐    聚类标签    ┌─────────────────┐
│   代理1         │  ────────> │   代理2         │  ─────────>   │   代理3         │
│ 特征提取代理     │   Features │ 分类代理        │  Cluster      │ 模型选择代理     │
│ Feature         │            │ Classification  │  Labels       │ Model Selection │
│ Extraction      │            │ Agent          │               │ Agent          │
└─────────────────┘            └─────────────────┘               └─────────────────┘
       ↑                              ↑                              ↑
       │        反馈循环 Feedback Loop │                              │
       └──────────────────────────────┴──────────────────────────────┘

主要功能 Main Functions:
- 协调三个代理的协作 Coordinate collaboration of three agents
- 管理迭代反馈过程 Manage iterative feedback process
- 提供系统级接口 Provide system-level interface
- 处理模型训练和预测 Handle model training and prediction

作者 Author: ABCM Team
创建时间 Created: 2024
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from scipy import stats

from agents import FeatureExtractionAgent, ClassificationAgent, ModelSelectionAgent
from models import ForecastingModels
from utils import FileUtils
from config import get_config, ABCMConfig

warnings.filterwarnings("ignore")


class ABCMSystem:
    """
    ABCM主系统类
    Main ABCM System class
    
    该类是整个ABCM系统的核心，负责协调三个代理进行协作式间歇性需求预测
    This class is the core of the entire ABCM system, responsible for coordinating 
    three agents for collaborative intermittent demand forecasting
    
    系统特点 System Features:
    1. 代理协作 - 三个专门化代理的智能协作
       Agent Collaboration - Intelligent collaboration of three specialized agents
       
    2. 迭代优化 - 通过反馈机制不断改进预测性能
       Iterative Optimization - Continuously improve prediction performance through feedback
       
    3. 自适应学习 - 根据数据特性自动调整策略
       Adaptive Learning - Automatically adjust strategies based on data characteristics
       
    4. 模块化设计 - 每个组件都可以独立使用和扩展
       Modular Design - Each component can be used and extended independently
       
    5. 智能数据预处理 - 基于修正z-score的自适应异常值处理
       Intelligent Data Preprocessing - Adaptive outlier handling based on modified z-score
    """
    
    def __init__(self, config: Optional[ABCMConfig] = None, pretrained_metalearner_path: Optional[str] = None):
        """
        初始化ABCM系统
        Initialize the ABCM system
        
        参数 Args:
            config (ABCMConfig): 配置对象 Configuration object
            pretrained_metalearner_path (str): 预训练元学习器路径 Path to pretrained meta-learner from Cross-validation training
        """
        self.config = config or get_config()
        
        print("初始化ABCM系统... Initializing ABCM system...")
        
        # 初始化三个代理 Initialize three agents
        self.agent1 = FeatureExtractionAgent(
            encoding_dim=self.config.feature_extraction.encoding_dim,
            autoencoder_epochs=self.config.feature_extraction.autoencoder_epochs,
            batch_size=self.config.feature_extraction.batch_size
        )
        print("代理1(特征提取代理)初始化完成 Agent 1 (Feature Extraction) initialized")
        
        self.agent2 = ClassificationAgent(
            accuracy_threshold=self.config.classification.accuracy_threshold,
            max_clusters=self.config.classification.max_clusters
        )
        print("代理2(分类代理)初始化完成 Agent 2 (Classification) initialized")
        
        # 初始化代理3，支持预训练元学习器 Initialize Agent 3 with pretrained meta-learner support
        self.agent3 = ModelSelectionAgent(
            error_threshold=self.config.model_selection.error_threshold,
            models_list=self.config.system.candidate_models,
            pretrained_metalearner_path=pretrained_metalearner_path
        )
        print("代理3(模型选择代理)初始化完成 Agent 3 (Model Selection) initialized")
        
        # 初始化预测模型容器 Initialize forecasting models container
        self.forecasting_models = ForecastingModels(
            prediction_length=self.config.forecasting.prediction_length,
            freq=self.config.forecasting.freq
        )
        print("预测模型容器初始化完成 Forecasting models container initialized")
        
        # 系统状态变量 System state variables
        self.features = None
        self.cluster_labels = None
        self.model_errors = None
        self.best_models = None
        self.iteration_count = 0
        self.converged = False
        
        # 数据预处理状态 Data preprocessing state
        self.outlier_info = {}  # 存储异常值处理信息
        self.preprocessed_data = None  # 存储预处理后的数据
        
        # 存储预训练元学习器路径信息 Store pretrained meta-learner path info
        self.pretrained_metalearner_path = pretrained_metalearner_path
        
        print("ABCM系统初始化完成！ ABCM system initialization completed!")
        if pretrained_metalearner_path and self.agent3.use_pretrained:
            print(f"🎯 系统将使用预训练元学习器进行模型选择 System will use pretrained meta-learner for model selection")
        else:
            print(f"⚙️  系统将使用实时元学习进行模型选择 System will use real-time meta-learning for model selection")
    
    def calculate_adaptive_threshold(self, non_zero_data: np.ndarray) -> float:
        """
        计算自适应z-score阈值
        Calculate adaptive z-score threshold
        
        基于非零需求变异系数平方与临界值0.49的关系来动态确定z-score阈值
        The Z-score threshold is dynamically determined according to the squared coefficient of variation 
        of non-zero demand observations and its relationship with the critical value of 0.49
        
        参数 Args:
            non_zero_data (np.ndarray): 非零需求数据 Non-zero demand data
            
        返回 Returns:
            float: 自适应阈值 Adaptive threshold
        """
        if len(non_zero_data) == 0:
            return 2.5  # 默认阈值 Default threshold
        
        # 计算变异系数 Calculate coefficient of variation
        mean_val = np.mean(non_zero_data)
        std_val = np.std(non_zero_data)
        
        if mean_val == 0:
            return 2.5  # 默认阈值 Default threshold
        
        cv = std_val / mean_val  # 变异系数 Coefficient of variation
        cv_squared = cv ** 2  # 变异系数平方 Squared coefficient of variation
        
        # 基于与临界值0.49的关系计算阈值
        # Calculate threshold based on relationship with critical value of 0.49
        critical_value = 0.49
        
        if cv_squared <= critical_value:
            # 低变异性数据，使用较低阈值(更严格的异常值检测)
            # Low variability data, use lower threshold (stricter outlier detection)
            threshold = 2.0 + (cv_squared / critical_value) * 0.5  # 2.0 to 2.5
        else:
            # 高变异性数据，使用较高阈值(更宽松的异常值检测)
            # High variability data, use higher threshold (more lenient outlier detection)
            excess_ratio = cv_squared / critical_value
            threshold = 2.5 + min(excess_ratio - 1, 1.0) * 1.0  # 2.5 to 3.5
        
        return threshold
    
    def detect_outliers_modified_zscore(self, data: pd.Series, strategy: str = 'strategy1') -> Tuple[np.ndarray, Dict]:
        """
        使用修正z-score检测异常值
        Detect outliers using modified z-score
        
        参数 Args:
            data (pd.Series): 时间序列数据 Time series data
            strategy (str): 异常值处理策略 Outlier handling strategy
                          'strategy1': 使用非零需求的MAD和标准差，中位数替换
                          'strategy2': 使用非零需求均值，其他同strategy1
                          'strategy3': 不做异常值处理
                          
        返回 Returns:
            tuple: (outlier_mask, info_dict) 异常值掩码和信息字典
        """
        info_dict = {
            'strategy': strategy,
            'total_observations': len(data),
            'zero_observations': (data == 0).sum(),
            'non_zero_observations': (data > 0).sum(),
            'outliers_detected': 0,
            'outliers_replaced': 0,
            'threshold_used': None,
            'cv_squared': None
        }
        
        if strategy == 'strategy3':
            # 策略3：不做处理 Strategy 3: No processing
            return np.zeros(len(data), dtype=bool), info_dict
        
        # 提取非零需求数据 Extract non-zero demand data
        non_zero_mask = data > 0
        non_zero_data = data[non_zero_mask].values
        
        if len(non_zero_data) == 0:
            # 没有非零数据，无需处理 No non-zero data, no processing needed
            return np.zeros(len(data), dtype=bool), info_dict
        
        # 计算自适应阈值 Calculate adaptive threshold
        threshold = self.calculate_adaptive_threshold(non_zero_data)
        info_dict['threshold_used'] = threshold
        
        # 计算变异系数平方用于记录 Calculate squared CV for recording
        if np.mean(non_zero_data) > 0:
            cv_squared = (np.std(non_zero_data) / np.mean(non_zero_data)) ** 2
            info_dict['cv_squared'] = cv_squared
        
        # 根据策略计算参数 Calculate parameters based on strategy
        if strategy == 'strategy1':
            # 策略1：使用MAD作为替代均值，非零需求标准差，中位数替换
            # Strategy 1: Use MAD as substitute mean, non-zero std, median replacement
            mad = np.median(np.abs(non_zero_data - np.median(non_zero_data)))  # MAD
            substitute_mean = mad
            substitute_std = np.std(non_zero_data)
            replacement_value = np.median(non_zero_data)
            
        elif strategy == 'strategy2':
            # 策略2：使用非零需求均值，其他同strategy1
            # Strategy 2: Use non-zero mean, others same as strategy1
            mad = np.median(np.abs(non_zero_data - np.median(non_zero_data)))  # MAD
            substitute_mean = np.mean(non_zero_data)
            substitute_std = np.std(non_zero_data)
            replacement_value = np.median(non_zero_data)
        
        else:
            raise ValueError(f"未知的异常值处理策略: {strategy}")
        
        # 计算修正z-score (仅对非零数据) Calculate modified z-score (only for non-zero data)
        if substitute_std == 0:
            # 标准差为0，无法计算z-score Standard deviation is 0, cannot calculate z-score
            outlier_mask = np.zeros(len(data), dtype=bool)
        else:
            # 初始化异常值掩码 Initialize outlier mask
            outlier_mask = np.zeros(len(data), dtype=bool)
            
            # 仅对非零数据计算z-score Calculate z-score only for non-zero data
            z_scores = np.abs(non_zero_data - substitute_mean) / substitute_std
            non_zero_outliers = z_scores > threshold
            
            # 将非零数据的异常值映射回原始数据 Map non-zero outliers back to original data
            non_zero_indices = np.where(non_zero_mask)[0]
            outlier_mask[non_zero_indices[non_zero_outliers]] = True
        
        info_dict['outliers_detected'] = np.sum(outlier_mask)
        
        return outlier_mask, info_dict
    
    def handle_outliers(self, data: pd.DataFrame, strategy: str = 'strategy1') -> Tuple[pd.DataFrame, Dict]:
        """
        处理整个数据集的异常值
        Handle outliers for the entire dataset
        
        参数 Args:
            data (pd.DataFrame): 输入时间序列数据 Input time series data
            strategy (str): 异常值处理策略 Outlier handling strategy
                          'strategy1': MAD + 中位数替换
                          'strategy2': 均值 + 中位数替换  
                          'strategy3': 不处理
                          
        返回 Returns:
            tuple: (processed_data, processing_info) 处理后的数据和处理信息
        """
        print("\n" + "=" * 70)
        print("🔧 数据预处理：异常值处理 DATA PREPROCESSING: OUTLIER HANDLING")
        print("=" * 70)
        print(f"使用策略 Using strategy: {strategy}")
        
        if strategy == 'strategy3':
            print("策略3：不进行异常值处理 Strategy 3: No outlier processing")
            processing_info = {
                'strategy': strategy,
                'total_spare_parts': data.shape[1],
                'total_outliers_detected': 0,
                'total_outliers_replaced': 0,
                'spare_parts_info': {}
            }
            return data.copy(), processing_info
        
        processed_data = data.copy()
        processing_info = {
            'strategy': strategy,
            'total_spare_parts': data.shape[1],
            'total_outliers_detected': 0,
            'total_outliers_replaced': 0,
            'spare_parts_info': {}
        }
        
        print(f"处理 {data.shape[1]} 个备件的时间序列数据...")
        print(f"Processing time series data for {data.shape[1]} spare parts...")
        
        outliers_by_part = []
        
        # 逐个备件处理 Process each spare part individually
        for col in data.columns:
            series = data[col]
            
            # 检测异常值 Detect outliers
            outlier_mask, part_info = self.detect_outliers_modified_zscore(series, strategy)
            
            if np.any(outlier_mask):
                # 计算替换值(中位数) Calculate replacement value (median)
                non_zero_data = series[series > 0]
                if len(non_zero_data) > 0:
                    replacement_value = np.median(non_zero_data)
                    
                    # 替换异常值 Replace outliers
                    processed_data.loc[outlier_mask, col] = replacement_value
                    part_info['outliers_replaced'] = np.sum(outlier_mask)
                    part_info['replacement_value'] = replacement_value
                else:
                    part_info['outliers_replaced'] = 0
                    part_info['replacement_value'] = None
            
            # 存储备件信息 Store spare part info
            processing_info['spare_parts_info'][col] = part_info
            processing_info['total_outliers_detected'] += part_info['outliers_detected']
            processing_info['total_outliers_replaced'] += part_info['outliers_replaced']
            
            outliers_by_part.append(part_info['outliers_detected'])
        
        # 打印处理结果摘要 Print processing summary
        print(f"\n📊 异常值处理摘要 Outlier Processing Summary:")
        print(f"总异常值检测数 Total outliers detected: {processing_info['total_outliers_detected']}")
        print(f"总异常值替换数 Total outliers replaced: {processing_info['total_outliers_replaced']}")
        print(f"异常值比例 Outlier ratio: {processing_info['total_outliers_detected']/(data.shape[0]*data.shape[1])*100:.2f}%")
        
        if processing_info['total_outliers_detected'] > 0:
            print(f"每个备件平均异常值数 Average outliers per spare part: {np.mean(outliers_by_part):.1f}")
            print(f"异常值数量分布 Outlier count distribution:")
            print(f"  最小值 Min: {np.min(outliers_by_part)}")
            print(f"  最大值 Max: {np.max(outliers_by_part)}")
            print(f"  中位数 Median: {np.median(outliers_by_part):.1f}")
        
        # 数据变化统计 Data change statistics  
        original_sum = data.sum().sum()
        processed_sum = processed_data.sum().sum()
        change_ratio = (processed_sum - original_sum) / original_sum * 100 if original_sum > 0 else 0
        
        print(f"\n📈 数据变化统计 Data Change Statistics:")
        print(f"原始总需求 Original total demand: {original_sum:.0f}")
        print(f"处理后总需求 Processed total demand: {processed_sum:.0f}")
        print(f"需求变化比例 Demand change ratio: {change_ratio:+.2f}%")
        
        return processed_data, processing_info
    
    def extract_features(self, data: pd.DataFrame, strategy: str = 'combined') -> pd.DataFrame:
        """
        步骤1：特征提取
        Step 1: Feature extraction
        
        使用代理1从时间序列数据中提取特征
        Use Agent 1 to extract features from time series data
        
        参数 Args:
            data (pd.DataFrame): 输入时间序列数据 Input time series data
            strategy (str): 特征提取策略 Feature extraction strategy
            
        返回 Returns:
            pd.DataFrame: 提取的特征 Extracted features
        """
        print("\n" + "=" * 60)
        print("步骤1：特征提取 STEP 1: FEATURE EXTRACTION")
        print("=" * 60)
        
        self.features = self.agent1.extract_features(data, strategy)
        
        print(f"成功提取{self.features.shape[1]}个特征，来自{self.features.shape[0]}个时间序列")
        print(f"Successfully extracted {self.features.shape[1]} features from {self.features.shape[0]} time series")
        print(f"使用策略 Used strategy: {strategy}")
        
        return self.features
    
    def classify_spare_parts(self, features: Optional[pd.DataFrame] = None) -> Tuple[np.array, Dict]:
        """
        步骤2：备件分类
        Step 2: Spare parts classification
        
        使用代理2根据需求模式对备件进行分类
        Use Agent 2 to classify spare parts based on demand patterns
        
        参数 Args:
            features (pd.DataFrame): 来自代理1的特征 Features from Agent 1
            
        返回 Returns:
            tuple: (cluster_labels, evaluation_results) 聚类标签和评估结果
        """
        print("\n" + "=" * 60)
        print("步骤2：备件分类 STEP 2: SPARE PARTS CLASSIFICATION")
        print("=" * 60)
        
        if features is None:
            features = self.features
            
        if features is None:
            raise ValueError("没有可用特征。请先运行extract_features()方法。No features available. Run extract_features() first.")
        
        self.cluster_labels, evaluation = self.agent2.classify(
            features,
            use_pca=self.config.classification.use_pca,
            use_cosine_similarity=self.config.classification.use_cosine_similarity
        )
        
        n_clusters = len(np.unique(self.cluster_labels))
        print(f"分类完成，共生成{n_clusters}个聚类")
        print(f"Classification completed with {n_clusters} clusters")
        print(f"聚类准确度 Clustering accuracy: {evaluation['accuracy']:.3f}")
        
        return self.cluster_labels, evaluation
    
    def run_forecasting_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        步骤3：运行候选预测模型
        Step 3: Run candidate forecasting models
        
        运行所有候选预测模型以获取错误指标
        Run all candidate forecasting models to get error metrics
        
        参数 Args:
            data (pd.DataFrame): 原始时间序列数据 Original time series data
            
        返回 Returns:
            dict: 所有预测模型的结果 Results from all forecasting models
        """
        print("\n" + "=" * 60)
        print("步骤3：运行候选预测模型 STEP 3: RUN CANDIDATE FORECASTING MODELS")
        print("=" * 60)
        
        results = self.forecasting_models.run_all_models(
            data, self.config.system.candidate_models
        )
        
        # 提取错误指标 Extract error metrics
        error_dict = {}
        successful_models = 0
        
        for model_name, result in results.items():
            if result['metrics'] is not None:
                # 使用MASE作为主要指标，备选MAE Use MASE as primary metric, fallback to MAE
                if isinstance(result['metrics'], dict):
                    # 单个指标值 Single metric value
                    error = result['metrics'].get('MASE', result['metrics'].get('MAE', 1.0))
                    error_dict[model_name] = [error] * len(data.columns)
                elif isinstance(result['metrics'], (list, np.ndarray)):
                    # 每个备件的指标值 Per spare part metric values
                    error_dict[model_name] = result['metrics']
                else:
                    # 标量值 Scalar value
                    error_dict[model_name] = [float(result['metrics'])] * len(data.columns)
                successful_models += 1
        
        # 创建错误数据框 Create error DataFrame
        if error_dict:
            self.model_errors = pd.DataFrame(error_dict)
        else:
            # 使用虚拟数据作为回退 Fallback with dummy data
            print("警告：没有成功的模型结果，使用虚拟数据 Warning: No successful model results, using dummy data")
            self.model_errors = pd.DataFrame({
                model: np.random.uniform(0.1, 2.0, len(data.columns))
                for model in self.config.system.candidate_models
            })
        
        print(f"预测模型运行完成：{successful_models}/{len(results)}个模型成功")
        print(f"Forecasting completed: {successful_models}/{len(results)} models successful")
        
        return results
    
    def select_best_models(self, features: Optional[pd.DataFrame] = None,
                          cluster_labels: Optional[np.array] = None,
                          model_errors: Optional[pd.DataFrame] = None) -> np.array:
        """
        步骤4：智能模型选择
        Step 4: Intelligent model selection
        
        使用代理3为每个备件选择最佳预测模型
        Use Agent 3 to select the best forecasting model for each spare part
        
        参数 Args:
            features (pd.DataFrame): 来自代理1的特征 Features from Agent 1
            cluster_labels (np.array): 来自代理2的聚类标签 Cluster labels from Agent 2
            model_errors (pd.DataFrame): 预测模型的错误 Forecasting model errors
            
        返回 Returns:
            np.array: 最佳模型推荐 Best model recommendations
        """
        print("\n" + "=" * 60)
        print("步骤4：智能模型选择 STEP 4: INTELLIGENT MODEL SELECTION")
        print("=" * 60)
        
        if features is None:
            features = self.features
        if cluster_labels is None:
            cluster_labels = self.cluster_labels
        if model_errors is None:
            model_errors = self.model_errors
            
        if any(x is None for x in [features, cluster_labels, model_errors]):
            raise ValueError("缺少必需数据。请先运行前面的步骤。Missing required data. Run previous steps first.")
        
        # 准备元学习数据 Prepare meta-learning data
        X_meta, y_meta = self.agent3.prepare_meta_learning_data(
            features, model_errors, cluster_labels
        )
        
        # 训练元学习器 Train meta-learner
        training_results = self.agent3.train_meta_learner(
            X_meta, y_meta,
            use_lightgbm=self.config.model_selection.use_lightgbm
        )
        
        # 获取模型推荐 Get model recommendations
        self.best_models = self.agent3.predict_best_models(features, cluster_labels)
        
        print(f"模型选择完成，元学习器准确度: {training_results['accuracy']:.3f}")
        print(f"Model selection completed with meta-learner accuracy: {training_results['accuracy']:.3f}")
        print(f"推荐模型分布 Recommended model distribution:")
        
        from collections import Counter
        model_counts = Counter(self.best_models)
        for model, count in model_counts.items():
            print(f"  {model}: {count}个备件 spare parts")
        
        return self.best_models
    
    def evaluate_and_provide_feedback(self) -> Dict[str, Any]:
        """
        步骤5：系统评估和反馈
        Step 5: System evaluation and feedback
        
        评估系统性能并在代理间提供反馈
        Evaluate system performance and provide feedback between agents
        
        返回 Returns:
            dict: 评估结果和反馈信号 Evaluation results and feedback signals
        """
        print("\n" + "=" * 60)
        print("步骤5：系统评估和反馈 STEP 5: SYSTEM EVALUATION AND FEEDBACK")
        print("=" * 60)
        
        # 代理2评估 Agent 2 evaluation
        agent2_evaluation = self.agent2.evaluate_clustering_quality()
        agent2_feedback = self.agent2.provide_feedback_to_agent1(agent2_evaluation)
        
        # 代理3评估 Agent 3 evaluation
        agent3_evaluation = self.agent3.evaluate_error_threshold(
            self.model_errors, self.best_models, self.cluster_labels
        )
        agent3_feedback = self.agent3.provide_feedback_to_agent1(agent3_evaluation)
        
        # 向代理1提供反馈 Provide feedback to Agent 1
        if self.config.feedback_enabled:
            if not agent2_evaluation['meets_threshold']:
                print("代理2反馈：聚类准确度低于阈值 Agent 2 feedback: Clustering accuracy below threshold")
                self.agent1.update_strategy(agent2_feedback)
            
            if agent3_evaluation['exceeds_threshold']:
                print("代理3反馈：预测错误超过阈值 Agent 3 feedback: Forecasting error exceeds threshold")
                self.agent1.update_strategy(agent3_feedback)
        
        evaluation_results = {
            'agent2_evaluation': agent2_evaluation,
            'agent2_feedback': agent2_feedback,
            'agent3_evaluation': agent3_evaluation,
            'agent3_feedback': agent3_feedback,
            'iteration': self.iteration_count
        }
        
        # 检查收敛性 Check convergence
        if (agent2_evaluation['meets_threshold'] and 
            not agent3_evaluation['exceeds_threshold']):
            self.converged = True
            print("🎉 系统已收敛！ System has converged!")
        else:
            print("系统尚未收敛，将继续迭代 System not converged, will continue iteration")
        
        return evaluation_results
    
    def train(self, data: pd.DataFrame, max_iterations: Optional[int] = None, 
              outlier_strategy: str = 'strategy1') -> Dict[str, Any]:
        """
        完整训练流程
        Complete training workflow
        
        使用迭代反馈训练完整的ABCM系统
        Train the complete ABCM system with iterative feedback
        
        参数 Args:
            data (pd.DataFrame): 输入时间序列数据 Input time series data
            max_iterations (int): 最大迭代次数 Maximum number of iterations
            outlier_strategy (str): 异常值处理策略 Outlier handling strategy
                                  'strategy1': 使用MAD和标准差，中位数替换
                                  'strategy2': 使用非零均值，其他同strategy1  
                                  'strategy3': 不进行异常值处理
            
        返回 Returns:
            dict: 完整的训练结果 Complete training results
        """
        if max_iterations is None:
            max_iterations = self.config.max_iterations
        
        # 输入数据校验 Input data validation
        if not isinstance(data, pd.DataFrame):
            raise ValueError("输入数据必须是pandas DataFrame Input data must be pandas DataFrame")
        if data.empty:
            raise ValueError("输入数据不能为空 Input data cannot be empty")
        if data.shape[0] < 10:
            raise ValueError("时间序列长度太短，至少需要10个时期 Time series too short, need at least 10 periods")
        if data.shape[1] < 5:
            raise ValueError("备件数量太少，至少需要5个备件 Too few spare parts, need at least 5 spare parts")
        
        # 验证异常值处理策略 Validate outlier strategy
        valid_strategies = ['strategy1', 'strategy2', 'strategy3']
        if outlier_strategy not in valid_strategies:
            raise ValueError(f"无效的异常值处理策略: {outlier_strategy}. 有效选项: {valid_strategies}")
        
        print("=" * 80)
        print("🚀 开始ABCM系统训练 STARTING ABCM SYSTEM TRAINING")
        print("=" * 80)
        print(f"数据规模 Data size: {data.shape[1]}个备件 spare parts, {data.shape[0]}个时期 periods")
        print(f"最大迭代次数 Maximum iterations: {max_iterations}")
        print(f"异常值处理策略 Outlier handling strategy: {outlier_strategy}")
        print(f"数据校验通过 ✅ Data validation passed")
        
        # 🔧 步骤0：数据预处理 Step 0: Data preprocessing
        print("\n" + "=" * 80)
        print("🔧 步骤0：数据预处理 STEP 0: DATA PREPROCESSING")
        print("=" * 80)
        
        try:
            # 执行异常值处理 Perform outlier handling
            processed_data, self.outlier_info = self.handle_outliers(data, strategy=outlier_strategy)
            self.preprocessed_data = processed_data
            
            print(f"✅ 数据预处理完成 Data preprocessing completed")
            print(f"使用数据 Using data: {'原始数据' if outlier_strategy == 'strategy3' else '预处理后数据'}")
            print(f"Using data: {'Original data' if outlier_strategy == 'strategy3' else 'Preprocessed data'}")
            
        except Exception as e:
            print(f"❌ 数据预处理失败: {e}")
            print(f"❌ Data preprocessing failed: {e}")
            print("回退到原始数据 Falling back to original data")
            processed_data = data.copy()
            self.outlier_info = {'error': str(e)}
            self.preprocessed_data = processed_data
        
        all_results = []
        training_errors = []  # 记录训练过程中的错误 Record errors during training
        
        for iteration in range(max_iterations):
            self.iteration_count = iteration + 1
            print(f"\n{'🔄' * 20}")
            print(f"迭代 ITERATION {self.iteration_count}/{max_iterations}")
            print(f"{'🔄' * 20}")
            
            try:
                # 步骤1：特征提取 Step 1: Feature extraction
                print("\n📊 步骤1：特征提取 Step 1: Feature extraction")
                features = self.extract_features(processed_data)
                
                # 数据一致性检查 Data consistency check
                if features.shape[0] != processed_data.shape[1]:
                    raise ValueError(f"特征数量({features.shape[0]})与备件数量({processed_data.shape[1]})不匹配")
                
                # 步骤2：分类 Step 2: Classification
                print("\n🎯 步骤2：备件分类 Step 2: Spare parts classification")
                cluster_labels, classification_eval = self.classify_spare_parts(features)
                
                # 验证聚类结果 Validate clustering results
                if len(cluster_labels) != features.shape[0]:
                    raise ValueError(f"聚类标签数量({len(cluster_labels)})与特征数量({features.shape[0]})不匹配")
                
                # 步骤3：运行预测模型(仅在第一次迭代) Step 3: Run forecasting models (only in first iteration)
                if iteration == 0:
                    print("\n🔮 步骤3：运行预测模型 Step 3: Run forecasting models")
                    forecasting_results = self.run_forecasting_models(processed_data)
                    
                    # 验证预测结果 Validate forecasting results
                    if self.model_errors is None or self.model_errors.empty:
                        raise ValueError("预测模型运行失败，未获得错误数据")
                    if len(self.model_errors) != processed_data.shape[1]:
                        raise ValueError(f"模型错误数量({len(self.model_errors)})与备件数量({processed_data.shape[1]})不匹配")
                
                # 步骤4：模型选择 Step 4: Model selection
                print("\n🤖 步骤4：智能模型选择 Step 4: Intelligent model selection")
                best_models = self.select_best_models(features, cluster_labels, self.model_errors)
                
                # 验证模型选择结果 Validate model selection results
                if len(best_models) != features.shape[0]:
                    raise ValueError(f"最佳模型数量({len(best_models)})与特征数量({features.shape[0]})不匹配")
                
                # 步骤5：评估和反馈 Step 5: Evaluation and feedback
                print("\n📈 步骤5：系统评估和反馈 Step 5: System evaluation and feedback")
                evaluation_results = self.evaluate_and_provide_feedback()
                
                # 存储迭代结果 Store iteration results
                iteration_results = {
                    'iteration': self.iteration_count,
                    'features_shape': features.shape,
                    'n_clusters': len(np.unique(cluster_labels)),
                    'classification_accuracy': classification_eval['accuracy'],
                    'best_models': best_models,
                    'evaluation': evaluation_results,
                    'converged': self.converged,
                    'success': True,
                    'error': None
                }
                
                all_results.append(iteration_results)
                
                print(f"\n✅ 迭代 {self.iteration_count} 成功完成")
                print(f"✅ Iteration {self.iteration_count} completed successfully")
                print(f"聚类准确度 Clustering accuracy: {classification_eval['accuracy']:.3f}")
                print(f"聚类数量 Number of clusters: {len(np.unique(cluster_labels))}")
                
                # 检查收敛性 Check convergence
                if self.converged:
                    print(f"\n🎊 系统在{self.iteration_count}次迭代后收敛！")
                    print(f"🎊 System converged after {self.iteration_count} iterations!")
                    break
                    
            except Exception as e:
                error_msg = f"迭代 {self.iteration_count} 发生错误: {str(e)}"
                print(f"\n❌ {error_msg}")
                print(f"❌ Error in iteration {self.iteration_count}: {str(e)}")
                
                training_errors.append({
                    'iteration': self.iteration_count,
                    'error': str(e),
                    'error_type': type(e).__name__
                })
                
                # 如果是第一次迭代失败，直接返回错误
                # If first iteration fails, return error immediately
                if iteration == 0:
                    return {
                        'success': False,
                        'error': error_msg,
                        'total_iterations': self.iteration_count,
                        'training_errors': training_errors,
                        'outlier_info': self.outlier_info
                    }
                
                # 否则尝试继续下一次迭代
                # Otherwise try to continue with next iteration
                print(f"⚠️ 尝试继续下一次迭代... Attempting to continue with next iteration...")
                continue
        
        # 训练完成，准备最终结果 Training completed, prepare final results
        final_results = {
            'success': True,
            'total_iterations': self.iteration_count,
            'converged': self.converged,
            'outlier_strategy': outlier_strategy,
            'outlier_info': self.outlier_info,
            'data_preprocessing': {
                'outliers_detected': self.outlier_info.get('total_outliers_detected', 0),
                'outliers_replaced': self.outlier_info.get('total_outliers_replaced', 0),
                'preprocessing_applied': outlier_strategy != 'strategy3'
            },
            'final_features': features if 'features' in locals() else None,
            'final_cluster_labels': cluster_labels if 'cluster_labels' in locals() else None,
            'final_best_models': best_models if 'best_models' in locals() else None,
            'iteration_results': all_results,
            'training_errors': training_errors,
            'config': self.config
        }
        
        print("\n" + "=" * 80)
        if final_results['success']:
            print("✅ ABCM系统训练成功完成 ABCM SYSTEM TRAINING COMPLETED SUCCESSFULLY")
        else:
            print("⚠️ ABCM系统训练完成但有错误 ABCM SYSTEM TRAINING COMPLETED WITH ERRORS")
        print("=" * 80)
        print(f"总迭代次数 Total iterations: {self.iteration_count}")
        print(f"是否收敛 Converged: {'是 Yes' if self.converged else '否 No'}")
        print(f"异常值处理策略 Outlier strategy: {outlier_strategy}")
        if hasattr(self, 'outlier_info') and 'total_outliers_detected' in self.outlier_info:
            print(f"异常值检测数 Outliers detected: {self.outlier_info['total_outliers_detected']}")
            print(f"异常值替换数 Outliers replaced: {self.outlier_info['total_outliers_replaced']}")
        if 'cluster_labels' in locals():
            print(f"最终聚类数 Final clusters: {len(np.unique(cluster_labels))}")
        if training_errors:
            print(f"训练错误数量 Training errors: {len(training_errors)}")
        
        return final_results
    
    def predict(self, data: pd.DataFrame, outlier_strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        对新数据进行预测
        Make predictions on new data
        
        使用训练好的ABCM系统对新数据进行预测
        Make predictions on new data using trained ABCM system
        
        参数 Args:
            data (pd.DataFrame): 新的时间序列数据 New time series data
            outlier_strategy (str, optional): 异常值处理策略 Outlier handling strategy
                                            如果为None，则使用训练时的策略
                                            If None, use the same strategy as training
            
        返回 Returns:
            dict: 预测结果 Prediction results
        """
        print("=" * 60)
        print("🔮 ABCM系统预测 ABCM SYSTEM PREDICTION")
        print("=" * 60)
        
        # 确定异常值处理策略 Determine outlier handling strategy
        if outlier_strategy is None:
            # 使用训练时的策略（如果有的话） Use training strategy if available
            if hasattr(self, 'outlier_info') and 'strategy' in self.outlier_info:
                outlier_strategy = self.outlier_info['strategy']
            else:
                outlier_strategy = 'strategy3'  # 默认不处理 Default no processing
        
        print(f"异常值处理策略 Outlier handling strategy: {outlier_strategy}")
        
        # 预处理新数据 Preprocess new data
        if outlier_strategy != 'strategy3':
            print("对新数据进行异常值处理... Applying outlier handling to new data...")
            try:
                processed_data, outlier_info = self.handle_outliers(data, strategy=outlier_strategy)
                print(f"✅ 新数据预处理完成 New data preprocessing completed")
            except Exception as e:
                print(f"⚠️ 新数据预处理失败，使用原始数据: {e}")
                print(f"⚠️ New data preprocessing failed, using original data: {e}")
                processed_data = data.copy()
                outlier_info = {'error': str(e)}
        else:
            processed_data = data.copy()
            outlier_info = {'strategy': 'strategy3', 'no_processing': True}
        
        # 提取特征 Extract features
        print("提取特征... Extracting features...")
        features = self.agent1.extract_features(processed_data)
        
        # 分类备件 Classify spare parts
        print("分类备件... Classifying spare parts...")
        cluster_labels = self.agent2.predict_new_data(features)
        
        # 选择最佳模型 Select best models
        print("选择最佳模型... Selecting best models...")
        best_models = self.agent3.predict_best_models(features, cluster_labels)
        
        # 获取推荐与置信度 Get recommendations with confidence
        recommendations = self.agent3.get_model_recommendations(features, cluster_labels)
        
        prediction_results = {
            'features': features,
            'cluster_labels': cluster_labels,
            'best_models': best_models,
            'recommendations': recommendations,
            'outlier_strategy': outlier_strategy,
            'outlier_info': outlier_info,
            'preprocessing_applied': outlier_strategy != 'strategy3',
            'n_spare_parts': len(data.columns),
            'n_clusters': len(np.unique(cluster_labels))
        }
        
        print(f"✅ 预测完成：{len(data.columns)}个备件，{len(np.unique(cluster_labels))}个聚类")
        print(f"✅ Predictions completed: {len(data.columns)} spare parts, {len(np.unique(cluster_labels))} clusters")
        
        if outlier_strategy != 'strategy3' and 'total_outliers_detected' in outlier_info:
            print(f"预测数据异常值处理：检测{outlier_info['total_outliers_detected']}个，替换{outlier_info['total_outliers_replaced']}个")
            print(f"Prediction data outlier handling: detected {outlier_info['total_outliers_detected']}, replaced {outlier_info['total_outliers_replaced']}")
        
        from collections import Counter
        model_counts = Counter(best_models)
        print("推荐模型分布 Recommended model distribution:")
        for model, count in model_counts.items():
            print(f"  {model}: {count}个备件 spare parts")
        
        return prediction_results
    
    def save_system(self, base_filename: str) -> None:
        """
        保存完整的ABCM系统
        Save the complete ABCM system
        
        参数 Args:
            base_filename (str): 保存的基本文件名 Base filename for saving
        """
        print(f"保存ABCM系统... Saving ABCM system to {base_filename}...")
        
        # 保存各个代理 Save individual agents
        self.agent1.save_strategy_pool(f"{base_filename}_agent1.pkl")
        self.agent2.save_model(f"{base_filename}_agent2.pkl")
        self.agent3.save_model(f"{base_filename}_agent3.pkl")
        
        # 保存系统状态 Save system state
        system_state = {
            'features': self.features,
            'cluster_labels': self.cluster_labels,
            'model_errors': self.model_errors,
            'best_models': self.best_models,
            'iteration_count': self.iteration_count,
            'converged': self.converged,
            'config': self.config
        }
        
        FileUtils.save_pickle(system_state, f"{base_filename}_system.pkl")
        
        print(f"✅ ABCM系统已保存，基本文件名: {base_filename}")
        print(f"✅ ABCM system saved with base filename: {base_filename}")
    
    def load_system(self, base_filename: str) -> None:
        """
        加载之前保存的ABCM系统
        Load a previously saved ABCM system
        
        参数 Args:
            base_filename (str): 加载的基本文件名 Base filename for loading
        """
        print(f"加载ABCM系统... Loading ABCM system from {base_filename}...")
        
        # 加载各个代理 Load individual agents
        self.agent1.load_strategy_pool(f"{base_filename}_agent1.pkl")
        self.agent2.load_model(f"{base_filename}_agent2.pkl")
        self.agent3.load_model(f"{base_filename}_agent3.pkl")
        
        # 加载系统状态 Load system state
        system_state = FileUtils.load_pickle(f"{base_filename}_system.pkl")
        
        self.features = system_state['features']
        self.cluster_labels = system_state['cluster_labels']
        self.model_errors = system_state['model_errors']
        self.best_models = system_state['best_models']
        self.iteration_count = system_state['iteration_count']
        self.converged = system_state['converged']
        self.config = system_state['config']
        
        print(f"✅ ABCM系统已从{base_filename}加载完成")
        print(f"✅ ABCM system loaded from {base_filename}")


def main():
    """
    ABCM系统使用示例
    Example usage of the ABCM system
    """
    print("ABCM系统示例 ABCM System Example")
    print("=" * 50)
    
    # 加载配置 Load configuration
    config = get_config()
    
    # 方法1：使用预训练元学习器(推荐) Method 1: Using pretrained meta-learner (Recommended)
    print("\n🎯 方法1：使用预训练元学习器 Method 1: Using pretrained meta-learner")
    print("注意：预训练元学习器应通过Cross-validation.py训练生成")
    print("Note: Pretrained meta-learner should be generated through Cross-validation.py training")
    
    # 示例元学习器路径(请替换为实际路径) Example meta-learner path (replace with actual path)
    metalearner_path = "experiments/pretrained_metalearner.pkl"  # 或 .pkl 文件
    
    # 初始化ABCM系统，使用预训练元学习器 Initialize ABCM system with pretrained meta-learner
    # abcm_with_metalearner = ABCMSystem(config, pretrained_metalearner_path=metalearner_path)
    
    # 方法2：使用实时元学习(回退选项) Method 2: Using real-time meta-learning (Fallback option)
    print("\n⚙️  方法2：使用实时元学习 Method 2: Using real-time meta-learning")
    
    # 初始化ABCM系统，不使用预训练元学习器 Initialize ABCM system without pretrained meta-learner
    abcm = ABCMSystem(config)
    
    print("\n" + "=" * 80)
    print("🔧 异常值处理功能说明 OUTLIER HANDLING FUNCTIONALITY")
    print("=" * 80)
    print("ABCM系统现在支持三种异常值处理策略：")
    print("ABCM system now supports three outlier handling strategies:")
    print()
    print("📊 策略1 (strategy1) - 推荐 Recommended:")
    print("  - 使用非零需求的MAD (Median Absolute Deviation) 作为替代均值")
    print("  - Use MAD of non-zero demand as substitute mean")
    print("  - 使用非零需求标准差替代标准差")
    print("  - Use non-zero demand standard deviation")
    print("  - 使用非零需求中位数替换异常值")
    print("  - Replace outliers with non-zero demand median")
    print()
    print("📈 策略2 (strategy2) - 替代方案 Alternative:")
    print("  - 使用非零需求均值替代均值，其他同策略1")
    print("  - Use non-zero demand mean, others same as strategy1")
    print()
    print("⭕ 策略3 (strategy3) - 无处理 No processing:")
    print("  - 不进行异常值处理，使用原始数据")
    print("  - No outlier processing, use original data")
    print()
    print("🎯 自适应阈值计算 Adaptive Threshold Calculation:")
    print("  - 基于非零需求变异系数平方与临界值0.49的关系")
    print("  - Based on squared coefficient of variation vs critical value 0.49")
    print("  - 低变异性：阈值2.0-2.5，更严格的异常值检测")
    print("  - Low variability: threshold 2.0-2.5, stricter detection")
    print("  - 高变异性：阈值2.5-3.5，更宽松的异常值检测")
    print("  - High variability: threshold 2.5-3.5, lenient detection")
    
    print("\n" + "=" * 80)
    print("📋 使用示例 USAGE EXAMPLES")
    print("=" * 80)
    print("# 示例：加载数据(请替换为实际数据加载)")
    print("# Example: Load data (replace with actual data loading)")
    print("# data = pd.read_excel('your_data.xlsx', index_col=0)")
    print()
    print("# 方法1：使用策略1训练系统(推荐)")
    print("# Method 1: Train with strategy1 (Recommended)")
    print("# results = abcm.train(data, outlier_strategy='strategy1')")
    print()
    print("# 方法2：使用策略2训练系统")
    print("# Method 2: Train with strategy2")
    print("# results = abcm.train(data, outlier_strategy='strategy2')")
    print()
    print("# 方法3：不进行异常值处理")
    print("# Method 3: No outlier processing")
    print("# results = abcm.train(data, outlier_strategy='strategy3')")
    print()
    print("# 预测时使用相同策略(自动)")
    print("# Prediction uses same strategy (automatic)")
    print("# predictions = abcm.predict(new_data)")
    print()
    print("# 或者指定不同的预测策略")
    print("# Or specify different prediction strategy")
    print("# predictions = abcm.predict(new_data, outlier_strategy='strategy2')")
    print()
    print("# 保存训练好的系统")
    print("# Save the trained system")
    print("# abcm.save_system('trained_abcm_model')")
    
    print("\n" + "=" * 80)
    print("重要说明 IMPORTANT NOTES:")
    print("=" * 80)
    print("1. 异常值处理：基于间歇性需求预测理论的修正z-score方法")
    print("   Outlier handling: Modified z-score method based on intermittent demand theory")
    print("2. 自适应阈值：变异系数平方与0.49临界值的动态关系")
    print("   Adaptive threshold: Dynamic relationship of CV² with critical value 0.49")
    print("3. 预训练元学习器：来源于experiments/Cross-validation meta-learning-lightgbm.py")
    print("   Pretrained meta-learner: From experiments/Cross-validation meta-learning-lightgbm.py")
    print("4. 元学习器：使用RandomForest/LightGBM学习特征-模型匹配关系")
    print("   Meta-learner: Uses RandomForest/LightGBM to learn feature-model mapping")
    print("5. 处理策略：推荐策略1，适用于大多数间歇性需求场景")
    print("   Processing strategy: Strategy1 recommended for most intermittent demand scenarios")
    
    print("\nABCM系统初始化成功！ ABCM system initialized successfully!")
    print("现在支持智能异常值处理功能！ Now supports intelligent outlier handling!")


if __name__ == "__main__":
    main() 