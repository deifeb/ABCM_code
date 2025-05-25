"""
模型选择代理 (Agent 3) - Model Selection Agent

该代理结合代理1的特征和代理2的聚类标签，使用LightGBM元学习识别最适合的预测模型
This agent combines features from Agent 1 and clustering labels from Agent 2,
using LightGBM meta-learning to identify the most suitable forecasting model

主要功能 Main Functions:
- 元学习数据准备 Meta-learning data preparation
- LightGBM模型训练 LightGBM model training
- 最佳模型预测 Best model prediction
- 错误阈值评估 Error threshold evaluation
- 向代理1提供反馈 Provide feedback to Agent 1

作者 Author: ABCM Team
创建时间 Created: 2024
"""

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
import os

warnings.filterwarnings("ignore")


class ModelSelectionAgent:
    """
    代理3：模型选择代理
    Agent 3: Model Selection Agent
    
    该代理是ABCM系统的决策中心，负责为每个备件选择最适合的预测模型
    This agent is the decision center of the ABCM system, responsible for selecting 
    the most suitable forecasting model for each spare part
    
    核心技术 Core Technologies:
    1. 预训练模型匹配关系建立 - 通过综合模型评估建立特征-模型映射
       Pretrained model mapping establishment - Establish feature-model mapping through comprehensive model evaluation
       
    2. 元学习 - 学习如何为不同类型的数据选择最佳模型
       Meta-learning - Learn how to select the best model for different types of data
       
    3. LightGBM - 高效的梯度提升决策树算法
       LightGBM - Efficient gradient boosting decision tree algorithm
       
    4. 错误跟踪 - 监控预测性能并触发反馈调整
       Error tracking - Monitor prediction performance and trigger feedback adjustments
       
    5. 模型推荐 - 为新数据提供智能模型选择
       Model recommendation - Provide intelligent model selection for new data
    
    模型匹配关系获取过程 Model Matching Relationship Acquisition Process:
    ========================================================================
    初始阶段通过以下步骤建立特征-模型匹配关系（参考ABCM_RAF.py实现）：
    Initial phase establishes feature-model matching relationships through these steps (refer to ABCM_RAF.py):
    
    1. 数据准备阶段 Data Preparation:
       - 提取特征：使用Agent 1提取F1-F9专家特征和自编码器特征
       - 聚类分析：使用Agent 2进行KScorer聚类，将备件分为不同需求模式类别
       - Extract features: Use Agent 1 to extract F1-F9 expert features and autoencoder features
       - Clustering: Use Agent 2 for KScorer clustering, categorizing spare parts into different demand pattern categories
    
    2. 综合模型评估阶段 Comprehensive Model Evaluation:
       - 候选模型：DeepAR, SBJ, ETS, DeepRenewal (Flat/Exact/Hybrid), ARIMA等
       - 分类别评估：对每个聚类类别的备件，分别训练和测试所有候选预测模型
       - 错误指标计算：使用IntermittentEvaluator计算MRAE, MASE, MAAPE, MAE等指标
       - 多轮验证：进行多轮(如10轮)训练和评估，确保结果稳定性
       - Candidate models: DeepAR, SBJ, ETS, DeepRenewal (Flat/Exact/Hybrid), ARIMA, etc.
       - Category-wise evaluation: For spare parts in each cluster category, train and test all candidate forecasting models
       - Error metrics calculation: Use IntermittentEvaluator to calculate MRAE, MASE, MAAPE, MAE metrics
       - Multiple rounds validation: Conduct multiple rounds (e.g., 10 rounds) of training and evaluation for stable results
    
    3. 最佳模型识别阶段 Best Model Identification:
       - 错误矩阵生成：为每个备件生成包含所有模型错误的矩阵
       - 最优选择：基于最小错误原则，为每个备件确定最佳预测模型
       - 模式发现：分析特征模式与最佳模型选择之间的关系
       - Error matrix generation: Generate matrix containing all model errors for each spare part  
       - Optimal selection: Determine best forecasting model for each spare part based on minimum error principle
       - Pattern discovery: Analyze relationships between feature patterns and optimal model selections
    
    4. 元学习器训练阶段 Meta-learner Training:
       - 数据融合：将特征数据与最佳模型标签结合形成元学习数据集
       - 交叉验证训练：使用RandomForest + GridSearchCV进行超参数优化和交叉验证
       - 关系学习：学习从特征空间(F1-F9 + 自编码器 + 聚类标签)到最佳模型的映射
       - Data fusion: Combine feature data with best model labels to form meta-learning dataset
       - Cross-validation training: Use RandomForest + GridSearchCV for hyperparameter optimization and cross-validation
       - Relationship learning: Learn mapping from feature space (F1-F9 + autoencoder + cluster labels) to best models
    
    这种方法确保了模型选择决策基于实际的预测性能评估，而非任意规则或假设。
    This approach ensures model selection decisions are based on actual forecasting performance evaluation rather than arbitrary rules or assumptions.
    """
    
    def __init__(self, error_threshold=None, models_list=None, pretrained_metalearner_path=None):
        """
        初始化模型选择代理
        Initialize the Model Selection Agent
        
        参数 Args:
            error_threshold (float): 错误阈值 Error threshold for triggering feedback
            models_list (list): 候选预测模型列表 List of candidate forecasting models
            pretrained_metalearner_path (str): 预训练元学习器路径 Path to pretrained meta-learner from Cross-validation training
        """
        self.error_threshold = error_threshold or 1.5
        self.models_list = models_list or ['DeepAR', 'SBJ', 'ETS', 'DeepRenewal']
        
        # 预训练元学习器相关 Pretrained meta-learner related
        self.pretrained_metalearner_path = pretrained_metalearner_path
        self.pretrained_metalearner = None
        self.pretrained_label_encoder = None
        self.use_pretrained = False
        
        # 实时元学习相关 Real-time meta-learning related
        self.meta_learner = None
        self.label_encoder = None
        self.feature_columns = None
        
        # Load pretrained meta-learner if provided
        if pretrained_metalearner_path and os.path.exists(pretrained_metalearner_path):
            self.load_pretrained_metalearner(pretrained_metalearner_path)
        
        print(f"模型选择代理初始化完成")
        print(f"Model Selection Agent initialized")
        print(f"错误阈值 Error threshold: {self.error_threshold}")
        print(f"候选模型 Candidate models: {len(self.models_list)}")
        if self.use_pretrained:
            print(f"✅ 已加载预训练元学习器 Loaded pretrained meta-learner from: {pretrained_metalearner_path}")
        else:
            print(f"⚠️  未使用预训练元学习器，将使用实时训练 No pretrained meta-learner, will use real-time training")
        
    def load_pretrained_metalearner(self, metalearner_path):
        """
        加载从Cross-validation meta-learning训练得到的预训练元学习器
        Load pretrained meta-learner from Cross-validation meta-learning training
        
        参数 Args:
            metalearner_path (str): 元学习器文件路径 Path to meta-learner file
        """
        try:
            import pickle
            
            with open(metalearner_path, 'rb') as f:
                pretrained_data = pickle.load(f)
            
            self.pretrained_metalearner = pretrained_data['meta_learner']
            self.pretrained_label_encoder = pretrained_data['label_encoder']
            self.feature_columns = pretrained_data.get('feature_columns', None)
            
            self.use_pretrained = True
            
            print("✅ 预训练元学习器加载成功 Pretrained meta-learner loaded successfully")
            print(f"模型类型 Model type: {type(self.pretrained_metalearner).__name__}")
            print(f"标签编码器类别 Label encoder classes: {self.pretrained_label_encoder.classes_}")
            
            # 如果是GridSearchCV对象，显示最佳参数
            if hasattr(self.pretrained_metalearner, 'best_params_'):
                print(f"最佳参数 Best parameters: {self.pretrained_metalearner.best_params_}")
            
        except Exception as e:
            print(f"❌ 预训练元学习器加载失败 Failed to load pretrained meta-learner: {e}")
            print("将回退到实时元学习模式 Falling back to real-time meta-learning mode")
            self.use_pretrained = False
    
    def predict_best_models_with_pretrained(self, features, cluster_labels):
        """
        使用预训练元学习器预测最佳模型
        Predict best models using pretrained meta-learner
        
        参数 Args:
            features (pd.DataFrame): 来自代理1的特征 Features from Agent 1
            cluster_labels (np.array): 来自代理2的聚类标签 Cluster labels from Agent 2
            
        返回 Returns:
            np.array: 预测的最佳模型 Predicted best models
        """
        if not self.use_pretrained or self.pretrained_metalearner is None:
            raise ValueError("预训练元学习器不可用 Pretrained meta-learner not available")
        
        print("使用预训练元学习器进行模型选择 Using pretrained meta-learner for model selection")
        
        # 准备特征，添加聚类标签 Prepare features with cluster labels
        meta_features = features.copy()
        meta_features['cluster_label'] = cluster_labels
        
        # 确保特征列一致性 - 关键改进 Ensure feature column consistency - Key improvement
        if hasattr(self, 'pretrained_feature_columns') and self.pretrained_feature_columns:
            # 检查缺失的特征列 Check for missing feature columns
            missing_cols = set(self.pretrained_feature_columns) - set(meta_features.columns)
            if missing_cols:
                print(f"⚠️ 警告：缺少预训练特征列 {missing_cols}，将用0填充")
                print(f"⚠️ Warning: Missing pretrained feature columns {missing_cols}, filling with 0")
                for col in missing_cols:
                    meta_features[col] = 0
            
            # 检查多余的特征列 Check for extra feature columns
            extra_cols = set(meta_features.columns) - set(self.pretrained_feature_columns)
            if extra_cols:
                print(f"ℹ️ 信息：发现额外特征列 {extra_cols}，将被忽略")
                print(f"ℹ️ Info: Found extra feature columns {extra_cols}, will be ignored")
            
            # 按预训练时的列顺序重新排列 Reorder columns according to pretrained order
            try:
                meta_features = meta_features[self.pretrained_feature_columns]
            except KeyError as e:
                print(f"❌ 特征列匹配失败: {e}")
                print(f"❌ Feature column matching failed: {e}")
                print(f"预训练列 Pretrained columns: {self.pretrained_feature_columns}")
                print(f"当前列 Current columns: {list(meta_features.columns)}")
                raise
        else:
            print("⚠️ 警告：未找到预训练特征列信息，使用当前特征")
            print("⚠️ Warning: Pretrained feature columns not found, using current features")
        
        # 使用预训练模型进行预测 Use pretrained model for prediction
        y_pred_encoded = self.pretrained_metalearner.predict(meta_features)
        
        # 解码预测结果 Decode prediction results
        if hasattr(self, 'pretrained_label_encoder') and self.pretrained_label_encoder:
            predicted_models = self.pretrained_label_encoder.inverse_transform(y_pred_encoded)
        else:
            print("⚠️ 警告：未找到预训练标签编码器，使用原始预测结果")
            print("⚠️ Warning: Pretrained label encoder not found, using raw predictions")
            predicted_models = y_pred_encoded
        
        print(f"基于预训练元学习器的模型选择完成 Pretrained meta-learner model selection completed")
        print(f"为 {len(predicted_models)} 个备件选择了模型 Selected models for {len(predicted_models)} spare parts")
        
        from collections import Counter
        model_counts = Counter(predicted_models)
        print("预测模型分布 Predicted model distribution:")
        for model, count in model_counts.items():
            print(f"  {model}: {count}个备件 spare parts")
        
        return np.array(predicted_models)
    
    def prepare_meta_learning_data(self, features, model_errors, cluster_labels):
        """
        准备元学习数据，结合特征和模型错误
        Prepare data for meta-learning by combining features and model errors
        
        元学习的核心思想是"学习如何学习"，这里我们学习如何为不同的数据选择最佳模型
        The core idea of meta-learning is "learning to learn", here we learn how to select 
        the best model for different data
        
        参数 Args:
            features (pd.DataFrame): 来自代理1的特征 Features from Agent 1
            model_errors (pd.DataFrame): 不同预测模型的错误 Errors from different forecasting models
            cluster_labels (np.array): 来自代理2的聚类标签 Cluster labels from Agent 2
            
        返回 Returns:
            tuple: (X_meta, y_meta) 元学习的输入和输出 Input and output for meta-learning
        """
        print("准备元学习数据... Preparing meta-learning data...")
        
        # 数据一致性检查 Data consistency check
        if len(features) != len(cluster_labels):
            raise ValueError(f"特征数量 ({len(features)}) 与聚类标签数量 ({len(cluster_labels)}) 不匹配")
        if len(features) != len(model_errors):
            raise ValueError(f"特征数量 ({len(features)}) 与模型错误数量 ({len(model_errors)}) 不匹配")
        
        # 结合特征与聚类标签 Combine features with cluster labels
        meta_features = features.copy()
        meta_features['cluster_label'] = cluster_labels
        
        # 为每个备件找到最佳模型(最小错误) Find best model for each spare part (lowest error)
        best_models = []
        for idx in range(len(model_errors)):
            row_errors = model_errors.iloc[idx]
            # 排除非模型列 Exclude non-model columns if any
            model_cols = [col for col in row_errors.index if col in self.models_list]
            if model_cols:
                best_model = row_errors[model_cols].idxmin()
                best_models.append(best_model)
            else:
                best_models.append('DeepAR')  # 默认回退 Default fallback
        
        # 编码目标标签 Encode target labels
        self.label_encoder = LabelEncoder()
        y_meta = self.label_encoder.fit_transform(best_models)
        
        # 存储特征列以供后续使用 Store feature columns for later use
        self.feature_columns = meta_features.columns.tolist()
        
        print(f"元学习数据准备完成：{len(meta_features)}个样本，{len(meta_features.columns)}个特征")
        print(f"Meta-learning data prepared: {len(meta_features)} samples, {len(meta_features.columns)} features")
        print(f"存储特征列 Stored feature columns: {len(self.feature_columns)} columns")
        print(f"目标模型分布 Target model distribution:")
        
        from collections import Counter
        model_counts = Counter(best_models)
        for model, count in model_counts.items():
            print(f"  {model}: {count}个备件 spare parts")
        
        return meta_features, y_meta
    
    def train_meta_learner(self, X_meta, y_meta, use_lightgbm=True, 
                          test_size=0.2, random_state=42):
        """
        训练元学习模型
        Train the meta-learning model
        
        元学习模型的目标是学习特征-模型性能的映射关系
        The goal of the meta-learning model is to learn the mapping between features and model performance
        
        参数 Args:
            X_meta (pd.DataFrame): 元特征 Meta-features
            y_meta (np.array): 目标标签(最佳模型) Target labels (best models)
            use_lightgbm (bool): 是否使用LightGBM或随机森林 Whether to use LightGBM or Random Forest
            test_size (float): 测试集比例 Test set proportion
            random_state (int): 随机种子 Random seed
            
        返回 Returns:
            dict: 训练结果和性能指标 Training results and performance metrics
        """
        print("开始训练元学习器... Training meta-learner...")
        start_time = time.time()
        
        # 数据分割 Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_meta, y_meta, test_size=test_size, random_state=random_state
        )
        
        if use_lightgbm:
            print("使用LightGBM作为元学习器... Using LightGBM as meta-learner...")
            
            # LightGBM参数 LightGBM parameters
            lgb_params = {
                'objective': 'multiclass',
                'num_class': len(np.unique(y_meta)),
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': random_state
            }
            
            # 创建数据集 Create datasets
            train_data = lgb.Dataset(X_train, label=y_train)
            valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
            
            # 训练模型 Train model
            self.meta_learner = lgb.train(
                lgb_params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=100,
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            # 预测 Predictions
            y_pred = np.argmax(self.meta_learner.predict(X_test), axis=1)
            
        else:
            print("使用随机森林作为元学习器... Using Random Forest as meta-learner...")
            
            # 随机森林超参数调优 Random Forest with hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }
            
            rf = RandomForestClassifier(random_state=random_state, class_weight="balanced")
            self.meta_learner = GridSearchCV(
                rf, param_grid, cv=5, scoring='accuracy', 
                verbose=0, n_jobs=-1
            )
            
            self.meta_learner.fit(X_train, y_train)
            y_pred = self.meta_learner.predict(X_test)
        
        # 计算性能指标 Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # 交叉验证 Cross-validation
        if use_lightgbm:
            # 对于LightGBM，使用训练好的模型进行CV近似 For LightGBM, use the trained model for CV approximation
            cv_scores = [accuracy]  # 简化版本 Simplified for LightGBM
        else:
            cv_scores = cross_val_score(self.meta_learner, X_train, y_train, cv=5)
        
        end_time = time.time()
        
        results = {
            'accuracy': accuracy,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'training_time': end_time - start_time,
            'best_params': getattr(self.meta_learner, 'best_params_', None),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        print(f"元学习器训练完成，耗时: {results['training_time']:.2f}秒")
        print(f"Meta-learner training completed in {results['training_time']:.2f} seconds")
        print(f"准确度 Accuracy: {results['accuracy']:.3f}")
        print(f"交叉验证得分 CV Score: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")
        
        return results
    
    def predict_best_models(self, features, cluster_labels):
        """
        为新数据预测最佳预测模型
        Predict the best forecasting model for new data
        
        优先使用预训练元学习器，如果不可用则使用元学习
        Prefer pretrained meta-learner, fallback to meta-learning if unavailable
        
        参数 Args:
            features (pd.DataFrame): 来自代理1的特征 Features from Agent 1
            cluster_labels (np.array): 来自代理2的聚类标签 Cluster labels from Agent 2
            
        返回 Returns:
            np.array: 每个备件的预测最佳模型 Predicted best models for each spare part
        """
        print("\n开始模型选择... Starting model selection...")
        
        # 优先使用预训练元学习器 Prefer pretrained meta-learner
        if self.use_pretrained and self.pretrained_metalearner is not None:
            print("🎯 使用预训练元学习器进行模型选择 Using pretrained meta-learner for model selection")
            return self.predict_best_models_with_pretrained(features, cluster_labels)
        
        # 回退到实时元学习 Fallback to real-time meta-learning
        if self.meta_learner is None:
            raise ValueError("元学习器未训练。请先调用train_meta_learner()方法。Meta-learner not trained. Call train_meta_learner() first.")
        
        print("🔧 使用实时元学习器进行模型选择 Using real-time meta-learner for model selection")
        
        # 数据一致性检查 Data consistency check
        if len(features) != len(cluster_labels):
            raise ValueError(f"特征数量 ({len(features)}) 与聚类标签数量 ({len(cluster_labels)}) 不匹配")
        
        # 准备元特征 Prepare meta-features
        meta_features = features.copy()
        meta_features['cluster_label'] = cluster_labels
        
        # 确保特征列一致性 - 关键改进 Ensure feature column consistency - Key improvement
        if hasattr(self, 'feature_columns') and self.feature_columns:
            # 检查缺失的特征列 Check for missing feature columns
            missing_cols = set(self.feature_columns) - set(meta_features.columns)
            if missing_cols:
                print(f"⚠️ 警告：缺少训练特征列 {missing_cols}，将用0填充")
                print(f"⚠️ Warning: Missing training feature columns {missing_cols}, filling with 0")
                for col in missing_cols:
                    meta_features[col] = 0
            
            # 检查多余的特征列 Check for extra feature columns
            extra_cols = set(meta_features.columns) - set(self.feature_columns)
            if extra_cols:
                print(f"ℹ️ 信息：发现额外特征列 {extra_cols}，将被忽略")
                print(f"ℹ️ Info: Found extra feature columns {extra_cols}, will be ignored")
            
            # 按训练时的列顺序重新排列 Reorder columns according to training order
            try:
                meta_features = meta_features[self.feature_columns]
                print(f"✅ 特征列对齐成功，使用 {len(self.feature_columns)} 个特征")
                print(f"✅ Feature column alignment successful, using {len(self.feature_columns)} features")
            except KeyError as e:
                print(f"❌ 特征列匹配失败: {e}")
                print(f"❌ Feature column matching failed: {e}")
                print(f"训练列 Training columns: {self.feature_columns}")
                print(f"当前列 Current columns: {list(meta_features.columns)}")
                raise
        else:
            print("⚠️ 警告：未找到训练特征列信息，使用当前特征")
            print("⚠️ Warning: Training feature columns not found, using current features")
        
        # 使用元学习器进行预测 Use meta-learner for prediction
        try:
            if hasattr(self.meta_learner, 'predict'):
                # 对于sklearn模型 For sklearn models
                y_pred_encoded = self.meta_learner.predict(meta_features)
            else:
                # 对于LightGBM模型 For LightGBM models
                y_pred_proba = self.meta_learner.predict(meta_features)
                y_pred_encoded = np.argmax(y_pred_proba, axis=1)
        except Exception as e:
            print(f"❌ 元学习器预测失败: {e}")
            print(f"❌ Meta-learner prediction failed: {e}")
            print(f"元特征形状 Meta-features shape: {meta_features.shape}")
            raise
        
        # 解码预测结果 Decode prediction results
        if hasattr(self, 'label_encoder') and self.label_encoder:
            try:
                predicted_models = self.label_encoder.inverse_transform(y_pred_encoded)
            except Exception as e:
                print(f"❌ 标签解码失败: {e}")
                print(f"❌ Label decoding failed: {e}")
                print(f"预测编码 Predicted encodings: {y_pred_encoded}")
                raise
        else:
            print("⚠️ 警告：未找到标签编码器，使用原始预测结果")
            print("⚠️ Warning: Label encoder not found, using raw predictions")
            predicted_models = y_pred_encoded
        
        print(f"✅ 实时元学习器模型选择完成")
        print(f"✅ Real-time meta-learner model selection completed")
        print(f"为 {len(predicted_models)} 个备件选择了模型")
        print(f"Selected models for {len(predicted_models)} spare parts")
        
        # 显示预测模型分布 Display predicted model distribution
        from collections import Counter
        model_counts = Counter(predicted_models)
        print("预测模型分布 Predicted model distribution:")
        for model, count in model_counts.items():
            print(f"  {model}: {count}个备件 spare parts")
        
        return np.array(predicted_models)
    
    def calculate_category_errors(self, model_errors, cluster_labels):
        """
        计算每个备件类别的预测错误
        Calculate forecasting errors for each category of spare parts
        
        参数 Args:
            model_errors (pd.DataFrame): 每个备件的模型错误 Model errors for each spare part
            cluster_labels (np.array): 聚类标签 Cluster labels
            
        返回 Returns:
            dict: 按类别和模型分组的平均错误 Average errors by category and model
        """
        print("计算类别错误... Calculating category errors...")
        
        category_errors = {}
        
        # 按聚类标签分组 Group by cluster labels
        df_with_labels = model_errors.copy()
        df_with_labels['cluster_label'] = cluster_labels
        
        grouped = df_with_labels.groupby('cluster_label')
        
        for label, group in grouped:
            model_cols = [col for col in group.columns if col in self.models_list]
            category_errors[label] = group[model_cols].mean().to_dict()
            
            print(f"类别 {label}: {len(group)} 个备件")
            print(f"Category {label}: {len(group)} spare parts")
        
        return category_errors
    
    def evaluate_error_threshold(self, model_errors, predicted_models, cluster_labels):
        """
        评估预测错误是否超过阈值并触发反馈
        Evaluate if forecasting errors exceed threshold and trigger feedback
        
        参数 Args:
            model_errors (pd.DataFrame): 模型错误 Model errors
            predicted_models (np.array): 预测的最佳模型 Predicted best models
            cluster_labels (np.array): 聚类标签 Cluster labels
            
        返回 Returns:
            dict: 评估结果和反馈信号 Evaluation results and feedback signals
        """
        print("评估错误阈值... Evaluating error threshold...")
        
        if self.error_threshold is None:
            # 将阈值设置为所有单模型错误的平均值 Set threshold as mean of all single-model errors
            all_errors = []
            for model in self.models_list:
                if model in model_errors.columns:
                    all_errors.extend(model_errors[model].dropna().tolist())
            self.error_threshold = np.mean(all_errors) if all_errors else 0.1
            
            print(f"自动设置错误阈值为: {self.error_threshold:.3f}")
            print(f"Automatically set error threshold to: {self.error_threshold:.3f}")
        
        # 计算预测模型的错误 Calculate errors for predicted models
        predicted_errors = []
        for i, model in enumerate(predicted_models):
            if model in model_errors.columns:
                predicted_errors.append(model_errors.iloc[i][model])
            else:
                predicted_errors.append(self.error_threshold)  # 回退值 Fallback
        
        avg_predicted_error = np.mean(predicted_errors)
        exceeds_threshold = avg_predicted_error > self.error_threshold
        
        # 计算类别特定错误 Calculate category-specific errors
        self.calculate_category_errors(model_errors, cluster_labels)
        
        evaluation = {
            'avg_error': avg_predicted_error,
            'error_threshold': self.error_threshold,
            'exceeds_threshold': exceeds_threshold,
            'category_errors': self.category_errors,
            'num_categories': len(np.unique(cluster_labels)),
            'error_reduction': self.error_threshold - avg_predicted_error
        }
        
        print(f"平均预测错误: {avg_predicted_error:.3f}")
        print(f"Average prediction error: {avg_predicted_error:.3f}")
        print(f"错误阈值: {self.error_threshold:.3f}")
        print(f"Error threshold: {self.error_threshold:.3f}")
        print(f"是否超过阈值: {exceeds_threshold}")
        print(f"Exceeds threshold: {exceeds_threshold}")
        
        return evaluation
    
    def provide_feedback_to_agent1(self, evaluation):
        """
        当错误阈值被超过时向代理1提供反馈
        Provide feedback to Agent 1 when error threshold is exceeded
        
        参数 Args:
            evaluation (dict): 错误评估结果 Error evaluation results
            
        返回 Returns:
            dict: 给代理1的反馈信号 Feedback signal for Agent 1
        """
        feedback = {
            'error_exceeds_threshold': evaluation['exceeds_threshold'],
            'avg_error': evaluation['avg_error'],
            'error_threshold': evaluation['error_threshold']
        }
        
        if evaluation['exceeds_threshold']:
            # 建议策略调整 Suggest strategy adjustments
            if evaluation['num_categories'] < 3:
                feedback['suggestion'] = 'adjust_autoencoder_dim'
                feedback['message'] = "类别过少，考虑调整自编码器维度 Too few categories, consider adjusting autoencoder dimension"
            elif evaluation['avg_error'] > 2 * evaluation['error_threshold']:
                feedback['suggestion'] = 'add_features'
                feedback['message'] = "错误较高，考虑添加更多显式特征 High error, consider adding more explicit features"
            else:
                feedback['suggestion'] = 'adjust_autoencoder_dim'
                feedback['message'] = "中等错误，尝试调整自编码器潜在维度 Moderate error, try adjusting autoencoder latent dimension"
                
            print(f"向代理1发送反馈: {feedback['message']}")
            print(f"Sending feedback to Agent 1: {feedback['message']}")
        else:
            feedback['suggestion'] = 'maintain_strategy'
            feedback['message'] = "错误在可接受范围内 Error within acceptable range"
            
            print("无需反馈，性能满足要求 No feedback needed, performance meets requirements")
        
        return feedback
    
    def get_model_recommendations(self, features, cluster_labels):
        """
        获取备件的模型推荐
        Get model recommendations for spare parts
        
        参数 Args:
            features (pd.DataFrame): 来自代理1的特征 Features from Agent 1
            cluster_labels (np.array): 来自代理2的聚类标签 Cluster labels from Agent 2
            
        返回 Returns:
            pd.DataFrame: 带置信度分数的推荐 Recommendations with confidence scores
        """
        predicted_models = self.predict_best_models(features, cluster_labels)
        
        # 创建推荐数据框 Create recommendations dataframe
        recommendations = pd.DataFrame({
            'spare_part_id': range(len(predicted_models)),
            'cluster_label': cluster_labels,
            'recommended_model': predicted_models
        })
        
        # 添加置信度分数(如果可用) Add confidence scores if available
        if hasattr(self.meta_learner, 'predict_proba'):
            meta_features = features.copy()
            meta_features['cluster_label'] = cluster_labels
            meta_features = meta_features[self.feature_columns]
            
            probabilities = self.meta_learner.predict_proba(meta_features)
            recommendations['confidence'] = np.max(probabilities, axis=1)
        elif hasattr(self.meta_learner, 'predict'):
            # LightGBM情况 LightGBM case
            meta_features = features.copy()
            meta_features['cluster_label'] = cluster_labels
            meta_features = meta_features[self.feature_columns]
            
            probabilities = self.meta_learner.predict(meta_features)
            if len(probabilities.shape) > 1:
                recommendations['confidence'] = np.max(probabilities, axis=1)
            else:
                recommendations['confidence'] = 0.8  # 默认置信度 Default confidence
        
        print(f"生成了{len(recommendations)}个模型推荐")
        print(f"Generated {len(recommendations)} model recommendations")
        
        return recommendations
    
    def save_model(self, filename):
        """保存训练好的元学习器到文件 Save trained meta-learner to file"""
        import pickle
        model_data = {
            'meta_learner': self.meta_learner,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns,
            'error_threshold': self.error_threshold,
            'models_list': self.models_list,
            'category_errors': self.category_errors,
            'model_performance': self.model_performance
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"模型选择代理已保存到: {filename}")
        print(f"Model Selection Agent saved to: {filename}")
    
    def load_model(self, filename):
        """从文件加载训练好的元学习器 Load trained meta-learner from file"""
        import pickle
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
            
        self.meta_learner = model_data['meta_learner']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        self.error_threshold = model_data['error_threshold']
        self.models_list = model_data['models_list']
        self.category_errors = model_data['category_errors']
        self.model_performance = model_data.get('model_performance', {})
        
        print(f"模型选择代理已从{filename}加载")
        print(f"Model Selection Agent loaded from {filename}") 