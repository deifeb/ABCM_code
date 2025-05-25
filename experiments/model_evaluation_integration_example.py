"""
综合模型评估集成示例 - Model Evaluation Integration Example

该示例展示如何将从ABCM_RAF.py提取的综合模型评估模块集成到ABCM系统中，
用于建立特征-模型匹配关系，为Agent 3的元学习器提供训练数据。

This example demonstrates how to integrate the comprehensive model evaluation module 
extracted from ABCM_RAF.py into the ABCM system to establish feature-model matching 
relationships and provide training data for Agent 3's meta-learner.

作者 Author: ABCM Team
创建时间 Created: 2024
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import warnings

# 导入ABCM组件 Import ABCM components
from agents import FeatureExtractionAgent, ClassificationAgent
from comprehensive_model_evaluation import ComprehensiveModelEvaluator

warnings.filterwarnings('ignore')


class ABCMModelEvaluationPipeline:
    """
    ABCM模型评估流水线
    ABCM Model Evaluation Pipeline
    
    集成特征提取、分类聚类和综合模型评估，为元学习器生成训练数据
    Integrates feature extraction, classification clustering, and comprehensive model evaluation 
    to generate training data for meta-learner
    """
    
    def __init__(self, 
                 encoding_dim: int = 12,
                 evaluation_epochs: int = 10,
                 prediction_length: int = 6):
        """
        初始化评估流水线
        Initialize evaluation pipeline
        
        参数 Args:
            encoding_dim (int): 自编码器维度 Autoencoder dimension
            evaluation_epochs (int): 评估轮数 Number of evaluation epochs
            prediction_length (int): 预测长度 Prediction length
        """
        # 初始化Agent 1: 特征提取代理 Initialize Agent 1: Feature Extraction Agent
        self.agent1 = FeatureExtractionAgent(
            encoding_dim=encoding_dim,
            autoencoder_epochs=50,
            batch_size=16
        )
        
        # 初始化Agent 2: 分类代理 Initialize Agent 2: Classification Agent
        self.agent2 = ClassificationAgent(
            accuracy_threshold=0.9,
            max_clusters=10
        )
        
        # 初始化综合模型评估器 Initialize comprehensive model evaluator
        self.model_evaluator = ComprehensiveModelEvaluator(
            evaluation_epochs=evaluation_epochs,
            prediction_length=prediction_length,
            freq='M'
        )
        
        print("ABCM模型评估流水线初始化完成")
        print("ABCM Model Evaluation Pipeline initialized")
        
    def extract_features_and_cluster(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, np.array, pd.DataFrame]:
        """
        步骤1-2: 特征提取和聚类分析
        Steps 1-2: Feature extraction and clustering analysis
        
        参数 Args:
            data (pd.DataFrame): 原始时间序列数据 Original time series data
            
        返回 Returns:
            tuple: (features, cluster_labels, grouped_data) 特征、聚类标签和分组信息
        """
        print("\n" + "=" * 80)
        print("步骤1-2: 特征提取和聚类分析")
        print("STEPS 1-2: FEATURE EXTRACTION AND CLUSTERING ANALYSIS")
        print("=" * 80)
        
        # 步骤1: 使用Agent 1提取特征 Step 1: Extract features using Agent 1
        print("🔧 正在提取特征... Extracting features...")
        features = self.agent1.extract_features(data, strategy='combined')
        print(f"✅ 特征提取完成: {features.shape[1]}个特征")
        print(f"✅ Feature extraction completed: {features.shape[1]} features")
        
        # 步骤2: 使用Agent 2进行聚类 Step 2: Perform clustering using Agent 2
        print("🔧 正在进行聚类分析... Performing clustering analysis...")
        cluster_labels, evaluation = self.agent2.classify(features, use_pca=True, use_cosine_similarity=True)
        
        n_clusters = len(np.unique(cluster_labels))
        print(f"✅ 聚类分析完成: {n_clusters}个聚类")
        print(f"✅ Clustering analysis completed: {n_clusters} clusters")
        print(f"聚类准确度 Clustering accuracy: {evaluation['accuracy']:.3f}")
        
        # 创建分组数据 Create grouped data
        grouped_data = self._create_grouped_data(data.columns.tolist(), cluster_labels)
        
        return features, cluster_labels, grouped_data
    
    def _create_grouped_data(self, series_names: List[str], cluster_labels: np.array) -> pd.DataFrame:
        """
        创建分组数据信息
        Create grouped data information
        
        参数 Args:
            series_names (list): 时间序列名称列表 List of time series names
            cluster_labels (np.array): 聚类标签 Cluster labels
            
        返回 Returns:
            pd.DataFrame: 分组数据信息 Grouped data information
        """
        # 创建DataFrame mapping series to clusters
        series_cluster_mapping = pd.DataFrame({
            'data_names': series_names,
            'category': cluster_labels
        })
        
        # 按类别分组 Group by category
        grouped_data = series_cluster_mapping.groupby('category')['data_names'].apply(list).reset_index()
        
        return grouped_data
    
    def comprehensive_model_evaluation(self, 
                                     data: pd.DataFrame,
                                     grouped_data: pd.DataFrame,
                                     save_results: bool = True) -> Tuple[List, pd.DataFrame]:
        """
        步骤3: 综合模型评估
        Step 3: Comprehensive model evaluation
        
        参数 Args:
            data (pd.DataFrame): 原始时间序列数据 Original time series data
            grouped_data (pd.DataFrame): 分组数据信息 Grouped data information
            save_results (bool): 是否保存结果 Whether to save results
            
        返回 Returns:
            tuple: (errors_by_epoch, final_error_matrix) 评估结果
        """
        print("\n" + "=" * 80)
        print("步骤3: 综合模型评估")
        print("STEP 3: COMPREHENSIVE MODEL EVALUATION")
        print("=" * 80)
        
        # 运行综合模型评估 Run comprehensive model evaluation
        errors_by_epoch, final_error_matrix = self.model_evaluator.comprehensive_evaluation(
            grouped_data=grouped_data,
            data=data,
            save_results=save_results,
            results_dir="model_evaluation_results"
        )
        
        return errors_by_epoch, final_error_matrix
    
    def prepare_meta_learning_data(self, 
                                 features: pd.DataFrame,
                                 cluster_labels: np.array,
                                 final_error_matrix: pd.DataFrame) -> Tuple[pd.DataFrame, np.array]:
        """
        步骤4: 准备元学习数据
        Step 4: Prepare meta-learning data
        
        参数 Args:
            features (pd.DataFrame): 特征数据 Feature data
            cluster_labels (np.array): 聚类标签 Cluster labels
            final_error_matrix (pd.DataFrame): 最终错误矩阵 Final error matrix
            
        返回 Returns:
            tuple: (X_meta, y_meta) 元学习的输入和输出数据
        """
        print("\n" + "=" * 80)
        print("步骤4: 准备元学习数据")
        print("STEP 4: PREPARE META-LEARNING DATA")
        print("=" * 80)
        
        # 结合特征与聚类标签 Combine features with cluster labels
        meta_features = features.copy()
        meta_features['cluster_label'] = cluster_labels
        
        # 为每个备件确定最佳模型 Determine best model for each spare part
        best_models = []
        for i, spare_part_cluster in enumerate(cluster_labels):
            # 找到该备件所属聚类的最佳模型 Find best model for the cluster this spare part belongs to
            cluster_row = final_error_matrix.iloc[spare_part_cluster]
            best_model = cluster_row.idxmin()
            best_models.append(best_model)
        
        # 编码目标标签 Encode target labels
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y_meta = label_encoder.fit_transform(best_models)
        
        print(f"✅ 元学习数据准备完成")
        print(f"✅ Meta-learning data preparation completed")
        print(f"特征维度 Feature dimensions: {meta_features.shape}")
        print(f"目标标签数量 Number of target labels: {len(np.unique(y_meta))}")
        
        # 显示目标模型分布 Display target model distribution
        from collections import Counter
        model_counts = Counter(best_models)
        print("\n🎯 目标模型分布 Target model distribution:")
        for model, count in model_counts.items():
            print(f"  {model}: {count}个备件 spare parts")
        
        return meta_features, y_meta
    
    def run_complete_evaluation_pipeline(self, 
                                       data: pd.DataFrame,
                                       save_results: bool = True) -> Dict:
        """
        运行完整的评估流水线
        Run complete evaluation pipeline
        
        参数 Args:
            data (pd.DataFrame): 原始时间序列数据 Original time series data
            save_results (bool): 是否保存结果 Whether to save results
            
        返回 Returns:
            dict: 完整评估结果 Complete evaluation results
        """
        print("🚀 开始ABCM模型评估流水线")
        print("🚀 STARTING ABCM MODEL EVALUATION PIPELINE")
        print("=" * 80)
        
        # 步骤1-2: 特征提取和聚类 Steps 1-2: Feature extraction and clustering
        features, cluster_labels, grouped_data = self.extract_features_and_cluster(data)
        
        # 步骤3: 综合模型评估 Step 3: Comprehensive model evaluation
        errors_by_epoch, final_error_matrix = self.comprehensive_model_evaluation(
            data, grouped_data, save_results
        )
        
        # 步骤4: 准备元学习数据 Step 4: Prepare meta-learning data
        X_meta, y_meta = self.prepare_meta_learning_data(
            features, cluster_labels, final_error_matrix
        )
        
        # 整理结果 Organize results
        results = {
            'features': features,
            'cluster_labels': cluster_labels,
            'grouped_data': grouped_data,
            'errors_by_epoch': errors_by_epoch,
            'final_error_matrix': final_error_matrix,
            'meta_learning_features': X_meta,
            'meta_learning_targets': y_meta,
            'n_clusters': len(np.unique(cluster_labels)),
            'n_features': features.shape[1],
            'n_models': len(self.model_evaluator.model_list)
        }
        
        print("\n" + "=" * 80)
        print("✅ ABCM模型评估流水线完成")
        print("✅ ABCM MODEL EVALUATION PIPELINE COMPLETED")
        print("=" * 80)
        print(f"聚类数量 Number of clusters: {results['n_clusters']}")
        print(f"特征数量 Number of features: {results['n_features']}")
        print(f"候选模型数量 Number of candidate models: {results['n_models']}")
        print(f"元学习数据形状 Meta-learning data shape: {X_meta.shape}")
        
        if save_results:
            # 保存元学习数据 Save meta-learning data
            X_meta.to_excel("meta_learning_features.xlsx")
            pd.DataFrame(y_meta, columns=['best_model_encoded']).to_excel("meta_learning_targets.xlsx")
            print("📁 元学习数据已保存 Meta-learning data saved")
        
        return results


def main():
    """
    ABCM模型评估流水线使用示例
    Example usage of ABCM Model Evaluation Pipeline
    """
    print("ABCM模型评估流水线示例")
    print("ABCM Model Evaluation Pipeline Example")
    print("=" * 60)
    
    # 生成模拟间歇性需求数据 Generate simulated intermittent demand data
    np.random.seed(42)
    n_series = 100  # 100个备件 100 spare parts
    n_periods = 84  # 84个时期 84 periods
    
    print("🔧 生成模拟数据... Generating simulation data...")
    
    simulation_data = pd.DataFrame()
    for i in range(n_series):
        # 生成间歇性时间序列 Generate intermittent time series
        # 使用不同的参数创建不同的需求模式 Use different parameters to create different demand patterns
        if i < n_series // 3:
            # 低频率高需求 Low frequency high demand
            demand = np.random.poisson(2.0, n_periods)
            demand = np.where(np.random.random(n_periods) > 0.8, demand, 0)
        elif i < 2 * n_series // 3:
            # 中频率中需求 Medium frequency medium demand
            demand = np.random.poisson(1.0, n_periods)
            demand = np.where(np.random.random(n_periods) > 0.6, demand, 0)
        else:
            # 高频率低需求 High frequency low demand
            demand = np.random.poisson(0.5, n_periods)
            demand = np.where(np.random.random(n_periods) > 0.4, demand, 0)
        
        simulation_data[f'spare_part_{i:03d}'] = demand
    
    print(f"✅ 模拟数据生成完成: {simulation_data.shape}")
    print(f"✅ Simulation data generated: {simulation_data.shape}")
    
    # 初始化评估流水线 Initialize evaluation pipeline
    pipeline = ABCMModelEvaluationPipeline(
        encoding_dim=12,
        evaluation_epochs=3,  # 减少轮数用于演示 Reduce epochs for demo
        prediction_length=6
    )
    
    # 运行完整流水线 Run complete pipeline
    try:
        results = pipeline.run_complete_evaluation_pipeline(
            data=simulation_data,
            save_results=True
        )
        
        print("\n🎉 流水线运行成功！ Pipeline execution successful!")
        print("现在可以使用生成的元学习数据训练Agent 3的元学习器")
        print("Now you can use the generated meta-learning data to train Agent 3's meta-learner")
        
        # 显示一些关键统计信息 Display some key statistics
        print("\n📊 关键统计信息 Key Statistics:")
        print(f"  最终错误矩阵形状 Final error matrix shape: {results['final_error_matrix'].shape}")
        print(f"  元学习特征列数 Meta-learning feature columns: {results['meta_learning_features'].shape[1]}")
        print(f"  元学习样本数 Meta-learning samples: {len(results['meta_learning_targets'])}")
        
    except Exception as e:
        print(f"❌ 流水线执行出错 Pipeline execution error: {e}")
        print("请检查依赖库安装情况")
        print("Please check dependency installations")


if __name__ == "__main__":
    main() 