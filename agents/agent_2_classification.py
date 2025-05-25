"""
分类代理 (Agent 2) - Classification Agent

该代理负责根据需求模式对备件进行分类，使用PCA降维、余弦相似性和K-means聚类等方法
This agent is responsible for classifying spare parts based on demand patterns using
PCA dimensionality reduction, cosine similarity, and K-means clustering

主要功能 Main Functions:
- PCA降维 PCA dimensionality reduction
- 余弦相似性特征计算 Cosine similarity feature computation
- K-means聚类 K-means clustering
- 聚类质量评估 Clustering quality evaluation
- 向代理1提供反馈 Provide feedback to Agent 1

作者 Author: ABCM Team
创建时间 Created: 2024
"""

import warnings
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
from kscorer.kscorer import KScorer

warnings.filterwarnings("ignore")


class ClassificationAgent:
    """
    代理2：分类代理
    Agent 2: Classification Agent
    
    该代理负责利用代理1提取的特征对备件进行分类，采用以下方法：
    This agent categorizes spare parts using features from Agent 1, employing:
    
    1. PCA降维 - 减少特征维度的同时保持信息
       PCA dimensionality reduction - Reduce feature dimensions while preserving information
       
    2. 余弦相似性设计 - 计算样本间的相似性特征
       Cosine similarity design - Compute similarity features between samples
       
    3. K-means聚类 - 根据需求模式将备件分组
       K-means clustering - Group spare parts based on demand patterns
       
    4. 多种聚类评估指标 - 评估聚类质量
       Multiple clustering evaluation metrics - Assess clustering quality
    """
    
    def __init__(self, accuracy_threshold=0.9, max_clusters=20):
        """
        初始化分类代理
        Initialize the Classification Agent
        
        参数 Args:
            accuracy_threshold (float): 聚类的最小准确度阈值 Minimum accuracy threshold for clustering
            max_clusters (int): 考虑的最大聚类数 Maximum number of clusters to consider
        """
        self.accuracy_threshold = accuracy_threshold
        self.max_clusters = max_clusters
        self.pca = None
        self.kmeans = None
        self.ks_scorer = None
        self.optimal_k = None
        self.cluster_labels = None
        self.cluster_metrics = {}
        
    def apply_pca(self, features, n_components=None, variance_threshold=0.95):
        """
        应用PCA进行降维
        Apply PCA for dimensionality reduction
        
        PCA能够在保持数据方差的同时减少特征维度，去除冗余信息
        PCA can reduce feature dimensions while preserving data variance, removing redundant information
        
        参数 Args:
            features (pd.DataFrame): 输入特征 Input features
            n_components (int): 主成分数量(如果为None，使用方差阈值) Number of components (if None, use variance threshold)
            variance_threshold (float): 累积方差阈值 Cumulative variance threshold
            
        返回 Returns:
            pd.DataFrame: PCA变换后的特征 PCA-transformed features
        """
        if n_components is None:
            # 根据方差阈值确定主成分数量 Determine components based on variance threshold
            pca_temp = PCA()
            pca_temp.fit(features)
            cumvar = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumvar >= variance_threshold) + 1
        
        self.pca = PCA(n_components=n_components)
        pca_features = self.pca.fit_transform(features)
        
        print(f"PCA降维：从{features.shape[1]}维降至{n_components}维")
        print(f"PCA: Reduced from {features.shape[1]} to {n_components} components")
        print(f"解释方差比例: {sum(self.pca.explained_variance_ratio_):.3f}")
        print(f"Explained variance ratio: {sum(self.pca.explained_variance_ratio_):.3f}")
        
        pca_df = pd.DataFrame(
            pca_features, 
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        return pca_df
    
    def compute_cosine_similarity_features(self, features):
        """
        计算基于余弦相似性的特征
        Compute cosine similarity-based features
        
        余弦相似性能够衡量向量间的角度相似性，不受向量长度影响
        Cosine similarity measures angular similarity between vectors, unaffected by vector length
        
        参数 Args:
            features (pd.DataFrame): 输入特征 Input features
            
        返回 Returns:
            pd.DataFrame: 增强了相似性指标的特征 Enhanced features with similarity metrics
        """
        # 计算成对余弦相似性 Compute pairwise cosine similarities
        similarity_matrix = cosine_similarity(features)
        
        # 提取基于相似性的特征 Extract similarity-based features
        similarity_features = pd.DataFrame({
            'mean_similarity': np.mean(similarity_matrix, axis=1),    # 平均相似度 Average similarity
            'max_similarity': np.max(similarity_matrix, axis=1),      # 最大相似度 Maximum similarity
            'min_similarity': np.min(similarity_matrix, axis=1),      # 最小相似度 Minimum similarity
            'std_similarity': np.std(similarity_matrix, axis=1)       # 相似度标准差 Similarity standard deviation
        })
        
        # 与原始特征合并 Combine with original features
        enhanced_features = pd.concat([features.reset_index(drop=True), 
                                     similarity_features], axis=1)
        
        print(f"添加了4个余弦相似性特征 Added 4 cosine similarity features")
        
        return enhanced_features
    
    def find_optimal_clusters(self, features, method='kscorer'):
        """
        使用指定方法寻找最优聚类数
        Find optimal number of clusters using specified method
        
        参数 Args:
            features (pd.DataFrame): 输入特征 Input features
            method (str): 寻找最优聚类数的方法 Method for finding optimal clusters
                         'kscorer': KScorer方法 KScorer method
                         'elbow': 肘部法则 Elbow method
                         'silhouette': 轮廓系数法 Silhouette method
            
        返回 Returns:
            int: 最优聚类数 Optimal number of clusters
        """
        if method == 'kscorer':
            # KScorer是专门为聚类优化设计的评估方法，参考ABCM_RAF.py实现
            # KScorer is an evaluation method specifically designed for clustering optimization, based on ABCM_RAF.py
            self.ks_scorer = KScorer()
            labels, centroids, _ = self.ks_scorer.fit_predict(features, retall=True)
            self.optimal_k = self.ks_scorer.optimal_
            
            # 获取详细的KScorer分析结果，参考ABCM_RAF.py
            # Get detailed KScorer analysis results, based on ABCM_RAF.py
            self.kscorer_ranked = self.ks_scorer.ranked_
            self.kscorer_peak_scores = self.ks_scorer.peak_scores_
            
            # 保存KScorer结果用于分析 Save KScorer results for analysis
            self.kscorer_results = {
                'optimal_k': self.optimal_k,
                'ranked_scores': self.kscorer_ranked,
                'peak_scores': self.kscorer_peak_scores,
                'labels': labels,
                'centroids': centroids
            }
            
            print(f"KScorer最优聚类数: {self.optimal_k}")
            print(f"KScorer optimal clusters: {self.optimal_k}")
            print(f"KScorer峰值分数: {self.kscorer_peak_scores}")
            print(f"KScorer peak scores: {self.kscorer_peak_scores}")
            
            # 显示聚类分布，参考ABCM_RAF.py
            # Display cluster distribution, based on ABCM_RAF.py  
            from collections import Counter
            label_count = Counter(labels)
            print("聚类分布 Cluster distribution:")
            for label, count in label_count.items():
                print(f"  Cluster {label}: {count} samples")
            
            return self.optimal_k
            
        elif method == 'silhouette':
            # 轮廓系数法：衡量聚类内紧密度和聚类间分离度
            # Silhouette method: measures intra-cluster tightness and inter-cluster separation
            scores = []
            K_range = range(2, min(self.max_clusters, len(features)//2))
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
                score = silhouette_score(features, labels)
                scores.append(score)
            
            self.optimal_k = K_range[np.argmax(scores)]
            print(f"轮廓系数法最优聚类数: {self.optimal_k}")
            print(f"Silhouette optimal clusters: {self.optimal_k}")
            return self.optimal_k
            
        elif method == 'elbow':
            # 肘部法则：寻找惯性下降曲线的拐点
            # Elbow method: find the elbow point in the inertia decline curve
            inertias = []
            K_range = range(1, min(self.max_clusters, len(features)//2))
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(features)
                inertias.append(kmeans.inertia_)
            
            # 寻找肘部点(简化方法) Find elbow point (simplified method)
            diffs = np.diff(inertias)
            diffs2 = np.diff(diffs)
            self.optimal_k = K_range[np.argmax(diffs2) + 2]
            
            print(f"肘部法则最优聚类数: {self.optimal_k}")
            print(f"Elbow method optimal clusters: {self.optimal_k}")
            return self.optimal_k
    
    def perform_clustering(self, features, n_clusters=None, use_kscorer_labels=True):
        """
        执行聚类，优先使用KScorer的标签结果
        Perform clustering, preferentially using KScorer label results
        
        参考ABCM_RAF.py的做法，如果使用KScorer方法，直接使用其标签结果而不是重新运行K-means
        Based on ABCM_RAF.py approach, if using KScorer method, directly use its label results instead of re-running K-means
        
        参数 Args:
            features (pd.DataFrame): 输入特征 Input features
            n_clusters (int): 聚类数(如果为None，则寻找最优数量) Number of clusters (if None, find optimal)
            use_kscorer_labels (bool): 是否使用KScorer的标签结果 Whether to use KScorer label results
            
        返回 Returns:
            np.array: 聚类标签 Cluster labels
        """
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(features)
        
        # 如果已经运行了KScorer并且要求使用其标签结果
        # If KScorer has been run and we want to use its label results
        if use_kscorer_labels and hasattr(self, 'kscorer_results') and self.kscorer_results is not None:
            print("使用KScorer的聚类标签结果 Using KScorer clustering label results")
            self.cluster_labels = self.kscorer_results['labels']
            
            # 创建一个虚拟的KMeans对象以保持兼容性
            # Create a dummy KMeans object for compatibility
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.kmeans.cluster_centers_ = self.kscorer_results['centroids']
            self.kmeans.labels_ = self.cluster_labels
            self.kmeans.inertia_ = np.sum([np.min([np.sum((features.iloc[i] - centroid)**2) 
                                                  for centroid in self.kscorer_results['centroids']]) 
                                          for i in range(len(features))])
        else:
            # 使用传统K-means聚类 Use traditional K-means clustering
            print("使用K-means聚类 Using K-means clustering")
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.cluster_labels = self.kmeans.fit_predict(features)
        
        # 计算聚类指标 Compute clustering metrics
        self.cluster_metrics = {
            'silhouette_score': silhouette_score(features, self.cluster_labels),
            'calinski_harabasz_score': calinski_harabasz_score(features, self.cluster_labels),
            'inertia': self.kmeans.inertia_,
            'n_clusters': n_clusters
        }
        
        # 如果有KScorer结果，添加KScorer相关指标
        # If KScorer results are available, add KScorer-related metrics
        if hasattr(self, 'kscorer_results') and self.kscorer_results is not None:
            self.cluster_metrics.update({
                'kscorer_optimal_k': self.kscorer_results['optimal_k'],
                'kscorer_peak_scores': self.kscorer_results['peak_scores']
            })
        
        # 打印聚类分布 Print cluster distribution
        from collections import Counter
        label_counts = Counter(self.cluster_labels)
        print(f"聚类分布 Cluster distribution: {dict(label_counts)}")
        print(f"轮廓系数 Silhouette score: {self.cluster_metrics['silhouette_score']:.3f}")
        
        return self.cluster_labels
    
    def evaluate_clustering_quality(self):
        """
        评估聚类质量并确定是否满足准确度阈值
        Evaluate clustering quality and determine if accuracy threshold is met
        
        改进版本，参考ABCM_RAF.py的评估方式，结合KScorer和传统指标
        Improved version based on ABCM_RAF.py evaluation approach, combining KScorer and traditional metrics
        
        返回 Returns:
            dict: 包含准确度和反馈的评估结果 Evaluation results including accuracy and feedback
        """
        if self.cluster_labels is None:
            return {'accuracy': 0, 'meets_threshold': False}
        
        # 使用轮廓系数作为基础聚类质量指标
        # Use silhouette score as base clustering quality metric
        silhouette = self.cluster_metrics['silhouette_score']
        
        # 如果有KScorer结果，结合KScorer的峰值分数进行评估
        # If KScorer results are available, combine with KScorer peak scores for evaluation
        if hasattr(self, 'kscorer_results') and self.kscorer_results is not None:
            # KScorer的峰值分数反映了聚类质量
            # KScorer peak scores reflect clustering quality
            peak_scores = self.kscorer_results['peak_scores']
            if len(peak_scores) > 0:
                max_peak_score = max(peak_scores.values()) if isinstance(peak_scores, dict) else max(peak_scores)
                # 结合轮廓系数和KScorer峰值分数
                # Combine silhouette score and KScorer peak score
                kscorer_weight = 0.6  # KScorer权重 KScorer weight
                silhouette_weight = 0.4  # 轮廓系数权重 Silhouette weight
                
                # 标准化KScorer峰值分数到[0,1]范围
                # Normalize KScorer peak score to [0,1] range
                normalized_kscorer = min(max_peak_score / 1.0, 1.0)  # 假设最大值为1.0
                normalized_silhouette = (silhouette + 1) / 2  # 轮廓系数从[-1,1]标准化到[0,1]
                
                combined_accuracy = (kscorer_weight * normalized_kscorer + 
                                   silhouette_weight * normalized_silhouette)
            else:
                # 如果没有峰值分数，仅使用轮廓系数
                # If no peak scores, use only silhouette score
                combined_accuracy = (silhouette + 1) / 2
        else:
            # 将轮廓系数标准化到[0,1]范围(轮廓系数范围是[-1,1])
            # Normalize silhouette score to [0, 1] range (silhouette is in [-1, 1])
            combined_accuracy = (silhouette + 1) / 2
        
        meets_threshold = combined_accuracy >= self.accuracy_threshold
        
        evaluation = {
            'accuracy': combined_accuracy,
            'meets_threshold': meets_threshold,
            'silhouette_score': silhouette,
            'n_clusters': self.cluster_metrics['n_clusters'],
            'cluster_metrics': self.cluster_metrics
        }
        
        # 如果有KScorer结果，添加到评估中
        # If KScorer results are available, add to evaluation
        if hasattr(self, 'kscorer_results') and self.kscorer_results is not None:
            evaluation.update({
                'kscorer_optimal_k': self.kscorer_results['optimal_k'],
                'kscorer_peak_scores': self.kscorer_results['peak_scores'],
                'kscorer_ranked': self.kscorer_results['ranked_scores']
            })
        
        return evaluation
    
    def classify(self, features, use_pca=True, use_cosine_similarity=True, clustering_method='kscorer'):
        """
        主分类方法，整合所有技术
        Main classification method that integrates all techniques
        
        这是分类代理的核心方法，整合了所有分类技术，默认使用KScorer方法进行聚类
        参考ABCM_RAF.py和classification-US.py中的实现
        This is the core method of the classification agent, integrating all classification techniques, 
        using KScorer method by default, based on ABCM_RAF.py and classification-US.py implementations
        
        参数 Args:
            features (pd.DataFrame): 来自代理1的输入特征 Input features from Agent 1
            use_pca (bool): 是否应用PCA Whether to apply PCA
            use_cosine_similarity (bool): 是否使用余弦相似性特征 Whether to use cosine similarity features
            clustering_method (str): 聚类方法 Clustering method ('kscorer', 'silhouette', 'elbow')
            
        返回 Returns:
            tuple: (cluster_labels, evaluation_results) 聚类标签和评估结果
        """
        print("开始分类过程... Starting classification process...")
        print(f"使用聚类方法 Using clustering method: {clustering_method}")
        
        # 预处理特征 Preprocess features
        processed_features = features.copy()
        
        # 移除任何NaN值 Remove any NaN values
        processed_features = processed_features.dropna()
        
        if use_cosine_similarity:
            print("计算余弦相似性特征... Computing cosine similarity features...")
            processed_features = self.compute_cosine_similarity_features(processed_features)
        
        if use_pca and processed_features.shape[1] > 10:
            print("应用PCA... Applying PCA...")
            processed_features = self.apply_pca(processed_features)
        
        # 使用指定的方法寻找最优聚类数 Find optimal clusters using specified method
        print(f"使用{clustering_method}方法寻找最优聚类数...")
        print(f"Finding optimal clusters using {clustering_method} method...")
        optimal_k = self.find_optimal_clusters(processed_features, method=clustering_method)
        
        # 执行聚类 Perform clustering
        print("执行聚类... Performing clustering...")
        cluster_labels = self.perform_clustering(processed_features, n_clusters=optimal_k)
        
        # 评估聚类质量 Evaluate clustering quality
        evaluation = self.evaluate_clustering_quality()
        
        print(f"分类完成。准确度: {evaluation['accuracy']:.3f}")
        print(f"Classification completed. Accuracy: {evaluation['accuracy']:.3f}")
        
        # 如果使用了KScorer，显示详细分析
        # If KScorer was used, show detailed analysis
        if clustering_method == 'kscorer' and hasattr(self, 'kscorer_results'):
            print(f"KScorer最优聚类数: {optimal_k}")
            print(f"KScorer optimal clusters: {optimal_k}")
            if hasattr(self, 'kscorer_peak_scores'):
                print(f"KScorer峰值分数: {self.kscorer_peak_scores}")
        
        return cluster_labels, evaluation
    
    def provide_feedback_to_agent1(self, evaluation):
        """
        根据聚类评估向代理1提供反馈信号
        Provide feedback signal to Agent 1 based on clustering evaluation
        
        参数 Args:
            evaluation (dict): 聚类评估结果 Clustering evaluation results
            
        返回 Returns:
            dict: 给代理1的反馈信号 Feedback signal for Agent 1
        """
        feedback = {
            'accuracy': evaluation['accuracy'],
            'meets_threshold': evaluation['meets_threshold']
        }
        
        if not evaluation['meets_threshold']:
            # 根据聚类结果建议改进 Suggest improvements based on clustering results
            if evaluation['silhouette_score'] < 0.3:
                feedback['suggestion'] = 'adjust_autoencoder_dim'
                feedback['message'] = "轮廓系数较低，建议调整自编码器维度"
            elif evaluation['n_clusters'] < 3:
                feedback['suggestion'] = 'add_features'
                feedback['message'] = "聚类数量过少，建议添加更多特征"
            else:
                feedback['suggestion'] = 'adjust_autoencoder_dim'
                feedback['message'] = "聚类质量不佳，尝试调整自编码器维度"
                
            feedback['message'] += f" | 聚类准确度 {evaluation['accuracy']:.3f} 低于阈值 {self.accuracy_threshold}"
        else:
            feedback['suggestion'] = 'maintain_strategy'
            feedback['message'] = "聚类满足准确度要求 Clustering meets accuracy requirements"
        
        return feedback
    
    def predict_new_data(self, new_features):
        """
        使用训练好的模型预测新数据的聚类标签
        Predict cluster labels for new data using trained models
        
        参考classification-US.py中的做法，优先使用KScorer进行预测
        Based on classification-US.py approach, preferentially use KScorer for prediction
        
        参数 Args:
            new_features (pd.DataFrame): 要分类的新特征 New features to classify
            
        返回 Returns:
            np.array: 预测的聚类标签 Predicted cluster labels
        """
        processed_features = new_features.copy().dropna()
        
        # 如果使用了PCA，需要先进行PCA变换
        # If PCA was used, apply PCA transformation first
        if self.pca is not None:
            processed_features = pd.DataFrame(
                self.pca.transform(processed_features),
                columns=[f'PC{i+1}' for i in range(self.pca.n_components_)]
            )
        
        # 优先使用KScorer进行预测，参考classification-US.py
        # Preferentially use KScorer for prediction, based on classification-US.py
        if self.ks_scorer is not None:
            try:
                print("使用KScorer预测新数据标签 Using KScorer to predict new data labels")
                predicted_labels = self.ks_scorer.predict(processed_features, retall=False)
                
                # 显示预测结果分布
                # Display prediction result distribution
                from collections import Counter
                label_count = Counter(predicted_labels)
                print("新数据预测标签分布 New data prediction label distribution:")
                for label, count in label_count.items():
                    print(f"  Cluster {label}: {count} samples")
                
                return predicted_labels
            except Exception as e:
                print(f"KScorer预测失败，回退到K-means方法: {e}")
                print(f"KScorer prediction failed, falling back to K-means: {e}")
        
        # 如果KScorer不可用，使用K-means进行预测
        # If KScorer is not available, use K-means for prediction
        if self.kmeans is None:
            raise ValueError("模型未训练。请先调用classify()方法。Model not trained. Call classify() first.")
        
        print("使用K-means预测新数据标签 Using K-means to predict new data labels")
        predicted_labels = self.kmeans.predict(processed_features)
        
        return predicted_labels
    
    def save_model(self, filename):
        """保存训练好的模型到文件 Save trained models to file"""
        import pickle
        model_data = {
            'pca': self.pca,
            'kmeans': self.kmeans,
            'ks_scorer': self.ks_scorer,
            'optimal_k': self.optimal_k,
            'cluster_metrics': self.cluster_metrics,
            'accuracy_threshold': self.accuracy_threshold,
            # 添加KScorer相关结果 Add KScorer-related results
            'kscorer_results': getattr(self, 'kscorer_results', None),
            'kscorer_ranked': getattr(self, 'kscorer_ranked', None),
            'kscorer_peak_scores': getattr(self, 'kscorer_peak_scores', None)
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"分类模型已保存到: {filename}")
        print(f"Classification model saved to: {filename}")
        
        # 如果有KScorer结果，也保存详细的分析结果
        # If KScorer results are available, also save detailed analysis results
        if hasattr(self, 'kscorer_ranked') and self.kscorer_ranked is not None:
            ranked_filename = filename.replace('.pkl', '_kscorer_ranked.xlsx')
            self.kscorer_ranked.to_excel(ranked_filename)
            print(f"KScorer详细分析结果已保存到: {ranked_filename}")
            print(f"KScorer detailed analysis results saved to: {ranked_filename}")
    
    def load_model(self, filename):
        """从文件加载训练好的模型 Load trained models from file"""
        import pickle
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.pca = model_data.get('pca')
        self.kmeans = model_data.get('kmeans')
        self.ks_scorer = model_data.get('ks_scorer')
        self.optimal_k = model_data.get('optimal_k')
        self.cluster_metrics = model_data.get('cluster_metrics', {})
        self.accuracy_threshold = model_data.get('accuracy_threshold', 0.9)
        
        # 加载KScorer相关结果 Load KScorer-related results
        self.kscorer_results = model_data.get('kscorer_results')
        self.kscorer_ranked = model_data.get('kscorer_ranked')
        self.kscorer_peak_scores = model_data.get('kscorer_peak_scores')
        
        print(f"分类模型已从 {filename} 加载")
        print(f"Classification model loaded from {filename}")
        
        if self.kscorer_results is not None:
            print(f"KScorer最优聚类数: {self.kscorer_results['optimal_k']}")
            print(f"KScorer optimal clusters: {self.kscorer_results['optimal_k']}")
    
    def show_kscorer_analysis(self, save_chart=False, chart_filename=None):
        """
        显示KScorer分析结果，参考ABCM_RAF.py中的show()方法
        Display KScorer analysis results, based on show() method in ABCM_RAF.py
        
        参数 Args:
            save_chart (bool): 是否保存图表 Whether to save chart
            chart_filename (str): 图表保存文件名 Chart save filename
        """
        if self.ks_scorer is None:
            print("KScorer未运行，无法显示分析结果")
            print("KScorer not run, cannot display analysis results")
            return
        
        print("\n" + "="*60)
        print("📊 KScorer 聚类分析结果 KScorer Clustering Analysis Results")
        print("="*60)
        
        try:
            # 显示KScorer图表，参考ABCM_RAF.py
            # Display KScorer chart, based on ABCM_RAF.py
            self.ks_scorer.show()
            
            print(f"\n最优聚类数 Optimal clusters: {self.optimal_k}")
            
            if hasattr(self, 'kscorer_peak_scores') and self.kscorer_peak_scores:
                print("\n峰值分数 Peak scores:")
                if isinstance(self.kscorer_peak_scores, dict):
                    for k, score in self.kscorer_peak_scores.items():
                        print(f"  K={k}: {score:.4f}")
                else:
                    print(f"  {self.kscorer_peak_scores}")
            
            if hasattr(self, 'kscorer_ranked') and self.kscorer_ranked is not None:
                print(f"\n排名结果形状 Ranked results shape: {self.kscorer_ranked.shape}")
                print("前5个聚类数的综合得分 Top 5 cluster numbers' scores:")
                print(self.kscorer_ranked.head())
            
            print("="*60)
            
        except Exception as e:
            print(f"显示KScorer分析结果时出错: {e}")
            print(f"Error displaying KScorer analysis results: {e}")
    
    def export_kscorer_results(self, base_filename):
        """
        导出KScorer结果到Excel文件，参考ABCM_RAF.py的做法
        Export KScorer results to Excel files, based on ABCM_RAF.py approach
        
        参数 Args:
            base_filename (str): 基础文件名 Base filename
        """
        if not hasattr(self, 'kscorer_results') or self.kscorer_results is None:
            print("没有KScorer结果可导出")
            print("No KScorer results to export")
            return
        
        try:
            # 导出排名结果 Export ranked results
            if hasattr(self, 'kscorer_ranked') and self.kscorer_ranked is not None:
                ranked_filename = f"{base_filename}_kscorer_ranked.xlsx"
                self.kscorer_ranked.to_excel(ranked_filename)
                print(f"KScorer排名结果已保存到: {ranked_filename}")
                print(f"KScorer ranked results saved to: {ranked_filename}")
            
            # 导出聚类标签 Export cluster labels
            if 'labels' in self.kscorer_results:
                labels_df = pd.DataFrame(
                    self.kscorer_results['labels'], 
                    columns=['cluster_label']
                )
                labels_filename = f"{base_filename}_cluster_labels.xlsx"
                labels_df.to_excel(labels_filename)
                print(f"聚类标签已保存到: {labels_filename}")
                print(f"Cluster labels saved to: {labels_filename}")
            
            # 导出峰值分数 Export peak scores
            if 'peak_scores' in self.kscorer_results:
                peak_scores_df = pd.DataFrame([self.kscorer_results['peak_scores']])
                peak_filename = f"{base_filename}_peak_scores.xlsx"
                peak_scores_df.to_excel(peak_filename)
                print(f"峰值分数已保存到: {peak_filename}")
                print(f"Peak scores saved to: {peak_filename}")
                
        except Exception as e:
            print(f"导出KScorer结果时出错: {e}")
            print(f"Error exporting KScorer results: {e}") 