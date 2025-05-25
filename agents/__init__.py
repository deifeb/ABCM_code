"""
ABCM代理模块 - ABCM Agents Module

该模块包含基于代理的协作模型的三个主要代理：
This module contains the three main agents of the Agent-Based Collaborative Model:

- 代理1：特征提取代理 - Agent 1: Feature Extraction Agent
- 代理2：分类代理 - Agent 2: Classification Agent  
- 代理3：模型选择代理 - Agent 3: Model Selection Agent

作者 Author: ABCM Team
创建时间 Created: 2024
"""

from .agent_1_feature_extraction import FeatureExtractionAgent
from .agent_2_classification import ClassificationAgent
from .agent_3_model_selection import ModelSelectionAgent

__all__ = [
    'FeatureExtractionAgent',
    'ClassificationAgent', 
    'ModelSelectionAgent'
]

# 代理说明 Agent Descriptions
AGENT_DESCRIPTIONS = {
    'FeatureExtractionAgent': {
        'chinese': '特征提取代理：负责从时间序列数据中提取显式和隐式特征',
        'english': 'Feature Extraction Agent: Responsible for extracting explicit and implicit features from time series data'
    },
    'ClassificationAgent': {
        'chinese': '分类代理：使用PCA、余弦相似性和K-means聚类对备件进行分类',
        'english': 'Classification Agent: Classifies spare parts using PCA, cosine similarity, and K-means clustering'
    },
    'ModelSelectionAgent': {
        'chinese': '模型选择代理：使用LightGBM元学习为每个备件选择最佳预测模型',
        'english': 'Model Selection Agent: Uses LightGBM meta-learning to select the best forecasting model for each spare part'
    }
} 