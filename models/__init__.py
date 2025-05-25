"""
模型模块 - Models Module

该模块包含ABCM系统的预测模型和支持组件
This module contains forecasting models and supporting components for the ABCM system.

主要组件 Main Components:
- 候选预测模型 - Candidate Forecasting Models

作者 Author: ABCM Team
创建时间 Created: 2024
"""

from .candidate_forecasting_models import ForecastingModels

# 注意：自编码器已集成到特征提取代理中，不再单独导入
# Note: Autoencoder is integrated into the Feature Extraction Agent, no longer imported separately

__all__ = [
    'ForecastingModels'
]

# 模型说明 Model Descriptions  
MODEL_DESCRIPTIONS = {
    'ForecastingModels': {
        'chinese': '预测模型容器：包含9种专门用于间歇性需求预测的候选模型',
        'english': 'Forecasting Models Container: Contains 9 candidate models specifically designed for intermittent demand forecasting'
    }
} 