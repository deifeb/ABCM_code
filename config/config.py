"""
Configuration Module

This module contains all configuration parameters for the ABCM system.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class FeatureExtractionConfig:
    """Configuration for Feature Extraction Agent"""
    # Autoencoder parameters
    encoding_dim: int = 12
    autoencoder_epochs: int = 50
    batch_size: int = 16
    
    # Expert features parameters
    chunk_size: int = 12  # For F6 calculation
    k_chunks: int = 4     # For F8 calculation
    
    # Strategy parameters
    default_strategy: str = 'combined'  # 'expert', 'autoencoder', 'combined'


@dataclass
class ClassificationConfig:
    """Configuration for Classification Agent"""
    # Clustering parameters
    accuracy_threshold: float = 0.9
    max_clusters: int = 20
    
    # PCA parameters
    variance_threshold: float = 0.95
    min_components: int = 2
    
    # Clustering method
    clustering_method: str = 'kscorer'  # 'kscorer', 'silhouette', 'elbow'
    
    # Cosine similarity
    use_cosine_similarity: bool = True
    use_pca: bool = True


@dataclass
class ModelSelectionConfig:
    """Configuration for Model Selection Agent"""
    # Meta-learning parameters
    use_lightgbm: bool = True
    test_size: float = 0.2
    random_state: int = 42
    
    # LightGBM parameters
    lgb_params: Dict[str, Any] = None
    
    # Random Forest parameters (fallback)
    rf_param_grid: Dict[str, List] = None
    
    # Error threshold
    error_threshold: float = None  # Will be calculated as mean of all model errors
    
    def __post_init__(self):
        if self.lgb_params is None:
            self.lgb_params = {
                'objective': 'multiclass',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': self.random_state
            }
        
        if self.rf_param_grid is None:
            self.rf_param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }


@dataclass
class ForecastingConfig:
    """Configuration for Forecasting Models"""
    # General parameters
    prediction_length: int = 6
    freq: str = 'M'  # Monthly frequency
    
    # DeepAR parameters
    deepar_config: Dict[str, Any] = None
    
    # Deep Renewal parameters
    deeprenewal_config: Dict[str, Any] = None
    
    # Evaluation parameters
    quantiles: List[float] = None
    
    def __post_init__(self):
        if self.deepar_config is None:
            self.deepar_config = {
                'learning_rate': 0.01,
                'epochs': 10,
                'cell_type': 'gru',
                'num_cells': 128,
                'dropout_rate': 0.1,
                'num_layers': 2,
                'batch_size': 128,
                'clip_gradient': 5.48481845049343,
                'weight_decay': 0.001
            }
        
        if self.deeprenewal_config is None:
            self.deeprenewal_config = {
                'learning_rate': 0.01,
                'epochs': 10,
                'cell_type': 'lstm',
                'num_cells': 64,
                'dropout_rate': 0.3,
                'num_lags': 1,
                'clip_gradient': 5.170127652392614,
                'weight_decay': 0.01
            }
        
        if self.quantiles is None:
            self.quantiles = [0.25, 0.5, 0.75, 0.85]


@dataclass
class DataConfig:
    """Configuration for Data Processing"""
    # Data paths
    data_dir: str = "data"
    results_dir: str = "results"
    models_dir: str = "saved_models"
    
    # Data processing
    start_date: str = '1996-01-01'
    train_test_split_ratio: float = 0.8  # 80% for training
    
    # File formats
    data_file_format: str = '.xlsx'
    model_file_format: str = '.pkl'
    results_file_format: str = '.xlsx'


@dataclass
class SystemConfig:
    """System-wide configuration"""
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Parallel processing
    n_jobs: int = -1
    
    # Logging
    log_level: str = 'INFO'
    log_to_file: bool = True
    log_file: str = 'abcm.log'
    
    # Memory management
    max_memory_usage: str = '8GB'
    
    # Model candidate list
    candidate_models: List[str] = None
    
    def __post_init__(self):
        if self.candidate_models is None:
            self.candidate_models = [
                'DeepAR', 
                'DeepRenewal Flat', 
                'DeepRenewal Exact', 
                'DeepRenewal Hybrid',
                'Croston', 
                'SBJ', 
                'ARIMA', 
                'ETS', 
                'NPTS'
            ]


@dataclass
class ABCMConfig:
    """Main ABCM Configuration"""
    # Sub-configurations
    feature_extraction: FeatureExtractionConfig = None
    classification: ClassificationConfig = None
    model_selection: ModelSelectionConfig = None
    forecasting: ForecastingConfig = None
    data: DataConfig = None
    system: SystemConfig = None
    
    # ABCM-specific parameters
    max_iterations: int = 10
    convergence_threshold: float = 0.01
    feedback_enabled: bool = True  # Enable inter-agent feedback mechanism
    
    def __post_init__(self):
        if self.feature_extraction is None:
            self.feature_extraction = FeatureExtractionConfig()
        if self.classification is None:
            self.classification = ClassificationConfig()
        if self.model_selection is None:
            self.model_selection = ModelSelectionConfig()
        if self.forecasting is None:
            self.forecasting = ForecastingConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.system is None:
            self.system = SystemConfig()


# Default configuration instance
DEFAULT_CONFIG = ABCMConfig()


def get_config() -> ABCMConfig:
    """Get default configuration"""
    return DEFAULT_CONFIG


def update_config(**kwargs) -> ABCMConfig:
    """Update configuration with new values"""
    config = get_config()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            if isinstance(value, dict):
                # Update nested configuration
                nested_config = getattr(config, key)
                for nested_key, nested_value in value.items():
                    if hasattr(nested_config, nested_key):
                        setattr(nested_config, nested_key, nested_value)
            else:
                setattr(config, key, value)
    
    return config


def save_config(config: ABCMConfig, filename: str) -> None:
    """Save configuration to file"""
    import json
    from dataclasses import asdict
    
    config_dict = asdict(config)
    
    with open(filename, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Configuration saved to {filename}")


def load_config(filename: str) -> ABCMConfig:
    """Load configuration from file"""
    import json
    
    with open(filename, 'r') as f:
        config_dict = json.load(f)
    
    # Reconstruct config object
    config = ABCMConfig()
    
    # Update with loaded values
    for key, value in config_dict.items():
        if hasattr(config, key):
            if isinstance(value, dict):
                nested_config = getattr(config, key)
                for nested_key, nested_value in value.items():
                    if hasattr(nested_config, nested_key):
                        setattr(nested_config, nested_key, nested_value)
            else:
                setattr(config, key, value)
    
    print(f"Configuration loaded from {filename}")
    return config


# Environment-specific configurations
def get_raf_config() -> ABCMConfig:
    """Get configuration optimized for RAF dataset"""
    config = get_config()
    
    # Optimize for RAF dataset (84 periods)
    config.forecasting.prediction_length = 6
    config.feature_extraction.encoding_dim = 12
    config.classification.accuracy_threshold = 0.9
    
    return config


def get_us_config() -> ABCMConfig:
    """Get configuration optimized for US dataset"""
    config = get_config()
    
    # Optimize for US dataset (51 periods)
    config.forecasting.prediction_length = 6
    config.feature_extraction.encoding_dim = 12
    config.classification.accuracy_threshold = 0.9
    
    return config


def get_dutch_config() -> ABCMConfig:
    """Get configuration optimized for Dutch dataset"""
    config = get_config()
    
    # Optimize for Dutch dataset
    config.forecasting.prediction_length = 6
    config.feature_extraction.encoding_dim = 12
    config.classification.accuracy_threshold = 0.9
    
    return config 