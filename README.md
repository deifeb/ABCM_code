# Agent-Based Collaborative Model (ABCM) for Intermittent Demand Forecasting

This repository contains the implementation of an Agent-Based Collaborative Model (ABCM) for intermittent demand forecasting of spare parts. The model consists of three cooperative agents that work together to provide accurate forecasting through feature extraction, classification, and model selection.

## Project Overview

The ABCM system consists of three main agents:

1. **Agent 1 - Feature Extraction Agent** (`agents/feature_extraction_agent.py`)
   - Extracts features using expert knowledge and autoencoders
   - Devises multiple extraction strategies based on feature types and combinations
   - Provides valuable information to Agent 2 through strategy sharing

2. **Agent 2 - Classification Agent** (`agents/classification_agent.py`)
   - Integrates explicit and implicit features from Agent 1
   - Employs PCA for dimensionality reduction, cosine similarity design, and K-means clustering
   - Categorizes spare parts based on different demand patterns
   - Sends feedback signals to Agent 1 if accuracy requirements are not met

3. **Agent 3 - Model Selection Agent** (`agents/model_selection_agent.py`)
   - Combines features from Agent 1 and clustering labels from Agent 2
   - Uses LightGBM meta-learning to identify the most suitable forecasting model
   - Tracks forecasting errors and triggers adjustments when thresholds are exceeded

## Project Structure

```
ABCM/
├── agents/
│   ├── __init__.py
│   ├── feature_extraction_agent.py    # Agent 1: Feature extraction
│   ├── classification_agent.py        # Agent 2: Classification and clustering
│   └── model_selection_agent.py       # Agent 3: Meta-learning model selection
├── models/
│   ├── __init__.py
│   ├── forecasting_models.py          # Implementation of forecasting models
│   └── autoencoder.py                 # Autoencoder for feature extraction
├── utils/
│   ├── __init__.py
│   ├── data_preprocessing.py          # Data transformation utilities
│   ├── evaluation_metrics.py         # Evaluation metrics and tools
│   └── file_utils.py                 # File I/O utilities
├── experiments/
│   ├── ABCM_RAF.py                    # Main training script for RAF dataset
│   ├── ABCM_US.py                     # Training script for US dataset
│   ├── ABCM_Dutch.py                  # Training script for Dutch dataset
│   ├── direct_testing_RAF.py          # Direct prediction on RAF data
│   ├── direct_testing_US.py           # Direct prediction on US data
│   └── direct_testing_Dutch.py        # Direct prediction on Dutch data
├── config/
│   └── config.py                      # Configuration parameters
├── requirements.txt                   # Project dependencies
└── README.md                         # This file
```

## Features

### Feature Extraction (Agent 1)
- **Expert Features (F1-F9)**:
  - F1: Average Demand Interval (ADI)
  - F2: Square of Coefficient of Variation (CV²)
  - F3: Approximate Entropy
  - F4: Percentage of Values Outside [mean ± std] Range
  - F5: Linear Least Squares Regression Coefficient
  - F6: Average Absolute Value of Consecutive Changes
  - F7: Ratio of Last Chunk's Sum of Squares to Total Series
  - F8: Percentage of Consecutive Zeros at the End

- **Autoencoder Features**: Deep learning-based latent features for capturing complex patterns

### Classification (Agent 2)
- PCA for dimensionality reduction
- Cosine similarity design
- K-means clustering with KScorer for optimal cluster determination
- Accuracy threshold validation (0.9)

### Model Selection (Agent 3)
- **Forecasting Models**:
  - DeepAR
  - Deep Renewal (Flat, Exact, Hybrid variants)
  - SBA
  - ETS (Exponential Smoothing)

- **Meta-Learning**: LightGBM for intelligent model selection

## Configuration

Modify `config/config.py` to adjust model parameters:

```python
# Model parameters
PREDICTION_LENGTH = 6
LEARNING_RATE = 0.01
EPOCHS = 10
AUTOENCODER_DIM = 12
ACCURACY_THRESHOLD = 0.9
```

## Evaluation Metrics

The system uses specialized metrics for intermittent demand:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Symmetric Mean Absolute Percentage Error (sMAPE)
- Mean Absolute Scale Error (MASE)

## Datasets

The model has been tested on:
- RAF (Royal Air Force) spare parts dataset
- US automotive spare parts dataset
- Dutch automotive spare parts dataset

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 1.0.0
- tensorflow >= 2.6.0
- gluonts >= 0.10.0
- lightgbm >= 3.2.0
- kscorer
- antropy
- statsmodels
- deeprenewal
...

### 🎯 Key Innovation

The **separation of concerns**:
- **Cross-validation meta-learning**: Intensive offline training to establish feature-model relationships using RandomForest with comprehensive cross-validation
- **ABCM System**: Efficient online inference using pretrained knowledge
- **Fallback mechanism**: Real-time meta-learning when pretrained knowledge is unavailable

This design ensures both **performance** (using proven relationships) and **flexibility** (adapting to new scenarios).

A detailed explanation of DeepRenewal is presented in the following paper:
"Turkmen A C, Wang Y, Januschowski T. Intermittent demand forecasting with deep renewal processes[J]. arXiv preprint arXiv:1911.10416, 2019."
