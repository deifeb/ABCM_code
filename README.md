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

#### 🎯 **Feature-Model Mapping Source**

The model selection agent uses **two modes** for determining the best forecasting model:

1. **Pretrained Meta-learner Mode (Recommended)**:
   - The feature-model matching relationship is obtained from **cross-validation meta-learning training in `experiments/Cross-validation meta-learning-lightgbm.py`**
   - Uses RandomForest with GridSearchCV to learn optimal feature-model mappings
   - The trained meta-learner encapsulates domain knowledge from extensive cross-validation evaluation
   - **File**: `pretrained_metalearner.pkl`
   - **Structure**: Features (F1-F9 + autoencoder + cluster_label) → Best Model

2. **Real-time Meta-learning Mode (Fallback)**:
   - When no pretrained meta-learner is available
   - Uses LightGBM to learn feature-model relationships on-the-fly
   - Less stable and requires more computational resources

**Workflow Integration**:
```
Cross-validation Training → Meta-learner Training → Model Selection Agent → Best Model Prediction
         ↓                        ↓                        ↓                      ↓
1. Load feature data         2. Train RandomForest      3. Load meta-learner   4. Feature→Model
2. Load model errors         3. Cross-validation        4. Predict best model  5. Return prediction
3. Merge datasets           4. GridSearchCV             
4. Train meta-learner       5. Save trained model
```

## Agent Information Flow and Collaboration

### 🔄 **Agent Interaction Architecture**

The ABCM system implements a sophisticated agent_based collaboration framework with the following information flow:

#### **(1) Agent 1 → Agent 2: Feature Strategy Sharing**
Agent 1 extracts features using expert knowledge and autoencoders, devising multiple extraction strategies based on feature types and combinations. This valuable information is passed on to Agent 2, showcasing strategy sharing among the agents. Upon receiving the feature extraction strategies, Agent 2 categorizes spare parts utilizing methods like PCA for dimensionality reduction, cosine similarity design, and K-means clustering, assessing the results with various metrics. If the evaluation falls short of accuracy requirements, feedback in the form of reward signals is sent to Agent 1 to adjust the extraction strategy.

#### **(2) Agent 2 → Agent 3: Classified Demand Patterns**
Agent 2 integrates the explicit and implicit features from Agent 1 and employs methods such as PCA for dimensionality reduction, cosine similarity design, and K-means clustering to classify spare parts based on different demand patterns. By leveraging similarities, it categorizes demand data into distinct groups, thereby identifying spare part patterns with similar demand behaviors. Once Agent 2 successfully meets the accuracy threshold 0.9 through clustering using Agent 1's strategy, the resultant categories are forwarded as input to Agent 3. Otherwise, it updates the feature extraction strategy.

#### **(3) Agent 3 → Agent 1: Model Performance Feedback**
Agent 3 combines the features from Agent 1 and clustering labels from Agent 2 to integrate demand forecasting errors from candidate models. Using LightGBM meta-learning, Agent 3 identifies the most suitable forecasting model for each spare part among the candidates and tracks the forecasting error of each category of spare parts. When the forecasting error exceeds an error threshold, which can be described as the mean of all single-model forecasting errors, it triggers Agent 1 to adjust the latent dimension of the autoencoder or adjust the addition or reduction of explicit features. Agent 1 maintains a strategy pool, with real-time performance metrics driving adaptive switching, ensuring alignment between strategies and dynamic demand patterns.

### 🎯 **Model Matching Relationship Establishment**

The core innovation in Agent 3 lies in how it establishes the initial feature-model matching relationships. This process involves comprehensive empirical evaluation rather than predetermined rules:

#### **Phase 1: Comprehensive Model Benchmarking**
Following the methodology in `experiments/ABCM_RAF.py`, the system:
- **Extracts comprehensive features**: Uses Agent 1 to extract F1-F9 expert features and autoencoder features
- **Performs clustering analysis**: Uses Agent 2 with KScorer to categorize spare parts into different demand pattern groups  
- **Evaluates all candidate models**: For each cluster category, trains and tests all forecasting models (DeepAR, Deep Renewal variants, SBA, ETS)
- **Calculates performance metrics**: Uses IntermittentEvaluator to compute MRAE, MASE, MAAPE, MAE for each model-category combination
- **Ensures statistical reliability**: Conducts multiple evaluation rounds (e.g., 10 epochs) to ensure stable and reliable results

#### **Phase 2: Optimal Model Selection**
- **Generates error matrices**: Creates comprehensive error matrices showing each model's performance on each spare part
- **Identifies best performers**: Selects optimal models based on minimum error principle for each spare part category
- **Discovers patterns**: Analyzes relationships between feature characteristics and optimal model choices

#### **Phase 3: Meta-Learning Training**
- **Fuses datasets**: Combines feature data with optimal model labels to create meta-learning training data
- **Trains meta-learner**: Uses RandomForest with GridSearchCV for cross-validation and hyperparameter optimization
- **Learns mappings**: Establishes mappings from feature space (F1-F9 + autoencoder + cluster labels) to best model recommendations

This empirical approach ensures that model selection decisions are grounded in actual forecasting performance rather than theoretical assumptions, creating a robust foundation for the meta-learning process.

## Usage
First, use comprehensive_model_evaluation.py to generate the most appropriate predictive model for each category of data
### 🎯 Method 1: Using Pretrained Meta-learner (Recommended)

```python
from abcm_main_system import ABCMSystem
from config import get_config

# Load configuration
config = get_config()

# Initialize with pretrained meta-learner from Cross-validation training
metalearner_path = "experiments/pretrained_metalearner.pkl"  # pkl file
abcm = ABCMSystem(config, pretrained_metalearner_path=metalearner_path)

# Load your data
# data = pd.read_excel("your_data.xlsx", index_col=0)

# Train the system (will use meta-learner for model selection)
# results = abcm.train(data)

# Make predictions
# predictions = abcm.predict(new_data)
```

### ⚙️ Method 2: Real-time Meta-learning (Fallback)

```python
from abcm_main_system import ABCMSystem
from config import get_config

# Initialize without pretrained meta-learner
abcm = ABCMSystem(get_config())

# Train with real-time meta-learning
# results = abcm.train(data)
```

### 📋 Generating Pretrained Meta-learner

```python
# Step 1: Run Cross-validation meta-learning script to generate meta-learner
import subprocess

# This will generate pretrained_metalearner.pkl
subprocess.run(["python", "experiments/Cross-validation meta-learning-lightgbm.py"])

# Step 2: Use the generated meta-learner in ABCM system
metalearner_path = "experiments/pretrained_metalearner.pkl"
abcm = ABCMSystem(config, pretrained_metalearner_path=metalearner_path)
```

### Using Individual Agents

```python
from agents.feature_extraction_agent import FeatureExtractionAgent
from agents.classification_agent import ClassificationAgent
from agents.model_selection_agent import ModelSelectionAgent

# Initialize agents
feature_agent = FeatureExtractionAgent()
classification_agent = ClassificationAgent()

# Initialize model selection agent with pretrained meta-learner
model_selection_agent = ModelSelectionAgent(
    pretrained_metalearner_path="experiments/pretrained_metalearner.pkl"
)

# Extract features
features = feature_agent.extract_features(data)

# Classify spare parts
labels = classification_agent.classify(features)

# Select best model (will use meta-learner if available)
best_models = model_selection_agent.predict_best_models(features, labels)
```

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

## Architecture Details

### 🔄 Training Workflow

1. **Offline Phase** (Cross-validation meta-learning):
   - Load feature data (F1-F9 + autoencoder features)
   - Load model error data from comparative evaluations
   - Merge datasets by cluster labels
   - Train RandomForest meta-learner with GridSearchCV
   - Save trained meta-learner for future use

2. **Online Phase** (ABCM System):
   - Load pretrained meta-learner (or use real-time meta-learning)
   - Extract features from new data
   - Classify spare parts into clusters
   - Use meta-learner to select best model for each spare part
   - Apply iterative feedback for continuous improvement

### 🎯 Key Innovation

The **separation of concerns**:
- **Cross-validation meta-learning**: Intensive offline training to establish feature-model relationships using RandomForest with comprehensive cross-validation
- **ABCM System**: Efficient online inference using pretrained knowledge
- **Fallback mechanism**: Real-time meta-learning when pretrained knowledge is unavailable

This design ensures both **performance** (using proven relationships) and **flexibility** (adapting to new scenarios).

A detailed explanation of DeepRenewal is presented in the following paper:
"Turkmen A C, Wang Y, Januschowski T. Intermittent demand forecasting with deep renewal processes[J]. arXiv preprint arXiv:1911.10416, 2019."