# Time Series Analysis and Forecasting System

## Project Overview

This project is a time series analysis and forecasting system based on multiple algorithms, primarily designed for intermittent demand forecasting scenarios. It implements feature extraction, classification, and prediction functionalities, and includes processing and testing for multiple datasets. The core algorithm is the ABCM method, which combines meta-learning and various time series forecasting models to optimize intermittent demand forecasting performance.

## Features

- **Feature Extraction**：Extracts various statistical features from time series data, including Average Demand Interval (ADI), Coefficient of Variation squared (CV²), Approximate Entropy, and more.
- **Time Series Classification**：Classifies time series based on extracted features to select the most suitable forecasting model.
- **Multi-model Forecasting**：Supports multiple forecasting models, including:
  - DeepAR
  - ARIMA
  - ETS
  - SBJ (Syntetos-Boylan-Johnston)
  - DeepRenewal (Flat/Exact/Hybrid)
- **Meta-learning Framework**：Uses a meta-learning framework to automatically select the optimal forecasting model for each class.
- **Multi-dataset Support**：Supports processing of multiple datasets, including RAF data, Dutch manufacturing data, and U.S. auto parts data.

## Project Structure

```
├── ABCM-RAF.py             # ABCM algorithm implementation for the RAF dataset
├── ABCM-Dutch.py           # ABCM algorithm implementation for the Dutch manufacturing dataset
├── ABCM-US.py              # ABCM algorithm implementation for the U.S. auto parts dataset
├── RAF_test.py             # Test script for the RAF dataset
├── Direct testing Dutch.py # Direct test script for the Dutch dataset
├── Direct testing RAF.py   # Direct test script for the RAF dataset
├── Direct testing US.py    # Direct test script for the U.S. dataset
├── feature_extraction.py   # Implementation of feature extraction functionality
├── classification-Dutch.py # Classification script for the Dutch dataset
├── classification-US.py    # Classification script for the U.S. dataset
└── README.md               # Project documentation
```

## Dependencies

This project depends on the following Python libraries:

```
numpy
pandas
tensorflow
statsmodels
scipy
antropy
gluonts
deeprenewal
kscorer
pickle
...
```

## Core Components

### 1. Feature Extraction

The compute_features function in feature_extraction.py and classification scripts extracts 9 key features from time series:

- F1: Average Demand Interval (ADI)
- F2: Coefficient of Variation squared (CV²)
- F3: Approximate Entropy
- F4: Percentage of Zero Values
- F5: Percentage of values outside the [mean-std, mean+std] range
- F6: Linear Least Squares Regression Coefficient
- F7: Mean Absolute Change of consecutive values
- F8: Proportion of squared sum of the last block to the whole series
- F9: Percentage of consecutive zero values at the end of the series

### 2. ABCM

ABCM-RAF.py implements the ABCM model training on the RAF dataset, including:
- Building a meta-learner using results from RAF subset A to learn optimal model selection strategies
- Integration of multiple forecasting models, including DeepAR, ARIMA, ETS, SBJ, etc.
- Model Evaluation: Evaluated using the IntermittentEvaluator to assess forecasting performance

### 3. Data Processing

The format_trans function in various scripts is responsible for transforming raw data into a format compatible with the GluonTS library, including:

- Adding time indices
- Splitting into training and test sets
- Feature processing and format conversion

## Notes

- Data paths in the project need to be adjusted according to your environment
- Training some models may require significant time and computational resources
- For large-scale datasets, it is recommended to perform feature extraction and classification first, and then select an appropriate model for forecasting
