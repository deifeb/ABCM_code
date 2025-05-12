# 间歇性需求预测系统

## 项目简介

本项目是一个基于多种算法间歇性需求预测系统，主要针对间歇性需求预测场景。项目实现了特征提取、分类和选择功能，并包含了多个数据集的处理和测试。核心算法包括ABCM（Agent-Based Collaborative Model）方法，该方法结合了元学习和多种时间序列预测模型，用于优化间歇性需求的预测效果。

## 功能特点

- **特征提取**：从时间序列数据中提取多种统计特征，包括平均需求间隔(ADI)、方差系数平方(CV²)、近似熵等
- **时间序列分类**：基于提取的特征对时间序列进行分类，以选择最适合的预测模型
- **多模型预测**：支持多种预测模型，包括：
  - DeepAR
  - ARIMA
  - ETS
  - SBJ (Syntetos-Boylan-Johnston)
  - DeepRenewal (Flat/Exact/Hybrid)
- **强化学习框架**：使用Q-learning算法自动选择最优预测模型
- **多数据集支持**：支持处理多个数据集，包括RAF数据、荷兰制造公司数据和美国汽车零部件数据

## 项目结构

```
├── ABCM-RAF.py          # RAF数据集的ABCM算法实现
├── ABCM-Dutch.py        # 荷兰制造公司数据集的ABCM算法实现
├── ABCM-US.py           # 美国汽车零部件数据集的ABCM算法实现
├── RAF_test.py          # RAF数据集测试脚本
├── Direct testing Dutch.py  # 荷兰数据集直接测试脚本
├── Direct testing RAF.py    # RAF数据集直接测试脚本
├── Direct testing US.py     # 美国数据集直接测试脚本
├── feature_extraction.py    # 特征提取功能实现
├── classification-Dutch.py  # 荷兰数据集分类脚本
├── classification-US.py     # 美国数据集分类脚本
└── README.md            # 项目说明文档
```

## 依赖库

项目依赖以下Python库：

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
```

## 核心组件

### 1. 特征提取

`feature_extraction.py` 和各分类文件中的 `compute_features` 函数实现了从时间序列中提取9种关键特征：

- F1: 平均需求间隔 (ADI)
- F2: 方差系数平方 (CV²)
- F3: 近似熵
- F4: 零值百分比
- F5: 超出 [mean-std, mean+std] 范围的值的百分比
- F6: 线性最小二乘回归系数
- F7: 连续变化的平均绝对值
- F8: 最后一个块的平方和占整个序列的比例
- F9: 序列末尾连续零值的百分比

### 2. ABCM-RAF 算法

`ABCM-RAF.py` 实现了在RAF上进行ABCM模型的训练，主要组件包括：
- 基于RAF子集A的训练结果形成ABCM训练数据或使用Q-learning：学习最优模型选择策略
- 多种预测模型集成：包括DeepAR、ARIMA、ETS、SBJ等
- 模型评估：使用IntermittentEvaluator评估预测性能

### 3. 数据处理

各个脚本中的 `format_trans` 函数负责将原始数据转换为适合GluonTS库处理的格式，包括：

- 时间索引添加
- 训练集和测试集分割
- 特征处理和格式转换

## 注意事项

- 项目中的数据路径需要根据实际环境进行调整
- 部分模型训练可能需要较长时间和较大计算资源
- 对于大规模数据集，建议先进行特征提取和分类，再选择合适的模型进行预测
