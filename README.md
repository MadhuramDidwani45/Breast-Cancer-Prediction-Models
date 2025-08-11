# Breast Cancer Prediction Models: Logistic Regression vs Decision Tree

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-green)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents
- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Dataset Description](#-dataset-description)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)

## Project Overview

Breast cancer is one of the most common cancers worldwide, with early detection being crucial for successful treatment and improved patient outcomes. This project focuses on developing and comparing two machine learning models **Logistic Regression** and **Decision Tree** to predict whether a breast tumor is **benign** (non-cancerous) or **malignant** (cancerous) based on cell nuclei characteristics derived from fine needle aspirate (FNA) images.

The analysis uses the **Breast Cancer Wisconsin (Diagnostic) Dataset** containing **569 samples** with **30 numerical features** computed from digitized FNA images of breast masses.

## Problem Statement

**Goal**: Build and compare two machine learning models **Logistic Regression** and **Decision Tree** for binary classification to predict tumor malignancy based on cell nuclei measurements from breast tissue samples.

**Objective**: Assist healthcare professionals and pathologists in accurate breast cancer diagnosis, contributing to early detection and improved patient care through computational pathology.

## Dataset Description

The dataset consists of **569 instances and 32 columns**, representing breast mass samples with detailed cell nuclei measurements:

### Target Variable
| Column | Description | Values |
|--------|-------------|---------|
| **Diagnosis** | Tumor classification | M (Malignant), B (Benign) |

### Feature Categories
The 30 numerical features are grouped into three statistical categories:

#### 1. **Mean Features** (Average values)
| Feature | Description |
|---------|-------------|
| **radius_mean** | Mean of distances from center to perimeter points |
| **texture_mean** | Mean standard deviation of grayscale values |
| **perimeter_mean** | Mean tumor perimeter |
| **area_mean** | Mean tumor area |
| **smoothness_mean** | Mean local variation in radius lengths |
| **compactness_mean** | Mean (perimeter² / area - 1.0) |
| **concavity_mean** | Mean severity of concave portions |
| **concave_points_mean** | Mean number of concave portions |
| **symmetry_mean** | Mean tumor symmetry |
| **fractal_dimension_mean** | Mean "coastline approximation" - 1 |

#### 2. **Standard Error Features** (Variability measures)
- All features above with `_se` suffix (e.g., `radius_se`, `texture_se`)

#### 3. **Worst Features** (Largest/worst values)
- All features above with `_worst` suffix (e.g., `radius_worst`, `texture_worst`)

### Dataset Statistics
- **Total Samples**: 569
- **Benign Cases**: 357 (62.7%)
- **Malignant Cases**: 212 (37.3%)
- **Missing Values**: None
- **Data Quality**: Clean and preprocessed

**Data Source**: [Breast Cancer Wisconsin Dataset - Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

## 📈 Model Performance

### Models Implemented
- **Logistic Regression**: A statistical model used for binary classification tasks
- **Decision Tree**: A machine learning algorithm that uses a tree-like structure for decision-making

### Results Summary

| Model | Accuracy | Precision (Benign) | Precision (Malignant) | Recall (Benign) | Recall (Malignant) |
|-------|----------|---------------------|------------------------|------------------|-------------------|
| **Logistic Regression** | **97.4%** | 98.1% | 96.2% | 97.2% | 97.9% |
| **Decision Tree** | 94.7% | 95.3% | 93.8% | 94.4% | 95.1% |

### Best Model Performance (Logistic Regression)
```
Classification Report for Logistic Regression:
                 precision    recall  f1-score   support
Benign (B)          0.98      0.97      0.98       107
Malignant (M)       0.96      0.98      0.97        65
accuracy                                0.97       172
macro avg           0.97      0.97      0.97       172
weighted avg        0.97      0.97      0.97       172
```

### Decision Tree Performance
```
Classification Report for Decision Tree:
                 precision    recall  f1-score   support
Benign (B)          0.95      0.94      0.95       107
Malignant (M)       0.94      0.95      0.94        65
accuracy                                0.95       172
macro avg           0.95      0.95      0.95       172
weighted avg        0.95      0.95      0.95       172
```
## 📊 Results

### Key Findings

1. **Logistic Regression** achieved superior performance with **97.4% accuracy**
2. **Decision Tree** showed good performance with **94.7% accuracy**
3. **Feature correlation analysis** revealed high correlation between radius, perimeter, and area features
4. **Worst features** (largest values) showed strongest predictive power for malignancy
5. Both models demonstrated strong capability for distinguishing between benign and malignant tumors

### Clinical Insights

- **High-risk indicators**: Large radius, high texture variation, irregular perimeter
- **Feature importance**: Worst concave points, worst perimeter, and mean concavity are top predictors
- **Model reliability**: Logistic Regression showed more consistent performance across metrics
- **Interpretability**: Both models provide valuable insights for clinical decision-making

### Model Recommendations

- **Primary Choice**: **Logistic Regression** for higher accuracy and better generalization
- **Alternative**: **Decision Tree** for interpretability and feature importance insights
- Both models can serve as valuable tools for healthcare professionals in breast cancer risk assessment

## 📈 Workflow

### 1. Data Exploration & Preprocessing
- Comprehensive exploratory data analysis (EDA)
- Feature correlation analysis and visualization
- Data normalization and standardization
- Train-test split with stratification (80:20 ratio)

### 2. Model Development
- Implemented Logistic Regression classifier
- Implemented Decision Tree classifier
- Applied cross-validation for robust evaluation
- Feature scaling and normalization for Logistic Regression

### 3. Model Evaluation
- Evaluated the models based on accuracy, precision.
- Compared the performance of the models to identify the most reliable one for diabetes prediction.

### Conclusion
**Both models demonstrated effective predictive performance, but Logistic Regression provided more consistent results with higher precision and recall compared to the Decision Tree. These models can serve as valuable tools for healthcare professionals to assess Breast Cancer risk and initiate early interventions.**
---
