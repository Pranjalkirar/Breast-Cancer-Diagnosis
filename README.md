# Breast Cancer Classification using Logistic Regression

## Overview
This project builds a machine learning model to classify breast tumors as benign or malignant using Logistic Regression.

## Dataset

This project uses the Breast Cancer Wisconsin (Diagnostic) Dataset from the UCI Machine Learning Repository.

Dataset Details:
- Total Samples: 569
- Features: 30 numerical features computed from digitized images of breast mass
- Target Classes:
    - 0 → Malignant
    - 1 → Benign
- Feature Examples:
    - Mean radius
    - Mean texture
    - Mean perimeter
    - Mean area
    - Smoothness
    - Compactness

Source:
UCI Machine Learning Repository
https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

The dataset is loaded using scikit-learn's built-in dataset loader.

## Methodology
- Train-Test Split (80/20)
- StandardScaler for feature normalization
- Logistic Regression model
- Evaluation using:
  - Accuracy
  - Confusion Matrix
  - Classification Report
  - ROC Curve
  - AUC Score

## Results
- Accuracy: ~97%
- AUC Score: ~0.99

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
Seaborn

