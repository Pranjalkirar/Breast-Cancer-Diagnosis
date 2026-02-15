# =====================================================
# Breast Cancer Classification using Logistic Regression
# =====================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

import warnings
warnings.filterwarnings("ignore")

def main():
    # 1. Load Dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    print("Dataset Loaded Successfully")
    print("Shape of dataset:", X.shape)
    print("Classes:", data.target_names)

    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3. Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Model Training
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    # 5. Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 6. Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

    # 7. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=data.target_names,
                yticklabels=data.target_names)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # 8. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {round(auc_score, 3)}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nAUC Score:", round(auc_score, 4))
    print("\nProject Completed Successfully 🚀")


if __name__ == "__main__":
    main()
