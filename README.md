# Network-Intrusion-Detection-System-Using-TabTransformer

## A Transformer-Based Approach for Cybersecurity Threat Detection Using the CIC-IDS2017 Dataset

##  Overview

This project implements a TabTransformer-based Network Intrusion Detection System (NIDS) to classify network traffic as either benign or malicious using the CIC-IDS2017 dataset. The model leverages Transformer Encoder layers to effectively handle both numerical and categorical features, addressing the challenges posed by class imbalance and complex feature interactions.


## Key Features

✅ TabTransformer Architecture: Efficient handling of tabular data using Multi-Head Self-Attention (MHSA) and Feed-Forward Networks (FFN).

✅ Class Imbalance Handling: Balanced dataset achieved using Borderline-SMOTE and RandomUnderSampler techniques.

✅ Class Weights in Loss Function: Cross-Entropy Loss with weights [1.0, 5.0] to prioritize minority class samples.

✅ Evaluation Metrics: Comprehensive evaluation with Accuracy, Precision, Recall, F1 Score, MCC, and AUC-ROC Score.

✅ Visualization Tools: Training-validation curves, confusion matrix heatmaps, and classification report heatmaps for deeper insights.


## Dataset

Dataset: [CIC-IDS2017](https://www.kaggle.com/datasets/dhoogla/cicids2017/data)

Classes: Benign and Attack

Imbalance Ratio: 84.92% Benign | 15.08% Attack

## Technologies Used

Python 3.x

PyTorch

Transformers

SMOTE (imbalanced-learn)

Scikit-learn

Seaborn & Matplotlib


## Results

Accuracy: 98.56%

Precision: 97.38%

Recall: 99.80%

F1 Score: 98.58%

MCC: 0.971

AUC-ROC Score: 0.9987


## Visualizations

Confusion Matrix Heatmap

Classification Report Heatmap

Training and Validation Accuracy Curves

Training and Validation Loss Curves
