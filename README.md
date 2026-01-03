# Credit Card Fraud Detection (Deep Learning)

A machine learning project that detects fraudulent credit card transactions using deep learning techniques. This project focuses on handling highly imbalanced financial data, feature preprocessing, and training a neural network to accurately identify fraud while minimizing false positives.

---

## Project Overview

Credit card fraud poses a significant challenge due to:
- Extremely imbalanced datasets where fraud cases are rare
- High cost of false negatives
- Real-time detection requirements in financial systems

This project implements a deep learning–based fraud detection pipeline that preprocesses transaction data, trains a neural network model, and evaluates performance using industry-relevant metrics.

---

## Key Features

- Data preprocessing and normalization  
- Handling class imbalance with appropriate strategies  
- Deep learning model built using PyTorch  
- Model training, validation, and evaluation  
- Performance metrics focused on fraud detection effectiveness  
- Modular and extensible project structure  

---

## Tech Stack

Languages
- Python  

Machine Learning and Data
- PyTorch  
- NumPy  
- Pandas  
- Scikit-learn  

Data Visualization
- Matplotlib  
- Seaborn  

---

## Data Description

The dataset contains anonymized credit card transactions, where:
- The target variable indicates fraudulent (1) or legitimate (0) transactions
- Fraud cases represent a very small percentage of the total data

This imbalance makes traditional accuracy an unreliable metric, requiring more robust evaluation strategies.

---

## Model Architecture

- Fully connected neural network  
- ReLU activation functions  
- Binary classification output layer  
- Optimized using cross-entropy loss  
- Trained with attention to overfitting and class imbalance  

---

## Evaluation Metrics

The model is evaluated using:
- Precision  
- Recall  
- F1-score  
- Confusion matrix  

---
