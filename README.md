# Data-Science-Capstone-Project-2
📊 Semiconductor Sensor Data – Pass/Fail Classification

Machine Learning | Imbalanced Data Handling | Industrial Data Analytics

🔎 Project Overview

This major project focuses on building a machine learning classification system to predict Pass/Fail outcomes using high-dimensional industrial sensor data.

The dataset contains multiple numerical sensor readings collected during a manufacturing process. The objective is to develop a robust predictive model capable of handling:

High-dimensional feature space

Missing values

Constant (non-informative) features

Severe class imbalance

The final outcome is a trained and serialized machine learning model ready for deployment.

🎯 Problem Statement

Manufacturing systems generate large volumes of sensor data. Detecting whether a product will Pass or Fail quality inspection is critical for:

Reducing production waste

Improving yield rate

Early fault detection

Process optimization

This project builds classification models to automate that prediction process.

📂 Dataset

File used: sensor-data.csv

Target variable: Pass/Fail

Remapped from {-1, 1} → {0, 1}

The dataset contains:

Time-based process measurements (Time column removed)

High-dimensional numerical sensor readings

Missing values

Class imbalance

🛠 Tech Stack
Category	Tools Used
Language	Python
Data Handling	Pandas, NumPy
Visualization	Matplotlib, Seaborn
ML Models	Random Forest, Logistic Regression, SVM
Model Selection	GridSearchCV
Imbalance Handling	SMOTE
Evaluation Metrics	Accuracy, F1-score, Confusion Matrix
Model Saving	Joblib
Environment	Jupyter Notebook
🔄 Project Workflow
1️⃣ Data Exploration

Loaded dataset and inspected dimensions

Checked missing values

Reviewed feature distributions

2️⃣ Data Cleaning

✔ Dropped columns with >70% missing values
✔ Removed constant features (VarianceThreshold)
✔ Removed Time column (non-predictive)
✔ Used median imputation (due to skewness detected via mean-median difference analysis)
✔ Remapped target labels to binary format

Mean–median distribution analysis confirmed positive skewness, making median imputation more robust than mean imputation.

3️⃣ Handling Class Imbalance

The dataset showed significant imbalance in Pass/Fail classes.

To address this:

Applied SMOTE (Synthetic Minority Oversampling Technique)

Performed resampling only on training data

Prevented data leakage during cross-validation

4️⃣ Feature Preparation

Split into training (80%) and testing (20%)

Standardized features for Logistic Regression and SVM

Retained raw features for Random Forest

🤖 Models Implemented
🔹 Random Forest

Hyperparameter tuning via GridSearchCV

Evaluated using F1-score

Confusion matrix visualization

Selected as best-performing model

🔹 Logistic Regression

Scaled input features

Hyperparameter tuning on regularization strength (C)

🔹 Support Vector Machine (SVM)

RBF kernel

Grid search optimization

📊 Model Evaluation

Evaluation metrics used:

Accuracy

F1-score (primary metric due to class imbalance)

Confusion Matrix

Key Observation

Although some models achieved high accuracy, low F1-scores revealed difficulty in predicting the minority class, confirming class imbalance as a central challenge.

Random Forest performed best overall and was selected as the final model.
