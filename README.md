# ML-Supervised

A **hands-on repository for supervised machine learning** in Python.
This repo focuses on **understanding Linear and Logistic Regression** from the ground up, including **manual implementations** and **scikit-learn implementations** for comparison.
It also explores **tree-based models, ensemble methods, and neural networks** for more advanced tasks.

---

## About This Repository

Machine learning is most approachable when you **see how the algorithms work internally**.
This repository contains:

1. **Linear Regression** – predicting continuous values
2. **Logistic Regression** – predicting binary outcomes
3. **Decision Trees & Ensemble Methods** – predicting survival on the Titanic dataset
4. **Neural Networks (CNN on CIFAR-10)** – multiclass image classification

For the first two, we provide:

* **From Scratch Implementation**: Using Python and NumPy to understand the **mathematics behind gradient descent**, cost functions, and model updates.
* **scikit-learn Implementation**: Using industry-standard tools to train models efficiently and evaluate them using real-world metrics.

For tree-based and neural models, we use **scikit-learn and TensorFlow/Keras** to show how modern machine learning handles complex, high-dimensional data.

This allows learners to **connect theory with practice** and visualize how models improve step by step.

---

## Projects Overview

### 1️⃣ Linear Regression

*(Predicting continuous outcomes like housing prices)*
- [See Linear Regression Notebook & README](./linear_regression)

---

### 2️⃣ Logistic Regression

*(Predicting binary outcomes like heart disease diagnosis)*
- [See Logistic Regression Notebook & README](./logistic_regression)

---

### 3️⃣ 🛳️ Titanic Dataset – Decision Trees and Ensemble Methods

*(Survival prediction with Decision Trees, Random Forest, and XGBoost)*
- [See Tree Models Notebook & README](./tree_models)

---

### 4️⃣ 🧠 Neural Networks – Multiclass Classification (CIFAR-10)

*This project applies a **Convolutional Neural Network (CNN)** using **TensorFlow/Keras** to classify images from the **CIFAR-10 dataset** into 10 categories.*
- [See Neural Networks Notebook & README](./neural_networks)

---

## Folder Structure

```
ml-supervised/
├── linear_regression/
│   ├── boston_housing.ipynb
│   └── README.md
├── logistic_regression/
│   ├── heart_disease.ipynb
│   └── README.md
├── tree_models/
│   ├── titanic_tree_models.ipynb
│   └── README.md
└── neural_networks/
    ├── cifar10-classifier.ipynb
    └── README.md
```

---

## Requirements

* Python 3.x
* numpy
* pandas
* scikit-learn
* matplotlib
* seaborn
* xgboost
* shap
* tensorflow

Install dependencies via:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost shap tensorflow
```

---

