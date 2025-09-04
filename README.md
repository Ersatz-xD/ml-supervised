# ML-Supervised

A **hands-on repository for supervised machine learning** in Python.
This repo focuses on **understanding Linear and Logistic Regression** from the ground up, including **manual implementations** and **scikit-learn implementations** for comparison.
It also explores **tree-based models and ensemble methods** using the Titanic dataset.

---

## About This Repository

Machine learning is most approachable when you **see how the algorithms work internally**.
This repository contains:

1. **Linear Regression** – predicting continuous values
2. **Logistic Regression** – predicting binary outcomes
3. **Decision Trees & Ensemble Methods** – predicting survival on the Titanic dataset

For the first two, we provide:

* **From Scratch Implementation**: Using Python and NumPy to understand the **mathematics behind gradient descent**, cost functions, and model updates.
* **scikit-learn Implementation**: Using industry-standard tools to train models efficiently and evaluate them using real-world metrics.

For tree-based models, we explore **Decision Trees, Random Forests, and XGBoost** to demonstrate how ensemble methods outperform single models.

This allows learners to **connect theory with practice** and visualize how models improve step by step.

---

## Learning Approach

This repository lets you **see and experiment with machine learning algorithms in action**:

* **From Scratch Implementations** (linear & logistic regression) show the computations behind training.
* **scikit-learn Implementations** demonstrate how these models are used in real-world workflows.
* **Tree & Ensemble Models** highlight practical ML approaches for classification.
* Compare **metrics and visualizations** to understand the strengths and limitations of each method.
* Learn how **gradient descent, cost functions, predictions, and feature importance** work in practice.

---

## Projects Overview

### 1️⃣ Linear Regression

**Goal:** Predict continuous outcomes (e.g., housing prices).

* **Dataset:** Boston Housing dataset
* **From Scratch:**

  * Implemented **gradient descent** to minimize **Mean Squared Error (MSE)**
  * Manually computed **gradients** and **cost function**
  * Visualized **predictions, residuals, and cost convergence**
* **scikit-learn:**

  * Used `LinearRegression()` for faster training
  * Compared predictions and metrics with scratch implementation
* **Learning Outcome:** Understand how weights are updated and how gradient descent works in practice.
* [See Linear Regression Notebook & README](./linear_regression)

---

### 2️⃣ Logistic Regression

**Goal:** Predict binary outcomes (e.g., presence of heart disease).

* **Dataset:** Heart Disease Cleveland dataset (UCI ML Repository)
* **From Scratch:**

  * Implemented **gradient descent** for **logistic loss**
  * Computed **sigmoid function, cost, and gradients** manually
  * Visualized **ROC curve, confusion matrix, and cost convergence**
* **scikit-learn:**

  * Used `LogisticRegression()` for practical modeling
  * Compared metrics like **accuracy, precision, recall, and F1-score**
* **Learning Outcome:** Understand the math behind logistic regression, the role of the sigmoid function, and how predictions are made.
* [See Logistic Regression Notebook & README](./logistic_regression)

---

### 3️⃣ 🛳️ Titanic Dataset – Decision Trees and Ensemble Methods

This project applies **tree-based machine learning models** to the famous **Titanic dataset** to predict passenger survival.
It explores the progression from a **single Decision Tree** to powerful **ensemble methods** like Random Forests and XGBoost.

* **Dataset:** [Titanic Dataset (Seaborn)](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv)
* **Models Used:**

  * 🌳 Decision Tree
  * 🌲 Random Forest (bagging ensemble)
  * 🚀 XGBoost (boosting ensemble)
* **Key Steps:**

  * Exploratory Data Analysis (EDA)
  * Preprocessing & encoding categorical variables
  * Training and evaluating models (Accuracy, Precision, Recall, F1-score)
  * Comparing feature importance across models
* **Key Insights:**

  * Random Forest and XGBoost outperform a single Decision Tree
  * Sex and Passenger Class are the most important survival predictors
  * Decision Trees are interpretable, but ensembles are more robust
* [See Tree Models Notebook & README](./tree_models)

---

## Key Concepts Covered

* **Gradient Descent:** Weight updates to minimize cost
* **Cost Functions:** MSE for linear regression, log loss for logistic regression
* **Feature Scaling:** Why it matters for convergence
* **Model Evaluation:** Accuracy, precision, recall, F1-score, ROC curve, confusion matrix, R², MSE
* **Visualization:** Interpreting model performance
* **Feature Importance:** Understanding tree-based models

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
└── tree_models/
    ├── titanic_tree_models.ipynb
    └── README.md
```

---

## Usage

1. Clone the repository:

```bash
git clone https://github.com/Ersatz-xD/ml-supervised.git
```

2. Navigate to a project folder and open the notebook:

```bash
cd ml-supervised/linear_regression
jupyter notebook boston_housing.ipynb
```

or

```bash
cd ml-supervised/logistic_regression
jupyter notebook heart_disease.ipynb
```

or

```bash
cd ml-supervised/tree_models
jupyter notebook titanic_tree_models.ipynb
```

3. Run all cells to explore training, evaluation, and visualizations.

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

Install dependencies via:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost shap
```

---
