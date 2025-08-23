# ML-Supervised

A **hands-on repository for supervised machine learning** in Python.
This repo focuses on **understanding Linear and Logistic Regression** from the ground up, including **manual implementations** and **scikit-learn implementations** for comparison.

---

## About This Repository

Machine learning is most approachable when you **see how the algorithms work internally**.
This repository contains:

1. **Linear Regression** – predicting continuous values
2. **Logistic Regression** – predicting binary outcomes

For both, we provide:

* **From Scratch Implementation**: Using Python and NumPy to understand the **mathematics behind gradient descent**, cost functions, and model updates.
* **scikit-learn Implementation**: Using industry-standard tools to train models efficiently and evaluate them using real-world metrics.

This allows learners to **connect theory with practice** and visualize how models improve step by step.

---

## Learning Approach

This repository lets you **see and experiment with machine learning algorithms in action**:

* **From Scratch Implementations** show the step-by-step computations behind linear and logistic regression.
* **scikit-learn Implementations** demonstrate how these models are used in real-world workflows.
* Compare **metrics and visualizations** to understand the strengths and limitations of each approach.
* Learn how **gradient descent, cost functions, and predictions** work in practice.

---

## Projects Overview

### 1️⃣ Linear Regression

**Goal:** Predict continuous outcomes (e.g., housing prices).

* **Dataset:** Boston Housing dataset
* **From Scratch:**

  * Implemented **gradient descent** to minimize the **Mean Squared Error (MSE)**
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

## Key Concepts Covered

* **Gradient Descent:** Step-by-step weight updates to minimize cost
* **Cost Functions:** MSE for linear regression, log loss for logistic regression
* **Feature Scaling:** Why it matters for faster convergence
* **Model Evaluation:** Accuracy, confusion matrix, precision, recall, F1-score, ROC curve, R², MSE
* **Visualization:** How to interpret model performance graphically

---

## Folder Structure

```
ml-supervised/
├── linear_regression/
│   ├── boston_housing.ipynb
│   └── README.md
└── logistic_regression/
    ├── heart_disease.ipynb
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

3. Run all cells to explore:

* Training from scratch with gradient descent
* Training with scikit-learn
* Step-by-step visualizations of predictions and metrics

---

## Requirements

* Python 3.x
* numpy
* pandas
* scikit-learn
* matplotlib
* seaborn

Install dependencies via:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```


