# Linear Regression - From Scratch & scikit-learn

A comparative implementation of **Linear Regression** in Python, both **from scratch using gradient descent** and using **scikit-learn**, demonstrated on the **Boston Housing dataset**.

---

## Project Overview

This project covers:

1. **Linear Regression from Scratch**

   * Implemented using **gradient descent**.
   * Manually computed **cost function**, **gradients**, and parameter updates.
   * Visualized **training and test predictions**, **cost convergence**, and **residuals**.

2. **Linear Regression with scikit-learn**

   * Leveraged `LinearRegression()` for training and prediction.
   * Feature scaling applied for better convergence.
   * Evaluated using **Mean Squared Error (MSE)** and **R² score**.
   * Visualized predictions and residuals.

3. **Comparison**

   * Compared the **performance metrics** of scratch vs scikit-learn models.
   * Visual comparison of predictions vs actual values.

---

## Folder Structure

```
ml-supervised/
└── linear_regression/
    └── boston_housing.ipynb
```

---

## Usage

1. Clone the repo:

```bash
git clone https://github.com/Ersatz-xD/ml-supervised.git
```

2. Navigate to the project folder:

```bash
cd ml-supervised/linear_regression
```

3. Open the notebook:

```bash
jupyter notebook boston_housing.ipynb
```

4. Run all cells to see:

   * Training from scratch with gradient descent
   * Training with scikit-learn
   * Performance comparison and visualizations

---

## Performance Comparison: Scratch vs scikit-learn

| Metric      | Scratch       | scikit-learn |
|------------|---------------|--------------|
| Train MSE  | 21.737105     | 21.641413    |
| Test MSE   | 24.917444     | 24.291119    |
| Train R²   | 0.749784      | 0.750886     |
| Test R²    | 0.660219      | 0.668759     |

---

## Requirements

* Python 3.x
* numpy
* pandas
* scikit-learn
* matplotlib

Install dependencies via:

```bash
pip install numpy pandas scikit-learn matplotlib
```

---
