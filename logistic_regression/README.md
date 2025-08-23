# Heart Disease Prediction - Logistic Regression (Scratch & scikit-learn)

A comparative implementation of **Logistic Regression** in Python, both **from scratch using gradient descent** and using **scikit-learn**, demonstrated on the **Heart Disease Cleveland dataset** from the UCI Machine Learning Repository.

---

## Project Overview

This project covers:

1. **Logistic Regression from Scratch**

   * Implemented using **gradient descent**.
   * Manually computed **cost function**, **gradients**, and parameter updates.
   * Visualized **cost convergence**, **confusion matrix**, **ROC curve**, and **predicted probabilities**.

2. **Logistic Regression with scikit-learn**

   * Leveraged `LogisticRegression()` for training and prediction.
   * Feature scaling applied for better convergence.
   * Evaluated using **accuracy, confusion matrix, classification report, and ROC curve**.
   * Visualized **confusion matrix and ROC curve** for comparison.

3. **Comparison**

   * Compared the **performance metrics** of scratch vs scikit-learn models.
   * Side-by-side visualization of **confusion matrices** and **ROC curves**.

---

## Dataset

The **Heart Disease Cleveland dataset** is a well-known benchmark in medical machine learning research.

* Contains **patient medical records** with 14 clinically relevant features:

  * Age, sex, chest pain type, resting blood pressure, cholesterol levels, ECG results, etc.
* Target variable indicates heart disease presence:

  * `0` → No disease
  * `1-4` → Disease present (different levels)
* Framed as a **binary classification problem** for this project:

  * `0` → No disease
  * `1-4` → Disease present
* ⚠️ Contains missing values that are handled during preprocessing.

---

## Folder Structure

```
ml-supervised/
└── logistic_regression/
    └── heart_disease.ipynb
```

---

## Usage

1. Clone the repo:

```bash
git clone https://github.com/Ersatz-xD/ml-supervised.git
```

2. Navigate to the project folder:

```bash
cd ml-supervised/logistic_regression
```

3. Open the notebook:

```bash
jupyter notebook heart_disease.ipynb
```

4. Run all cells to see:

   * Logistic Regression from scratch with gradient descent
   * Logistic Regression with scikit-learn
   * Performance comparison and visualizations

---

## Performance Comparison: Scratch vs scikit-learn

| Metric         | Scratch   | scikit-learn |
| -------------- | --------- | ------------ |
| Train Accuracy | 0.8760    | 0.8677       |
| Test Accuracy  | 0.8525    | 0.8525       |
| Precision      | 0.8275    | 0.8275       |
| Recall         | 0.85      | 0.85         |
| F1-Score       | 0.84      | 0.84         |

*Confusion matrices and ROC curves are visualized in the notebook for direct comparison.*

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
