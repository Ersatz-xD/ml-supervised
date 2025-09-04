# 🛳️ Titanic Dataset - Decision Trees and Ensemble Methods

This project applies **tree-based machine learning models** to the famous **Titanic dataset** to predict passenger survival.  
It explores the progression from a **single Decision Tree** to powerful **ensemble methods** like Random Forests and XGBoost.

---

## 📌 Project Overview

- **Dataset**: [Titanic Dataset (Seaborn)](https://github.com/mwaskom/seaborn-data/blob/master/titanic.csv)  
- **Goal**: Predict whether a passenger survived based on features like age, class, sex, and embarkation port.  
- **Models Used**:
  - 🌳 Decision Tree  
  - 🌲 Random Forest (bagging ensemble)  
  - 🚀 XGBoost (boosting ensemble)

---

## ⚙️ Steps in the Notebook

1. **Introduction & Dataset Loading**  
   - Overview of the Titanic dataset.  
   - Exploratory Data Analysis (EDA) with plots.  

2. **Data Preprocessing**  
   - Handle missing values.  
   - Encode categorical variables.  
   - Train-test split.  

3. **Model Training**  
   - Train Decision Tree, Random Forest, and XGBoost.  
   - Evaluate using Accuracy, Precision, Recall, and F1-score.  
   - Visualize confusion matrices.  

4. **Model Comparison**  
   - Side-by-side performance comparison (table + bar plot).  

5. **Feature Importance**  
   - Identify the most important features for survival prediction.  
   - Compare importance across all three models.  

---

## 📊 Key Insights

- **Random Forest and XGBoost** outperform a single Decision Tree due to better generalization.  
- **Sex and Passenger Class** are the most important features for predicting survival.  
- Ensemble methods balance accuracy and robustness, while Decision Trees are more interpretable.  

---

## 🚀 How to Run

1. Clone the main repo:
   ```bash
   git clone https://github.com/Ersatz-xD/ml-supervised.git
   cd ml-supervised/tree_models
   ```

2. Install dependencies:

   ```bash
   pip install scikit-learn xgboost matplotlib seaborn shap
   ```

3. Open the Jupyter Notebook:

   ```bash
   jupyter notebook titanic_tree_models.ipynb
   ```

---

## 📂 Repository Structure

```
ml-supervised/
│── tree_models/
│   ├── titanic_tree_models.ipynb   # Main notebook
│   ├── README.md                   # This file
```

---

## 📖 What I Learned

* How Decision Trees split data and where they risk overfitting.
* How Random Forests stabilize predictions with bagging.
* How XGBoost improves performance with boosting.
* Best practices in model evaluation and interpretation.

---



