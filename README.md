# ML-Supervised Learning Repository

A **hands-on repository for supervised machine learning** practice in Python, developed as part of the [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?source=search) on Coursera.

This repository focuses on **understanding core ML algorithms from the ground up**, featuring **manual implementations** alongside **scikit-learn implementations** for comprehensive learning and comparison.

---

## 🎯 About This Project

This repository was created as a **practical learning companion** for the Machine Learning Specialization course. The goal is to bridge the gap between theoretical understanding and hands-on implementation by providing:

- **From-scratch implementations** using Python and NumPy to understand the mathematics
- **Industry-standard implementations** using scikit-learn and TensorFlow/Keras
- **Real-world datasets** to demonstrate practical applications
- **Step-by-step explanations** connecting theory with practice

Machine learning becomes truly approachable when you **see how algorithms work internally**. Each project in this repository demonstrates both the mathematical foundations and practical applications of key supervised learning techniques.

---

## 📚 Learning Path

### 🔢 **Fundamentals: Regression Models**
Understanding the core building blocks of supervised learning

### 1️⃣ **Linear Regression**
*Predicting continuous outcomes like housing prices*

**What you'll learn:**
- Cost function optimization using gradient descent
- Feature scaling and normalization techniques
- Model evaluation with RMSE, MAE, and R² metrics
- Comparing manual implementation vs. scikit-learn

**Implementation approaches:**
- **From Scratch**: Pure Python/NumPy implementation showing gradient descent step-by-step
- **scikit-learn**: Industry-standard approach with built-in optimization

📂 [**Explore Linear Regression →**](./linear_regression)

---

### 2️⃣ **Logistic Regression**
*Predicting binary outcomes like medical diagnosis*

**What you'll learn:**
- Sigmoid function and probability interpretation
- Cross-entropy cost function and its derivatives
- Classification metrics: accuracy, precision, recall, F1-score
- Decision boundary visualization

**Implementation approaches:**
- **From Scratch**: Manual gradient descent for logistic regression
- **scikit-learn**: Professional-grade implementation with advanced features

📂 [**Explore Logistic Regression →**](./logistic_regression)

---

### 🌳 **Advanced Methods: Tree-Based Models**
Exploring ensemble learning and feature importance

### 3️⃣ **🛳️ Titanic Survival Prediction**
*Decision Trees, Random Forests, and Gradient Boosting*

**What you'll learn:**
- Decision tree construction and pruning
- Ensemble methods: bagging and boosting
- Feature importance analysis with SHAP values
- Hyperparameter tuning with cross-validation

**Models covered:**
- Decision Trees
- Random Forest
- XGBoost
- Feature importance visualization

📂 [**Explore Tree Models →**](./tree_models)

---

### 🧠 **Deep Learning: Neural Networks**
Scaling to high-dimensional data with deep learning

### 4️⃣ **CIFAR-10 Image Classification**
*Convolutional Neural Networks with TensorFlow/Keras*

**What you'll learn:**
- CNN architecture design and layer functions
- Image preprocessing and data augmentation
- Training deep networks: optimization, regularization
- Multi-class classification evaluation

**Deep learning concepts:**
- Convolutional and pooling layers
- Backpropagation in neural networks
- Overfitting prevention techniques
- Model performance visualization

📂 [**Explore Neural Networks →**](./neural_networks)

---

## 📁 Repository Structure

```
ml-supervised/
├── 📊 linear_regression/
│   ├── boston_housing.ipynb          # Complete linear regression tutorial
│   ├── README.md                     # Detailed project explanation
│   
│
├── 🏥 logistic_regression/
│   ├── heart_disease.ipynb           # Binary classification tutorial
│   ├── README.md                     # Project documentation
│   
│
├── 🛳️ tree_models/
│   ├── titanic_tree_models.ipynb     # Ensemble methods tutorial
│   ├── README.md                     # Tree models explanation
│   
│
├── 🧠 neural_networks/
│   ├── cifar10_classifier.ipynb      # CNN implementation
│   ├── README.md                     # Deep learning guide


```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ recommended
- Basic understanding of Python programming
- Familiarity with NumPy and Pandas (helpful but not required)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Ersatz-xD/ml-supervised.git
cd ml-supervised
```

2. **Install dependencies:**
```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost shap tensorflow jupyter
```

3. **Launch Jupyter and start learning:**
```bash
jupyter notebook
```

### Recommended Learning Order
1. Start with **Linear Regression** to understand gradient descent
2. Move to **Logistic Regression** for classification concepts
3. Explore **Tree Models** for ensemble learning
4. Advance to **Neural Networks** for deep learning

---

## 🎓 Course Connection

This repository serves as a **practical companion** to the [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction?source=search) by Stanford University and DeepLearning.AI on Coursera.

**How this repository enhances your learning:**
- **Reinforces course concepts** with hands-on coding practice
- **Provides additional examples** beyond course assignments  
- **Compares different implementation approaches** (manual vs. library)
- **Includes real-world datasets** for practical experience
- **Offers extended explanations** of mathematical concepts

Each project folder contains detailed README files that reference specific course concepts and provide additional context for deeper understanding.

---

## 🛠️ Technologies Used

| Category | Tools | Purpose |
|----------|-------|---------|
| **Core ML** | NumPy, scikit-learn | Manual implementations and standard ML algorithms |
| **Deep Learning** | TensorFlow, Keras | Neural network construction and training |
| **Data Processing** | Pandas, NumPy | Data manipulation and preprocessing |
| **Visualization** | Matplotlib, Seaborn | Creating plots and model visualizations |
| **Advanced ML** | XGBoost, SHAP | Gradient boosting and model interpretability |
| **Environment** | Jupyter Notebook | Interactive development and documentation |

---

## 🤝 Contributing

This repository is designed for learning purposes. If you find errors, have suggestions, or want to add new examples:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-addition`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-addition`)
5. Open a Pull Request

---

## 📖 Additional Resources

- [Machine Learning Specialization Course](https://www.coursera.org/specializations/machine-learning-introduction?source=search)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Machine Learning Yearning by Andrew Ng](https://www.deeplearning.ai/machine-learning-yearning/)

---

## 🌟 Acknowledgments

- **Stanford University & DeepLearning.AI** for the excellent Machine Learning Specialization course
- **Open source community** for the amazing tools that make learning ML accessible
- **Dataset providers** for making real-world data available for educational purposes

---
