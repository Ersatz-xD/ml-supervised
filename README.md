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

## 🤝 Let's Connect & Collaborate!

<div align="center">
  <a href="https://www.linkedin.com/in/ayaan-ahmed-khan-448600351/">
    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white&border-radius=10" alt="LinkedIn" style="margin: 5px;"/>
  </a>
  <a href="mailto:aayan.shazim@gmail.com">
    <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white&border-radius=10" alt="Email" style="margin: 5px;"/>
  </a>
  <a href="https://github.com/Ersatz-xD">
    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white&border-radius=10" alt="GitHub" style="margin: 5px;"/>
  </a>
  

<div align="center" style="margin-top: 30px;">
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="30">
  <strong style="font-size: 18px; color: #00D9FF;">Always open to interesting conversations and collaboration opportunities!</strong>
  <img src="https://user-images.githubusercontent.com/74038190/212284100-561aa473-3905-4a80-b561-0d28506553ee.gif" width="30">
</div>

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer&animation=twinkling" width="100%"/>
</div>
