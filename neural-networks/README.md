# 🧠 Neural Networks - Multiclass Classification (CIFAR-10)

This project applies a **Convolutional Neural Network (CNN)** using **TensorFlow/Keras** to classify images from the **CIFAR-10 dataset** into 10 categories.
It extends beyond simple linear/logistic models and decision trees, showing how neural networks capture **complex, non-linear patterns** in data.

---

## 📌 Project Overview

* **Dataset**: CIFAR-10 (60,000 color images, 10 classes, 32×32 resolution).
* **Goal**: Train a CNN to classify objects like airplanes, cars, cats, and ships.
* **Challenges**:

  * Small image size and low resolution make classification harder.
  * Some classes (e.g., cat vs. dog) are visually very similar.

---

## 🔧 What I Did

* Built and trained a CNN using **TensorFlow/Keras**.
* Experimented with **activation functions** (ReLU, softmax).
* Compared **Adam optimizer** vs. vanilla gradient descent.
* Applied best practices:

  * **Dropout** for regularization
  * **EarlyStopping** and **ModelCheckpoint**
  * **Data Augmentation** (rotation, shifts, flips)
* Monitored training with **TensorBoard**.
* Evaluated results with:

  * Test accuracy
  * Confusion matrices
  * Per-class precision, recall, F1-scores

---

## 📊 Results

* **Baseline model (no augmentation)**: \~69% accuracy
* **Augmented model**: \~69% accuracy, but more robust on some classes (Automobile, Dog, Frog)
* Key takeaway: augmentation didn’t boost accuracy immediately (due to early stopping + small network), but helps generalization and robustness.

---

## 📝 What I Learned

* How CNNs are structured (convolutions, pooling, dense layers).
* The role of **activation functions** and **loss functions** in classification.
* Why **Adam optimizer** accelerates training.
* How to detect **overfitting vs. underfitting** using learning curves.
* Practical use of **TensorBoard** and **scikit-learn metrics**.

---

## 🚀 Next Steps

* Train with **more epochs** for augmented data.
* Try **deeper CNN architectures** (VGG, ResNet).
* Add **learning rate schedules**.
* Use **transfer learning** from pretrained models on ImageNet.

---

📂 Part of my **[ml-supervised](../)** repository, which also includes:

* 📈 Linear Regression
* 🔢 Logistic Regression
* 🌳 Decision Trees & Ensemble Methods
* 🧠 Neural Networks (this project)

---

Do you also want me to add **badges** (TensorFlow, Keras, Python, CIFAR-10 dataset) at the top, same as you’re doing for other READMEs in this repo?
