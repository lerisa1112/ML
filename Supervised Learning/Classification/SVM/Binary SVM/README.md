# 🤖 What is Binary SVM?

**Binary SVM** (Support Vector Machine) is a supervised machine learning algorithm used to classify data into **two classes**. 
It aims to find the best possible **decision boundary (hyperplane)** that separates the two classes with the **maximum margin**.

- Works well for **linearly and non-linearly separable** data using kernel tricks.
- Example: Classifying emails as **spam** or **not spam**.

---

## 🛠️ **How to Use Binary SVM**

1. **Collect Labeled Binary Data**  
   ~ The target label `y` should contain **two distinct classes** (e.g., 0 and 1).

2. **Choose SVM Classifier**  
   ~ Use `SVC()` from scikit-learn for binary classification.

3. **Select Kernel Type**  
   ~ Use `'linear'`, `'rbf'`, `'poly'`, or `'sigmoid'` depending on the problem.

4. **Train the Model**  
   ~ Fit your model on training data using `.fit(X_train, y_train)`.

5. **Evaluate the Model**  
   ~ Use `.predict(X_test)` and evaluate with accuracy, precision, recall.

6. **Visualize the Results**  
   ~ For 2D data, use `matplotlib` to draw the decision boundary and margins.

---

## ❓ **Why Do We Use Binary SVM**

1. **Handles Two-Class Problems** ~ Designed specifically for binary classification.
2. **Robust to Overfitting** ~ Especially in high-dimensional spaces.
3. **Effective with Kernel Trick** ~ Performs well for non-linear boundaries.
4. **Margin Maximization** ~ Maximizes the margin between the two classes.
5. **Efficient Support Vector Optimization** ~ Uses only support vectors for prediction.

> 💡 SVM is ideal when you have a **clear margin of separation** between two classes.

---

## ⚙️ **How Does Binary SVM Work?**

1. **Construct Optimal Hyperplane**  
   ~ Finds the line/hyperplane that best separates the two classes.

2. **Use Support Vectors**  
   ~ Only the closest data points to the boundary (support vectors) influence the model.

3. **Maximize Margin**  
   ~ Ensures better generalization by maximizing distance between classes.

4. **Handle Non-linear Cases**  
   ~ Uses **kernels** to project data into higher dimensions when data isn't linearly separable.

---
