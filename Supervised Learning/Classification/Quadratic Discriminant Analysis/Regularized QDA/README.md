# 🧠 What is Regularized Quadratic Discriminant Analysis (R-QDA)?

**Regularized QDA** is a variation of **Quadratic Discriminant Analysis (QDA)** that adds **regularization** to the covariance matrices of each class. This improves stability and performance when:

- The number of features is large
- The dataset is small
- Covariance matrices are ill-conditioned (nearly singular)

It blends between QDA and simpler models like **LDA or Naive Bayes** by **shrinking** the covariance matrices towards a common structure.

---

## 🛠️ How to Use Regularized QDA

1. **Prepare the dataset**  
   → Clean missing data, normalize features, encode labels

2. **Train the model with regularization**  
   → Use `QuadraticDiscriminantAnalysis(reg_param=...)` from `sklearn`  
   → `reg_param` controls the amount of shrinkage (0 = QDA, 1 = Naive Bayes-like)

3. **Make predictions**  
   → Use `.predict(X_test)` and `.predict_proba(X_test)`

4. **Evaluate the model**  
   → Use confusion matrix, classification report, accuracy/F1-score

---

## ❓ Why Do We Use Regularized QDA?

- 🧷 **Stability**: Works well even when features > samples
- 🧮 **Handles multicollinearity**: Better than plain QDA with highly correlated features
- ⚖️ **Balance**: Combines QDA’s flexibility with LDA’s robustness
- 🔢 **Controls overfitting**: Especially useful in small datasets

---

## ⚙️ How Regularized QDA Works

1. **Estimate class-specific covariance matrices**:  
   \[
   \Sigma_k = (1 - \alpha)\Sigma_k + \alpha I
   \]  
   where \( \alpha = \text{reg\_param} \in [0, 1] \)

2. **Apply Bayes’ Rule** with these regularized covariances:  
   \[
   P(y=k|x) \propto P(x|y=k)P(y=k)
   \]

3. **Classify input x** based on maximum posterior probability

4. **Decision boundaries** remain **quadratic**, but are more stable under noise and data imbalance

---
