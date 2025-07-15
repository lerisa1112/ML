# 🔢 Logistic Regression in Classification

**Logistic Regression** is a supervised machine learning algorithm used for **classification tasks**.  
It predicts **probabilities** for class membership using the **sigmoid (logistic)** function.

Despite its name, it's used for **classification**, not regression.

---

## 🛠️ How to Use Logistic Regression

1. **Collect labeled data**  
   → Input features with known class labels

2. **Preprocess the data**  
   → Clean missing values, scale numeric data, encode categorical variables

3. **Split the dataset**  
   → Typically 80% for training and 20% for testing

4. **Train the model**  
   → Use `LogisticRegression()` from `scikit-learn` or other libraries

5. **Predict on new data**  
   → `model.predict(X_test)`

6. **Evaluate the model**  
   → Use metrics like accuracy, F1-score, ROC-AUC, etc.

---

## ❓ Why Do We Use Logistic Regression?

- ✅ **Simple and fast** to train and implement
- 📊 **Works well for linearly separable** data
- 🧮 **Outputs probabilities**, not just labels
- 🔍 **Interpretable** — easy to understand feature impact
- 🧠 A good **baseline classifier** before trying complex models

---

## 🧠 How It Works

1. Calculate linear score:  
   `z = w1*x1 + w2*x2 + ... + wn*xn + b`

2. Apply sigmoid activation:  
   `P(y=1) = 1 / (1 + e^(-z))`

3. Classify:  
   - If `P > 0.5`, predict class 1  
   - Otherwise, predict class 0

---
