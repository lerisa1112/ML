# 🔢 Multilabel Logistic Regression

**Multilabel Logistic Regression** is a supervised machine learning algorithm used for **multilabel classification** tasks, where each instance can belong to **multiple classes simultaneously** (e.g., tags, genres, symptoms).

It extends binary logistic regression by **training one binary classifier per label** (also called One-vs-Rest or OvR strategy).

---

## 🛠️ How to Use Multilabel Logistic Regression

1. **Collect Multilabel Data**  
   - Dataset should include input features and a **set of labels** per instance (e.g., [1, 0, 1, 0])

2. **Preprocess the Data**  
   - Encode labels using `MultiLabelBinarizer` or similar  
   - Normalize/scale features  
   - Handle missing values

3. **Split the Dataset**  
   - Typical ratio: 80% training / 20% testing

4. **Train the Model (Using scikit-learn)**  
   ```python
   from sklearn.linear_model import LogisticRegression
   from sklearn.multioutput import MultiOutputClassifier

   model = MultiOutputClassifier(LogisticRegression())
   model.fit(X_train, Y_train)
   ```

5. **Make Predictions**  
   - Predict multilabel output: `model.predict(X_test)`

6. **Evaluate the Model**  
   - Hamming loss  
   - Precision/Recall per label  
   - F1-score (macro/micro/weighted)

---

## ❓ Why Do We Use Multilabel Logistic Regression?

- ✅ **Simple extension** of binary logistic regression
- 🔁 **Handles multiple labels** per instance
- 🔍 **Interpretable** and explainable predictions per label
- 🛠️ **Easy to implement** using One-vs-Rest (OvR)
- 🚀 Strong baseline for multilabel problems

---

## 🧠 How Multilabel Logistic Regression Works

1. **One Classifier per Label**
   - For `k` labels, train `k` binary logistic regression models

2. **Binary Classification for Each Label**
   - Each classifier outputs probability: `P(y_i=1 | x)` for label `i`

3. **Thresholding (Optional)**
   - Use `> 0.5` threshold or tune it per label to control precision/recall

---
