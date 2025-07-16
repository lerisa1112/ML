# 🔢 Binary Logistic Regression

**Binary Logistic Regression** is a supervised machine learning algorithm used for **binary classification** tasks, where the target variable has only **two possible outcomes** (e.g., 0 or 1, Yes or No, True or False).

It predicts the **probability** that an input belongs to a specific class using the **sigmoid (logistic)** function.  
Despite the name, it is used for **classification**, not regression.

---

## 🛠️ How to Use Binary Logistic Regression

1. **Collect Data**  
   - Dataset must include input features and a **binary target** (0 or 1).

2. **Preprocess the Data**  
   - Handle missing values  
   - Normalize or scale features  
   - Encode categorical variables

3. **Split the Dataset**  
   - Typically **80% training / 20% testing**

4. **Train the Model (Using scikit-learn)**  
   ```python
   from sklearn.linear_model import LogisticRegression
   model = LogisticRegression()
   model.fit(X_train, y_train)
   ```

5. **Make Predictions**  
   - Predict class: `model.predict(X_test)`  
   - Predict probability: `model.predict_proba(X_test)`

6. **Evaluate the Model**  
   - Accuracy  
   - Confusion matrix  
   - Precision, Recall, F1-score  
   - ROC-AUC score

---

## ❓ Why Do We Use Binary Logistic Regression?

- ✅ **Simple** and **quick** to implement
- 📊 Great for **linearly separable** classes
- 🔢 **Probabilistic output** (not just labels)
- 🔍 **Interpretable** — easy to understand which features affect the result
- 🚀 Acts as a **strong baseline** before trying more complex models

---

## 🧠 How Binary Logistic Regression Works

1. **Linear Combination of Features**
   - The model calculates a weighted sum:
     ```
     z = w1*x1 + w2*x2 + ... + wn*xn + b
     ```

2. **Apply Sigmoid Function**
   - Converts the linear score `z` to a probability:
     ```
     P(y=1) = 1 / (1 + e^(-z))
     ```

3. **Make Classification Decision**
   - If `P(y=1) > 0.5`, predict **class 1**
   - Else, predict **class 0**

---
