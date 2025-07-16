# 🔢 Multinomial Logistic Regression

**Multinomial Logistic Regression** is a supervised machine learning algorithm used for **multiclass classification** tasks, where the target variable can take **more than two classes** (e.g., class 0, 1, 2, ...).

Unlike binary logistic regression, which predicts between two classes, multinomial logistic regression can handle **three or more mutually exclusive classes**.

It uses the **softmax function** to compute probabilities across multiple classes.

---

## 🛠️ How to Use Multinomial Logistic Regression

1. **Collect Multiclass Data**  
   - Dataset must include features and a target variable with **more than two classes**

2. **Preprocess the Data**  
   - Handle missing values  
   - Encode categorical features  
   - Normalize or standardize features if needed

3. **Split the Dataset**  
   - Common practice: **80% training / 20% testing**

4. **Train the Model (Using scikit-learn)**  
   ```python
   from sklearn.linear_model import LogisticRegression

   model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
   model.fit(X_train, y_train)
   ```

5. **Make Predictions**  
   - Predict class: `model.predict(X_test)`  
   - Predict class probabilities: `model.predict_proba(X_test)`

6. **Evaluate the Model**  
   - Accuracy  
   - Confusion matrix  
   - Precision, Recall, F1-score (macro/micro/weighted)  
   - Log-loss

---

## ❓ Why Do We Use Multinomial Logistic Regression?

- ✅ Works for problems with **3+ mutually exclusive classes**
- 🔢 Outputs a **probability distribution** over all classes
- 🔍 **Interpretable** compared to black-box models like neural networks
- 📦 Easy to implement with scikit-learn
- 🚀 Efficient and fast for linearly separable problems

---

## 🧠 How Multinomial Logistic Regression Works

1. **Linear Score for Each Class**
   - For `k` classes, compute `z_i = w_i·x + b_i` for each class `i`

2. **Apply Softmax Function**
   - Converts raw scores into a probability distribution:
     ```
     P(y=i) = exp(z_i) / sum(exp(z_j)) for all j in classes
     ```

3. **Predict Class**
   - Choose the class with the **highest probability**

---
