# 🧠 What is Ridge Classifier?

**Ridge Classifier** is a type of **linear classifier** that applies **L2 regularization** (also known as Ridge Regression) to prevent overfitting.

📚 Think of it like a strict teacher grading a student's answer. It tries to keep the model simple and penalizes it if it becomes too complex.

It is based on solving a **regression** problem but used for **classification** tasks — it chooses the class with the **highest confidence score**.

---

## 🛠️ How to Use Ridge Classifier?

1. **Import Required Libraries**  
   ~ Use `sklearn.linear_model.RidgeClassifier`

2. **Load and Prepare Data**  
   ~ Use a labeled dataset (features and target values)  
   📦 Example: Iris dataset, binary or multi-class labels

3. **Initialize the Classifier**  
   ~ `model = RidgeClassifier(alpha=1.0)`

4. **Train the Model**  
   ~ `model.fit(X_train, y_train)`

5. **Make Predictions**  
   ~ `y_pred = model.predict(X_test)`

6. **Evaluate Accuracy**  
   ~ Use metrics like `accuracy_score` or `classification_report`

---

## ❓ Why Do We Use Ridge Classifier?

1. ⚖️ **Handles Multicollinearity**  
   ~ Works well when input features are highly correlated

2. 🛡️ **Prevents Overfitting**  
   ~ Uses **L2 regularization** to keep weights small and stable

3. 🚀 **Fast & Efficient**  
   ~ Much faster than non-linear models like SVM or Random Forest

4. 📈 **Works for Binary & Multi-Class**  
   ~ Can be used for both simple and moderate classification tasks

5. 🧪 **Great Baseline Model**  
   ~ Can be used as a starting point to compare with more complex models

---

## ⚙️ How Does Ridge Classifier Work?

### 📐 Mathematically:
It minimizes the following cost function:
```
Loss = ||y - Xw||² + α||w||²
```

- `||y - Xw||²`: squared error between predicted and actual
- `α||w||²`: regularization penalty (α is a tuning parameter)

Then it assigns class labels based on:
```
class = sign(Xw)
```

### 🧮 Internally:
1. Treats classification as a regression problem  
2. Adds **penalty for large coefficients**  
3. Predicts continuous output → selects class with highest value

---
