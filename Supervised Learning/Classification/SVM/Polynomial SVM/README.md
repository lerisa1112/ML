# 🤖 What is Polynomial SVM?

**Polynomial SVM** is a type of **Support Vector Machine (SVM)** that uses a **polynomial kernel function** to separate non-linearly separable data by mapping it into a higher-dimensional space. It's especially useful when the relationship between the input features and class labels is **non-linear** but still follows a **polynomial pattern**.

---

## 🛠️ How to Use Polynomial SVM

1. **Import Libraries**
   ```python
   from sklearn.svm import SVC
   ```

2. **Prepare Your Dataset**
   - Use any dataset (synthetic or real-world)
   - Normalize if needed

3. **Create the Model**
   ```python
   model = SVC(kernel='poly', degree=3, C=1.0, coef0=1)
   ```

4. **Train the Model**
   ```python
   model.fit(X_train, y_train)
   ```

5. **Make Predictions**
   ```python
   y_pred = model.predict(X_test)
   ```

6. **Evaluate Performance**
   ```python
   from sklearn.metrics import accuracy_score
   print("Accuracy:", accuracy_score(y_test, y_pred))
   ```

---

## ❓ Why Do We Use Polynomial SVM

1. ✅ **Handles Non-Linear Data:** Perfect when data isn't linearly separable.
2. ✅ **Captures Polynomial Relationships:** Useful when data follows polynomial trends.
3. ✅ **More Flexible than Linear SVM:** Maps inputs to higher dimensions.
4. ✅ **Better Control over Fit:** Degree of polynomial controls the complexity.

---

## ⚙️ How Does Polynomial SVM Work?

Polynomial SVM uses the following **kernel function**:

\$
K(x, x') = (x^T x' + c)^d
\$

Where:
- **x, x′** = input vectors
- **c** = constant term (coef0 in scikit-learn)
- **d** = degree of the polynomial

🔍 This function allows SVM to **project data into a higher-dimensional space** where it becomes **linearly separable**.

---
