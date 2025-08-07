## 📌 What is RBF SVM?

**RBF SVM** is a type of Support Vector Machine that uses the **Radial Basis Function (RBF)** kernel to classify data that is **non-linearly separable** in the original input space. It maps data into a **higher-dimensional space** using the RBF kernel and then finds a separating hyperplane.

> RBF is the most commonly used kernel in SVM due to its ability to handle complex and non-linear relationships.

---

## 🛠️ How to Use RBF SVM

1. **Import Required Libraries**
   ```python
   from sklearn.svm import SVC
   ```

2. **Create and Train the Model**
   ```python
   model = SVC(kernel='rbf', C=1.0, gamma='scale')
   model.fit(X_train, y_train)
   ```

3. **Make Predictions**
   ```python
   predictions = model.predict(X_test)
   ```

4. **Evaluate the Model**
   ```python
   from sklearn.metrics import accuracy_score
   print(accuracy_score(y_test, predictions))
   ```

---

## ❓ Why Do We Use RBF SVM?

- ✅ Handles **non-linear decision boundaries**
- ✅ Works well with **high-dimensional data**
- ✅ **No need to manually map features** — RBF does it for you
- ✅ Effective when **features interact in complex ways**

---

## ⚙️ How Does RBF SVM Work?

1. **Kernel Trick**  
   Transforms data into higher dimensions using the RBF kernel:
   \\( K(x, x') = \exp(-\gamma \|x - x'\|^2) \\)

2. **Hyperplane Construction**  
   After transformation, SVM finds the optimal separating hyperplane.

3. **Hyperparameters**  
   - `C`: Controls trade-off between margin width and misclassification.
   - `gamma`: Controls how far the influence of a single training example reaches.

---
