
# 🔷 Non-Linear SVM

**Non-Linear SVM** is a type of Support Vector Machine that handles **non-linearly separable data** by transforming it into a higher-dimensional space using **kernel functions**, where a linear hyperplane can separate the data.

📌 It’s especially useful when data cannot be separated by a straight line in its original form.

---

## 🛠️ How to Use Non-Linear SVM

1. **Import the Classifier**
   ```python
   from sklearn.svm import SVC
   ```

2. **Initialize with a Non-Linear Kernel**
   ```python
   model = SVC(kernel='rbf', C=1.0, gamma='auto')  # RBF is most common
   ```

3. **Train the Model**
   ```python
   model.fit(X_train, y_train)
   ```

4. **Make Predictions**
   ```python
   y_pred = model.predict(X_test)
   ```

5. **Evaluate the Model**
   ```python
   model.score(X_test, y_test)
   ```

---

## ❓ Why Do We Use Non-Linear SVM?

1. 🌀 **Handles Complex Data**
   ~ Works with data that cannot be separated linearly.

2. 🧠 **Kernel Trick Magic**
   ~ Transforms data into higher dimensions to find a hyperplane that separates the classes.

3. 💪 **Highly Flexible**
   ~ Adapts to the shape of the data boundary with various kernels like:
   - RBF (Radial Basis Function)
   - Polynomial
   - Sigmoid

---

## ⚙️ How Does Non-Linear SVM Work?

### 📐 Mathematical Intuition:

Instead of explicitly mapping data to higher dimensions, SVM uses a **kernel function** `K(xᵢ, xⱼ)` to compute the dot product in the new space:

```
K(xᵢ, xⱼ) = φ(xᵢ) · φ(xⱼ)
```

Popular kernel examples:

- **RBF Kernel**:  
  ```
  K(x, x') = exp(-γ ||x - x'||²)
  ```

- **Polynomial Kernel**:  
  ```
  K(x, x') = (x · x' + c)^d
  ```

---

### 🧠 Mechanism:

1. Transforms input space using a **non-linear kernel**
2. Finds an **optimal hyperplane** in this new space
3. Uses **support vectors** to define the decision boundary
4. Creates **non-linear decision boundaries** in the original input space

---
