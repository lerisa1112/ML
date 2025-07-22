## 🌱 What is MARS?

**MARS** (Multivariate Adaptive Regression Splines) is a **non-parametric regression model** that automatically fits **non-linear relationships** between inputs and outputs using **piecewise linear regressions**.

📚 Think of it like building multiple linear models for different regions of the input data, and then stitching them together.

---

## 🛠️ How to Use MARS?

1. **Install the Required Library**

```bash
pip install sklearn-contrib-py-earth
```

2. **Basic Usage in Python**

```python
from pyearth import Earth
model = Earth()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

3. **Evaluate the Model**

Use evaluation metrics like:
- ✅ R² Score
- 📉 Mean Squared Error (MSE)
- 📊 Visualize predicted vs actual values

---

## ❓ Why Do We Use MARS?

- ✅ **Captures Nonlinear Relationships**  
  Automatically models curves and interactions between features.

- 🧠 **Built-in Feature Selection**  
  Only includes relevant features and interactions.

- 🔎 **Interpretable**  
  Output is readable and understandable, unlike black-box models.

- ⚡ **Efficient for Structured Data**  
  Works well with tabular datasets and moderate feature sizes.

---

## ⚙️ How MARS Works

1. **Forward Step**
   - Adds **basis functions** (hinge functions like `max(0, x - c)`) to improve prediction.
   - Explores feature interactions.

2. **Backward Pruning**
   - Removes unhelpful basis functions.
   - Uses **Generalized Cross Validation (GCV)** to avoid overfitting.

3. **Final Model**
   - A sum of selected hinge functions that best fit the training data.

---
