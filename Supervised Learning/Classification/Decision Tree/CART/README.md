## 🤖 What is CART?

**CART** is a machine learning algorithm that builds a **binary decision tree** using:
- **Gini Index** for **classification**
- **Mean Squared Error (MSE)** for **regression**

At each step, it splits the dataset into two groups that are as pure as possible (based on Gini or MSE).

📌 Invented by Breiman et al. in 1986.

---

## 🛠️ How to Use CART

1. **Collect Labeled Data**  
   ~ Example: iris flowers labeled by species

2. **Preprocess Data**  
   ~ Handle missing values, normalize if needed

3. **Train the CART Model**
   ```python
   from sklearn.tree import DecisionTreeClassifier
   model = DecisionTreeClassifier(criterion='gini')  # For classification
   model.fit(X_train, y_train)
   ```

4. **Make Predictions**
   ```python
   predictions = model.predict(X_test)
   ```

5. **Evaluate**
   ```python
   from sklearn.metrics import accuracy_score
   accuracy_score(y_test, predictions)
   ```

6. **Visualize**
   ```python
   from sklearn.tree import plot_tree
   plot_tree(model, filled=True)
   ```

---

## ❓ Why Do We Use CART?

1. **Simple & Interpretable**  
   ~ Easy to visualize and understand

2. **Supports Both Classification and Regression**  
   ~ Versatile for many real-world tasks

3. **Works with Numeric and Categorical Features**  
   ~ Handles various data types

4. **Handles Non-linear Relationships**  
   ~ No need for data transformation

5. **No Assumptions About Data Distribution**  
   ~ Non-parametric algorithm

6. **Used in Ensembles (Random Forest, XGBoost)**  
   ~ Forms the backbone of many high-performance models

---

## ⚙️ How Does CART Work?

1. **Start with entire dataset**
2. At each node:
   - For classification: find the feature that gives **lowest Gini index**
   - For regression: find the feature that gives **lowest MSE**
3. **Split** the dataset into 2 branches (binary tree)
4. Repeat recursively until:
   - Maximum depth is reached
   - No improvement in impurity
