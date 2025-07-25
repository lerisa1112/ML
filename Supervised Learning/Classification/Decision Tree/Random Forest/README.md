# 🌲 What is Random Forest?

**Random Forest** is a powerful and popular **ensemble machine learning algorithm** that builds multiple **decision trees** and merges their outputs to improve accuracy and prevent overfitting.  
It can be used for **classification**, **regression**, and even **feature selection**.

> 🎯 Think of it like asking a group of experts (trees) and taking a majority vote (classification) or average (regression).

---

## 🛠️ How to Use Random Forest

1. **Import the Random Forest Library**  
   ```python
   from sklearn.ensemble import RandomForestClassifier
   ```

2. **Prepare the Data**
  ```python
    X = [[5, 3], [10, 15], [15, 12], [24, 10], [30, 45]]
    y = [1, 0, 0, 1, 0]
   ```

3. **Initialize and Train the Model**
  ```python
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
   ```

4. **Make Predictions**
  ```python
   predictions = model.predict([[10, 10]])
   print(predictions)
   ```

5. **Evaluate the Model**
  ```python
   from sklearn.metrics import accuracy_score
   accuracy = accuracy_score(y_true, y_pred)
   print("Accuracy:", accuracy)
   ```
---

## ❓ Why Do We Use Random Forest

- ✅ **High Accuracy** – Combines multiple models for better performance.  
- 🌲 **Reduces Overfitting** – Uses average of trees, reducing bias.  
- 🧠 **Handles Large Datasets** – Efficient with high-dimensional data.  
- 🧮 **Feature Importance** – Helps in selecting key features.  
- 🔀 **Works with Missing Values** – Tolerates incomplete datasets.  
- 🔍 **Robust to Noise** – Stable even with noisy inputs.  
- 🔄 **Flexible** – Can be used for classification, regression, and feature ranking.

---

## ⚙️ How Does Random Forest Work?

1. **Bootstrap Sampling**  
   Random subsets of the original dataset are drawn **with replacement**.

2. **Train Decision Trees**  
   Each subset is used to train an individual **decision tree**.  
   At each split, a **random subset of features** is chosen.

3. **Aggregate Predictions**  
   - **For Classification**: Majority vote from all trees.  
   - **For Regression**: Average of all tree outputs.

4. **Final Output**  
   The aggregated result is used as the model’s final prediction.
---

