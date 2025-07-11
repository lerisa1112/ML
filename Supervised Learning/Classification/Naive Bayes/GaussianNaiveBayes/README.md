
# 🤖 What is Gaussian Naive Bayes?

**Gaussian Naive Bayes (GNB)** is a classification algorithm based on **Bayes' Theorem** with the assumption that the features follow a **normal (Gaussian) distribution**.

It is a variant of Naive Bayes suited for **continuous features**. It assumes independence between features and models each one using a **bell-curve (normal distribution)**.

> ✅ Commonly used in spam filtering, medical diagnosis, and real-time prediction systems.

---

## 🛠️ How to Use Gaussian Naive Bayes

1. **Import Required Libraries**  
   ```python
   from sklearn.naive_bayes import GaussianNB
   from sklearn.model_selection import train_test_split
   from sklearn.datasets import load_iris
   from sklearn.metrics import accuracy_score
   ```

2. **Load and Prepare Data**  
   ```python
   iris = load_iris()
   X, y = iris.data, iris.target
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
   ```

3. **Train the Model**  
   ```python
   model = GaussianNB()
   model.fit(X_train, y_train)
   ```

4. **Make Predictions**  
   ```python
   y_pred = model.predict(X_test)
   ```

5. **Evaluate the Model**  
   ```python
   print("Accuracy:", accuracy_score(y_test, y_pred))
   ```

---

## ❓ Why Do We Use Gaussian Naive Bayes?

1. ✅ **Fast and Efficient** — Training and prediction are quick.  
2. ✅ **Best for Real-Valued Inputs** — Assumes features follow a normal distribution.  
3. ✅ **Effective with Small Data** — Performs well even with limited training data.  
4. ✅ **Simple and Interpretable** — Easy to understand and implement.  
5. ✅ **Low Memory Footprint** — Stores only mean and variance for each feature class.  
6. ✅ **Good Starting Point** — Often used as a baseline for classification tasks.  
7. ✅ **Used in Real-Life Applications** — Email spam filters, disease prediction, etc.

---

## ⚙️ How Does Gaussian Naive Bayes Work?
1. **Applies Bayes’ Theorem:**

   \
   P(y|X) = \frac{P(X|y) \cdot P(y)}{P(X)}
   \

2. **Assumes each feature follows a Gaussian distribution:**

   \
   P(x_i|y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \cdot \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma_y^2}\right)
   \

3. **For each class, compute:**
   - Mean \(\mu_y\) and variance \(\sigma_y^2\) of each feature
   - Likelihood of features given the class
   - Posterior probability for each class

4. **Predict the class with the highest posterior probability**

---

