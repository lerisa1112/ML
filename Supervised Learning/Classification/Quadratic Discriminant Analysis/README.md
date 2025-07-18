

## 🧠 What is Quadratic Discriminant Analysis (QDA)?

**Quadratic Discriminant Analysis** is a supervised machine learning algorithm used for **classification tasks**. It models the probability distribution of the input features for each class using a **multivariate Gaussian distribution** — allowing **separate covariance matrices** for each class.

It is part of the **Discriminant Analysis** family, alongside **Linear Discriminant Analysis (LDA)**.

---

## 🛠️ How to Use Quadratic Discriminant Analysis

1. **Prepare the data**
   - Ensure the data is labeled and suitable for classification
   - Handle missing values and scale features if needed

2. **Split the dataset**
   - Typically 80% for training and 20% for testing

3. **Train the model**
   - Use `QuadraticDiscriminantAnalysis()` from `sklearn`

4. **Predict on new data**
   - Use `model.predict(X_test)`

5. **Evaluate the model**
   - Use accuracy, confusion matrix, and F1-score

---

## ❓ Why Do We Use QDA?

- ✅ **Flexible**: Allows non-linear (quadratic) decision boundaries
- 📊 **Class-specific covariance**: More accurate for complex distributions
- 🧮 **Probabilistic model**: Outputs class probabilities
- 🚫 Better than LDA when the **covariances differ** significantly across classes

---

## ⚙️ How QDA Works

1. **Assumes Gaussian distribution** for each class:
   \[
   P(x|y=k) \sim \mathcal{N}(\mu_k, \Sigma_k)
   \]

2. **Bayes' Rule** is applied:
   \[
   P(y=k|x) \propto P(x|y=k)P(y=k)
   \]

3. **Classifies input x** to the class with the highest posterior:
   \[
   \text{Class}(x) = \arg\max_k P(y=k|x)
   \]

4. Because each class has its **own covariance matrix**, decision boundaries become **quadratic**.

---

