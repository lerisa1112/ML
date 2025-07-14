# 🤖 What is Bernoulli Naive Bayes?

**Bernoulli Naive Bayes** is a variant of the Naive Bayes algorithm that is particularly suited for **binary/boolean features**. It is commonly used in **text classification tasks** such as spam detection, sentiment analysis, and document categorization.

---
## 🛠️ How to Use Bernoulli Naive Bayes

1. **Prepare Your Dataset**  
   Collect labeled binary data (e.g., emails marked as spam or not spam)

2. **Preprocess the Data**  
   - Use `CountVectorizer(binary=True)` to convert text into binary feature vectors  
   - Each word in the vocabulary is marked as **1 (present)** or **0 (absent)**

3. **Split the Data**  
   Use `train_test_split()` to divide data into:
   - 🧠 Training Set
   - 🔎 Testing Set

4. **Train the Model**  
   - Use `BernoulliNB()` to train the classifier on the binary features

5. **Make Predictions**  
   - Predict spam/not-spam on unseen data using `.predict()`

6. **Evaluate the Model**  
   - Use `accuracy_score`, `confusion_matrix`, and `classification_report` to evaluate performance

---

## ❓ Why Do We Use Bernoulli Naive Bayes?

1. ⚡ **Efficient & Fast**  
   - Ideal for real-time predictions on large datasets

2. 🧪 **Great for Binary Data**  
   - Works well when features are yes/no, 0/1 (e.g., word present or not)

3. 📬 **Perfect for Text Classification**  
   - Especially effective for spam filters, sentiment analysis, etc.

4. 🧠 **Performs Well with Small Datasets**  
   - Even limited labeled data can yield good results

5. 🛠️ **Simple to Implement**  
   - Requires minimal parameter tuning and computation

---

## ⚙️ How Does Bernoulli Naive Bayes Work?

Bernoulli Naive Bayes uses **Bayes’ Theorem** to predict the probability that an input belongs to a particular class, based on binary features.

### 📊 Bayes' Theorem Formula:
\[
P(Class|Data) = \frac{P(Data|Class) \times P(Class)}{P(Data)}
\]

### 🔍 Steps:
1. It assumes **feature independence**
2. For each feature (e.g., word presence), it calculates:
   - The probability of the word being **present (1)** or **absent (0)** in each class
3. It multiplies the probabilities for all features
4. It selects the class with the **highest posterior probability**

---
