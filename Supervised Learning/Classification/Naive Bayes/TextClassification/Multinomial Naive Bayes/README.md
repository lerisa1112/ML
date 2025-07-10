# 🤖 What is Multinomial Naive Bayes (MNB)?

**Multinomial Naive Bayes (MNB)** is a **supervised machine learning algorithm** based on **Bayes' Theorem**, tailored specifically for **discrete data**, especially word counts or frequencies in text documents.

It assumes that the features (like words in a document) are conditionally independent given the class, and follows a **multinomial distribution**. It is one of the most popular algorithms for **text classification**, like:

- Email Spam Detection
- Sentiment Analysis
- Document Categorization
- News Article Classification

---

## 🛠️ How to Use Multinomial Naive Bayes

1. **Collect Text Data**  
   ~ Gather labeled text (e.g., sports, tech, spam, etc.)

2. **Preprocess Text**  
   ~ Clean text (lowercase, remove stopwords, punctuation)

3. **Convert Text to Numeric Features**  
   ~ Use `CountVectorizer` or `TfidfVectorizer` to turn text into feature vectors

4. **Split the Data**  
   ~ Use `train_test_split()` to divide into training and testing sets

5. **Train the MNB Model**  
   ~ Use `MultinomialNB().fit(X_train, y_train)`

6. **Predict on New Data**  
   ~ Use `model.predict(X_test)` or any new document

7. **Evaluate Performance**  
   ~ Use accuracy, confusion matrix, F1-score to evaluate

---

## ❓ Why Do We Use Multinomial Naive Bayes

1. **Effective for Text Classification**  
   ~ Especially good with word count features

2. **Fast & Scalable**  
   ~ Very quick training even on large datasets

3. **Low Resource Requirements**  
   ~ Minimal memory and computation needs

4. **Works Well with Sparse Data**  
   ~ Can handle thousands of features (words)

5. **Performs Well with Small Data**  
   ~ No need for millions of examples

6. **Returns Probabilities**  
   ~ Useful when you need confidence levels

7. **Simple to Implement**  
   ~ Easy to use with `scikit-learn`

---

## ⚙️ How Does Multinomial Naive Bayes Work?


<img width="992" height="333" alt="mn" src="https://github.com/user-attachments/assets/c5e9d569-55a2-40d4-a87a-454a87a8ec2a" />


