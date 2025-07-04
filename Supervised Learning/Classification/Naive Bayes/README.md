# 🤖 What is Naive Bayes?

**Naive Bayes** is a type of **Supervised Learning** algorithm based on **Bayes' Theorem**, primarily used for **classification tasks**.  
It assumes that all features are **independent** of one another — which is why it’s called “**naive**.”

📚 Think of it like this:  
If someone likes action movies 🎬 and popcorn 🍿, Naive Bayes assumes these preferences are unrelated — even if they’re not in real life.

---

## 🛠️ How to Use Naive Bayes?

1. **Collect Labeled Data**  
   ~ Each input must be paired with its correct label  
   📄 Example: Emails labeled as **Spam** or **Not Spam**

2. **Preprocess the Data**  
   ~ Clean and convert to numerical format  
   🧹 Example: Use **Bag of Words** or **TF-IDF** to transform text

3. **Split the Dataset**  
   ~ Divide into:  
   📊 **Training Set** (e.g., 80%)  
   🧪 **Test Set** (e.g., 20%)

4. **Train the Naive Bayes Model**  
   ~ Learn probabilities of features for each class  
   📈 Applies **Bayes’ Theorem** to model class likelihoods

5. **Make Predictions**  
   ~ For new input, calculate class with highest probability

6. **Evaluate Performance**  
   ~ Use metrics like:  
   ✅ **Accuracy**  
   ⚖️ **Precision & Recall**  
   🧮 **Confusion Matrix**

---

## ❓ Why Do We Use Naive Bayes?

1. ⚡ **Very Fast & Efficient**  
   ~ Great for large-scale and real-time applications

2. 📬 **Excellent for Text Classification**  
   ~ Ideal for spam detection, sentiment analysis, document tagging

3. 🧠 **Performs Well on Small Datasets**  
   ~ Effective even with limited training data

4. 🛠️ **Simple & Easy to Implement**  
   ~ Requires minimal computation and parameter tuning

---

## ⚙️ How Does Naive Bayes Work?

Naive Bayes uses **Bayes’ Theorem** to compute the probability of a class given input data:

\[
P(Class|Data) = \frac{P(Data|Class) \times P(Class)}{P(Data)}
\]

🧪 It multiplies the individual probabilities of each feature assuming they’re independent, and selects the class with the **highest posterior probability**.

![nv](https://github.com/user-attachments/assets/6990100f-baee-4437-8461-931f19aa1f6d)

---

