# 🌳 What is CHAID?

**CHAID (Chi-squared Automatic Interaction Detection)** is a type of **decision tree algorithm** used for **classification**.  
It builds the tree using **chi-square tests** to determine the best splits, especially for **categorical variables**.

📊 It allows **multi-way splits**, unlike CART or ID3, which are typically binary.

---

## 🛠️ **How to Use CHAID**

1. **Collect Categorical Data**  
   ~ CHAID works best with categorical or discretized numeric variables.

2. **Preprocess Data**  
   ~ Factorize or encode all categorical features numerically.

3. **Choose Features and Target**  
   ~ Select independent variables and a target variable (typically categorical).

4. **Build CHAID Tree**  
   ~ Use a CHAID library like [`python-CHAID`](https://github.com/Rambatino/CHAID) in Python.

5. **Interpret Results**  
   ~ The tree splits show which features and values most influence the target class.

---

## ❓ **Why Do We Use CHAID**

1. **Multi-way Splits**  
   ~ Unlike binary trees, CHAID can split into multiple branches at once.

2. **Great for Categorical Data**  
   ~ Works naturally with nominal and ordinal features.

3. **Interpretable Output**  
   ~ Results are easy to understand and explain to non-technical stakeholders.

4. **Used in Surveys & Market Research**  
   ~ Helps in segmenting the population and identifying influencing factors.

5. **Handles Missing Data**  
   ~ CHAID can gracefully handle missing values in datasets.

---

## ⚙️ **How Does CHAID Work?**

![CHAID](https://upload.wikimedia.org/wikipedia/commons/e/e1/Decision_tree_model.png)

> CHAID builds a decision tree using the following process:

1. **Chi-Square Test on Each Feature**  
   ~ For each predictor, CHAID runs a chi-square test to determine significance with the target variable.

2. **Merge Categories**  
   ~ If categories of a feature are statistically similar (p-value > threshold), they are merged.

3. **Choose Best Split**  
   ~ The feature with the lowest p-value (most significant split) is selected to split the node.

4. **Repeat Recursively**  
   ~ This continues until no further significant splits are found or a maximum depth is reached.

---
