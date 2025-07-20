## 🤖 What is ID3 (Iterative Dichotomiser 3)?

**ID3** is a classic **Decision Tree algorithm** developed by **Ross Quinlan** in 1986.  
It is used for **classification tasks** in **supervised learning**.

ID3 builds a decision tree by **recursively selecting the best attribute** that provides the **highest Information Gain** to split the dataset.  
It uses **Entropy** as a measure of impurity and chooses splits that **reduce uncertainty** in class labels.

🧠 ID3 is the foundation of many advanced tree-based algorithms like C4.5, C5.0, and CART.

---

## 🛠️ How to Use ID3

1. **Prepare Labeled Data**  
   - Input data must be **categorical** or **discretized**.
   - Each instance should include features and a target class label.

2. **Choose the Root Feature**  
   - Compute **Information Gain** for all features.
   - Select the feature with the **highest gain**.

3. **Recursively Split**  
   - Apply the same process to each subset of the data.
   - Build internal nodes and leaf nodes.

4. **Stop When**  
   - All samples in a node belong to the same class.
   - No features remain.
   - A maximum tree depth is reached (optional pruning).

---

## ❓ Why Do We Use ID3?

- 📊 **Effective for Categorical Data**  
  Great for datasets with discrete values (like survey data, decision logs).

- 🔍 **Uses Information Gain**  
  Prioritizes the most informative features first.

- 🧱 **Foundational Algorithm**  
  Basis for more advanced decision tree algorithms.

- 🧠 **Interpretable & Explainable**  
  The resulting tree is human-readable and easy to understand.

- ⚡ **Fast to Build**  
  Relatively simple to implement and computationally efficient.

---

## ⚙️ How Does ID3 Work?

1. **Entropy Calculation**  
   For each feature, calculate the entropy of the dataset if it were split on that feature.

2. **Information Gain**  
   Measure the reduction in entropy after splitting:
   \[
   \text{Information Gain} = \text{Entropy(Parent)} - \sum \left( \frac{\text{Child Size}}{\text{Parent Size}} \times \text{Entropy(Child)} \right)
   \]

3. **Feature Selection**  
   Choose the feature with the **highest Information Gain** to split the data.

4. **Repeat Recursively**  
   Build branches for each value of the selected feature and repeat on each subset.

5. **Stopping Condition**  
   - All instances have the same label
   - No more features to split on
   - Optional: Use pruning to reduce overfitting

🧮 *ID3 is a greedy algorithm: it makes locally optimal choices at each node to build the best tree.*

---
