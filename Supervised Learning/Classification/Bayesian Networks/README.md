# 🤖 What is Bayesian Networks?

**Bayesian Networks (BNs)** are **probabilistic graphical models** that represent a set of variables and their conditional dependencies using a **directed acyclic graph (DAG)**.  
They use **Bayes’ Theorem** to reason about uncertainty.

📚 Think of it like a **map of probabilities**:  
Each node is a variable, and each arrow shows how one variable influences another.

---

## 🛠️ How to Use Bayesian Networks?

1. **Define the Problem**  
   Identify the variables and their relationships.  
   📸 Example: Medical diagnosis — Symptoms, Diseases, and Test Results.

2. **Build the Structure**  
   Create a **directed acyclic graph (DAG)** where:
   - **Nodes** = Random variables
   - **Edges** = Conditional dependencies

3. **Assign Probabilities**  
   For each node:
   - Use a **Conditional Probability Table (CPT)** to describe how it depends on its parents.
   - Probabilities can be learned from data or provided by experts.

4. **Perform Inference**  
   Use algorithms (e.g., Variable Elimination, Belief Propagation) to:
   - Predict missing variables
   - Update beliefs when new evidence is added

5. **Evaluate the Model**  
   Check accuracy, log-likelihood, or prediction quality.

---

## ❓ Why Do We Use Bayesian Networks?

- 📊 **Handle Uncertainty**  
  They naturally work with incomplete or uncertain data.

- 🧠 **Interpretability**  
  Graph structure shows clear cause-effect relationships.

- 🎯 **Probabilistic Reasoning**  
  Allows reasoning about unknown variables using observed data.

- 🔄 **Update with New Evidence**  
  Supports dynamic updating when fresh information arrives.

---

## ⚙️ How Bayesian Networks Work

1. Represent the problem as a **graph** (nodes and edges).  
2. Encode **conditional probabilities** for each variable.  
3. Apply **Bayes’ theorem** to compute updated beliefs.  
4. Use inference algorithms to answer probability queries.

**Formula (Bayes’ Theorem):**
\[
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
\]

---

