# 🤖 What is Kernel Support Vector Machine (SVM)?

**Kernel SVM** is an advanced form of **Support Vector Machine (SVM)** that allows the algorithm to perform **non-linear classification** using the **kernel trick**. It implicitly maps the input features into a **higher-dimensional space** where a linear separator (hyperplane) can be found.

---

## 🛠️ How to Use Kernel SVM

1. **Collect Data**  
   ~ Gather labeled training data (e.g., handwriting samples, face images).

2. **Preprocess Data**  
   ~ Scale/normalize the features; handle missing or categorical data.

3. **Choose Kernel Function**  
   ~ Common kernels include:  
     - Linear  
     - Polynomial  
     - RBF (Radial Basis Function)  
     - Sigmoid

4. **Train the Model**  
   ~ Fit the SVM model using your chosen kernel on training data.

5. **Tune Hyperparameters**  
   ~ Adjust `C`, `gamma`, `degree`, and `kernel` to optimize performance.

6. **Evaluate the Model**  
   ~ Use confusion matrix, accuracy, precision, recall, F1-score, etc.

7. **Make Predictions**  
   ~ Classify unseen data using the trained kernel SVM model.

---

## ❓ Why Do We Use Kernel SVM

1. ✅ **Handles Non-linear Data**  
   ~ Transforms input into high dimensions to separate complex patterns.

2. ✅ **Effective in High-dimensional Spaces**  
   ~ Performs well when features outnumber samples.

3. ✅ **Memory Efficient**  
   ~ Only support vectors are stored and used in decision function.

4. ✅ **Customizable**  
   ~ Various kernels let you tailor the model to your data structure.

5. ✅ **Applicable in Real-world Use Cases**  
   ~ Text classification, image recognition, gene classification, etc.

---

## ⚙️ How Does Kernel SVM Work?

1. **Kernel Trick**  
   ~ Projects original features into a higher-dimensional space without explicitly computing the coordinates.

2. **Decision Boundary in Higher Space**  
   ~ Finds a linear hyperplane in transformed space, which translates to a non-linear boundary in original space.

3. **Common Kernel Functions**:  
   - **Linear Kernel:** `K(x, x') = x · x'`  
   - **Polynomial Kernel:** `K(x, x') = (x · x' + c)^d`  
   - **RBF Kernel:** `K(x, x') = exp(-γ ||x - x'||²)`  
   - **Sigmoid Kernel:** `K(x, x') = tanh(κ x · x' + c)`

4. **Soft Margin Concept**  
   ~ Allows certain misclassified points to avoid overfitting (with slack variables).

---
