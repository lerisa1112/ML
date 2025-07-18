# 🧠 What is Kernel Discriminant Analysis (KDA)?

**Kernel Discriminant Analysis (KDA)** is a non-linear extension of **Linear Discriminant Analysis (LDA)** that uses the **kernel trick** to project input data into a high-dimensional feature space. In this transformed space, it performs linear discriminant analysis to find better separation between classes, even when the original data is not linearly separable.

It’s particularly powerful for complex datasets where class boundaries are non-linear.

---

## 🛠️ How to Use Kernel Discriminant Analysis

1. **Prepare the data**  
   → Handle missing values, normalize features, and label encode if necessary

2. **Select a kernel**  
   → Common choices: RBF (Gaussian), Polynomial, Sigmoid, Linear  
   → Use a kernel function that maps the input to a higher-dimensional space

3. **Fit KDA on training data**  
   → Use `KernelDiscriminantAnalysis` from `sklearn` (via extensions or custom implementation)

4. **Project new data and classify**  
   → Transform the data using the same kernel and classify using the KDA model

5. **Evaluate**  
   → Use accuracy, confusion matrix, ROC-AUC, F1-score, etc.

---

## ❓ Why Do We Use Kernel Discriminant Analysis?

- ✅ **Non-linear classification**: Handles non-linearly separable data
- 📈 **Powerful projection**: Projects data to higher-dimensional space for better separability
- 📊 **Effective for small datasets** where linear methods fail
- 🔄 **Adaptable with different kernels** (e.g., RBF for smooth decision boundaries)

---

## ⚙️ How KDA Works

1. **Apply the kernel function** to project input features into a high-dimensional space:  
   \[
   \phi: \mathbb{R}^n \rightarrow \mathbb{R}^d, \quad \text{where } d \gg n
   \]

2. **Compute scatter matrices** in the feature space:
   - Between-class scatter: \( S_B^\phi \)
   - Within-class scatter: \( S_W^\phi \)

3. **Solve the generalized eigenvalue problem** in kernel space:
   \[
   \mathbf{K}S_B^\phi \mathbf{K}^\top w = \lambda \mathbf{K}S_W^\phi \mathbf{K}^\top w
   \]

4. **Classify new points** based on their projection and distance from class means in the kernel-transformed space.

---
