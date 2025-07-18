
## 🧠 What is Bayesian Quadratic Discriminant Analysis?

**Bayesian QDA** is a probabilistic extension of **Quadratic Discriminant Analysis (QDA)** that uses **Bayesian inference** to estimate the parameters of the model (class means and covariance matrices).

Unlike traditional QDA, which estimates parameters using **Maximum Likelihood Estimation (MLE)**, Bayesian QDA treats these parameters as **random variables with prior distributions**, allowing for more **robust inference** especially when data is limited or noisy.

---

## 🛠️ How to Use Bayesian QDA

1. **Prepare the dataset**  
   - Ensure you have a clean, labeled dataset suitable for classification.

2. **Choose priors**  
   - For means: use Gaussian priors  
   - For covariances: use Inverse-Wishart or LKJ priors

3. **Set up a probabilistic model**  
   - Use a Bayesian framework like **PyMC**, **TensorFlow Probability**, or **Stan**

4. **Run inference (sampling)**  
   - Use MCMC or variational inference to sample from the posterior distribution of parameters

5. **Make predictions**  
   - Use posterior predictive distribution to classify new instances

---

## ❓ Why Do We Use Bayesian QDA?

- ✅ **Uncertainty-aware**: Captures uncertainty in model parameters
- 🔍 **Robust to small datasets**: Priors help regularize in low-data settings
- 📊 **Probabilistic classification**: Outputs full posterior distributions, not just point estimates
- 🧠 **Principled approach**: Fits naturally in Bayesian decision theory

---

## ⚙️ How Bayesian QDA Works

1. **Assumes multivariate Gaussian distribution** for each class:
   \[
   P(x|y=k) \sim \mathcal{N}(\mu_k, \Sigma_k)
   \]

2. **Places prior distributions** on the parameters:
   - \(\mu_k \sim \mathcal{N}(\mu_0, \Lambda^{-1})\)
   - \(\Sigma_k \sim \text{Inverse-Wishart}(\Psi, \nu)\) or via LKJ priors

3. **Uses Bayes’ Rule** to update beliefs after seeing data:
   \[
   P(\mu_k, \Sigma_k | D_k) \propto P(D_k | \mu_k, \Sigma_k) \cdot P(\mu_k) \cdot P(\Sigma_k)
   \]

4. **Classifies a new instance** by computing:
   \[
   \text{Class}(x) = \arg\max_k \int P(x | \mu_k, \Sigma_k) \cdot P(\mu_k, \Sigma_k | D_k) \, d\mu_k \, d\Sigma_k
   \]
   (Usually approximated by sampling)

---
