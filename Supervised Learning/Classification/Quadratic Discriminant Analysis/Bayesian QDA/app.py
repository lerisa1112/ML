import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import multivariate_normal

# Load dataset (2-class for simplicity)
iris = load_iris()
X = iris.data[iris.target != 2, :2]  # Only classes 0 and 1, and first 2 features
y = iris.target[iris.target != 2]

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Separate classes
X_class0 = X_train[y_train == 0]
X_class1 = X_train[y_train == 1]

n_features = X.shape[1]

with pm.Model() as model:
    # Priors for class 0
    mu0 = pm.Normal("mu0", mu=0, sigma=1, shape=n_features)
    packed_L0 = pm.LKJCholeskyCov('packed_L0', n=n_features, eta=2.0, sd_dist=pm.HalfCauchy.dist(1))
    L0 = pm.expand_packed_triangular(n_features, packed_L0)
    cov0 = pm.Deterministic('cov0', L0 @ L0.T)

    # Priors for class 1
    mu1 = pm.Normal("mu1", mu=0, sigma=1, shape=n_features)
    packed_L1 = pm.LKJCholeskyCov('packed_L1', n=n_features, eta=2.0, sd_dist=pm.HalfCauchy.dist(1))
    L1 = pm.expand_packed_triangular(n_features, packed_L1)
    cov1 = pm.Deterministic('cov1', L1 @ L1.T)

    # Likelihood for each class
    obs0 = pm.MvNormal("obs0", mu=mu0, chol=L0, observed=X_class0)
    obs1 = pm.MvNormal("obs1", mu=mu1, chol=L1, observed=X_class1)

    # Sample from posterior
    trace = pm.sample(1000, tune=1000, chains=2, target_accept=0.9)

# Extract posterior means
mu0_post = trace.posterior['mu0'].mean(dim=("chain", "draw")).values
mu1_post = trace.posterior['mu1'].mean(dim=("chain", "draw")).values
cov0_post = trace.posterior['cov0'].mean(dim=("chain", "draw")).values
cov1_post = trace.posterior['cov1'].mean(dim=("chain", "draw")).values

# Predict
y_pred = []
for x in X_test:
    p0 = multivariate_normal.pdf(x, mean=mu0_post, cov=cov0_post)
    p1 = multivariate_normal.pdf(x, mean=mu1_post, cov=cov1_post)
    y_pred.append(0 if p0 > p1 else 1)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Bayesian QDA Accuracy: {acc:.2f}")
