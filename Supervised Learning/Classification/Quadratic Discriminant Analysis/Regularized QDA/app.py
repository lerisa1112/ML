import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.stats import multivariate_normal

class RegularizedQDA:
    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = {}
        self.covariances = {}
        self.priors = {}
        self.n_features = X.shape[1]

        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            cov = np.cov(X_c, rowvar=False)
            identity = np.eye(self.n_features)
            self.covariances[c] = (1 - self.reg_lambda) * cov + self.reg_lambda * identity
            self.priors[c] = X_c.shape[0] / X.shape[0]
    
    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = []
            for c in self.classes:
                mean = self.means[c]
                cov = self.covariances[c]
                prior = self.priors[c]
                # Use multivariate normal PDF
                likelihood = multivariate_normal.pdf(x, mean=mean, cov=cov)
                posteriors.append(prior * likelihood)
            predictions.append(np.argmax(posteriors))
        return np.array(predictions)

# Load and prepare Iris data
iris = load_iris()
X = iris.data[:, :2]  # only 2 features for visualization
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Regularized QDA
model = RegularizedQDA(reg_lambda=0.2)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Regularized QDA Accuracy: {acc:.2f}")
