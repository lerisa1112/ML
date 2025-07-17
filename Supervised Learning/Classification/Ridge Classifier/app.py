from sklearn.linear_model import RidgeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
X, y = load_iris(return_X_y=True)

# Use only two classes for better separability
# Let's take class 0 and class 1 only
X = X[y != 2]
y = y[y != 2]

# Train on the full data
model = RidgeClassifier(alpha=1.0)
model.fit(X, y)

# Predict on the same training data
y_pred = model.predict(X)

# Evaluate
print("Accuracy:", accuracy_score(y, y_pred))
print("\nClassification Report:\n", classification_report(y, y_pred))
