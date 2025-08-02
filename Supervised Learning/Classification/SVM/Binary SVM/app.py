# Binary SVM Example using Scikit-learn

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # SVM Classifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Convert to Binary Classification (Setosa = 0, Not Setosa = 1)
import numpy as np
y_binary = np.where(y == 0, 0, 1)  # 0 = Setosa, 1 = Not Setosa

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.3, random_state=42)

# Initialize SVM model
model = SVC(kernel='linear')  # You can also try 'rbf', 'poly'
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: Plot (2 features only for visualization)
def plot_svm(X, y, model):
    plt.figure(figsize=(8, 6))
    X_plot = X[:, :2]  # First two features
    model.fit(X_plot, y)
    h = .02
    x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
    y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Binary SVM (Setosa vs Not Setosa)')
    plt.show()

plot_svm(X_train, y_train, SVC(kernel='linear'))
