from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# 1. Load dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Take only 2 features for visualization
y = (iris.target != 0) * 1  # Binary classification: class 0 vs others

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 3. Initialize Linear SVM
model = SVC(kernel='linear', C=1.0)  # 'C' is the soft margin parameter

# 4. Train the model
model.fit(X_train, y_train)

# 5. Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 6. Visualize decision boundary (2D)
def plot_svm_boundary(model, X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=30)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 30),
                         np.linspace(ylim[0], ylim[1], 30))
    xy = np.c_[xx.ravel(), yy.ravel()]
    Z = model.decision_function(xy).reshape(xx.shape)

    # Plot decision boundary and margins
    plt.contour(xx, yy, Z, levels=[-1, 0, 1], linestyles=['--', '-', '--'])
    plt.title("Linear SVM Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

plot_svm_boundary(model, X, y)
