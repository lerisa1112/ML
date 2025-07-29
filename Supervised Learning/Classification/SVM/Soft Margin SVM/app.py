import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Load dataset (2D for visualization)
X, y = datasets.make_blobs(n_samples=100, centers=2, random_state=6)
y = np.where(y == 0, -1, 1)  # Convert labels to -1 and 1 (SVM standard)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Soft Margin SVM
clf = SVC(kernel='linear', C=1.0)  # Soft margin controlled by C

# Train model
clf.fit(X_train, y_train)

# Accuracy
print("Train Accuracy:", clf.score(X_train, y_train))
print("Test Accuracy:", clf.score(X_test, y_test))

# Plot decision boundary
def plot_svm(clf, X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', s=30)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # Plot margins and decision boundary
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # Plot support vectors
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
               s=100, linewidth=1, facecolors='none', edgecolors='k')
    plt.title("Soft Margin SVM (Linear Kernel)")
    plt.show()

plot_svm(clf, X, y)
