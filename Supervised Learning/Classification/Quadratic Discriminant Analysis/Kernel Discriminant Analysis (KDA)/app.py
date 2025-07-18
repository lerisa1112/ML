import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply Kernel PCA (RBF kernel)
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.2)  # gamma can be tuned
X_kpca_train = kpca.fit_transform(X_train)
X_kpca_test = kpca.transform(X_test)

# Apply LDA after kernel transformation
lda = LDA()
lda.fit(X_kpca_train, y_train)
y_pred = lda.predict(X_kpca_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"KDA Accuracy: {acc:.2f}")

# Visualization
def plot_kda(X, y, model, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=50)
    plt.title(title)
    plt.xlabel("Kernel PC1")
    plt.ylabel("Kernel PC2")
    plt.grid(True)
    plt.show()

# Plot the transformed space
plot_kda(X_kpca_train, y_train, lda, "Kernel Discriminant Analysis (KDA) Decision Boundary")
