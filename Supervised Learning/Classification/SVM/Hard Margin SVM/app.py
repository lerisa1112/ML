import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification

# Step 1: Create perfectly linearly separable data
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_clusters_per_class=1,
    class_sep=2.0,      # Controls separability — HIGH value ensures hard margin is possible
    random_state=42
)

# Step 2: Train Hard Margin SVM (Set C to a very large value)
model = svm.SVC(kernel='linear', C=1e10)
model.fit(X, y)

# Step 3: Plotting the decision boundary
w = model.coef_[0]
b = model.intercept_[0]

# Create grid for plotting
x_plot = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1, 100)
y_plot = -(w[0] * x_plot + b) / w[1]  # Decision boundary
margin = 1 / np.sqrt(np.sum(w ** 2))
y_margin_up = y_plot + margin
y_margin_down = y_plot - margin

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
plt.plot(x_plot, y_plot, 'k-')  # Decision boundary
plt.plot(x_plot, y_margin_up, 'k--')  # Margin
plt.plot(x_plot, y_margin_down, 'k--')
plt.title('Hard Margin SVM (Linear)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)
plt.show()
