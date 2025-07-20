# decision_tree_entropy.py

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree classifier (ID3-style with entropy)
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_train, y_train)

# Accuracy
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)

# Visualize tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.show()
