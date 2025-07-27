# ✅ Step 1: Import libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # SVC is C-Support Vector Classification
from sklearn.metrics import accuracy_score, classification_report

# ✅ Step 2: Load dataset (Iris)
iris = datasets.load_iris()
X = iris.data  # features (sepal/petal length & width)
y = iris.target  # labels (0, 1, 2 for different iris species)

# ✅ Step 3: Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Step 4: Create and train SVM classifier
model = SVC(C=1.0, kernel='linear')  # C is the regularization parameter
model.fit(X_train, y_train)

# ✅ Step 5: Predict and evaluate
y_pred = model.predict(X_test)

# ✅ Step 6: Output results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
