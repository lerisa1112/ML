from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, accuracy_score

# Load a multiclass dataset
data = load_wine()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Regularized LDA: shrinkage='auto', solver='lsqr'
rlda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

# Fit model
rlda.fit(X_train, y_train)

# Predict
y_pred = rlda.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
