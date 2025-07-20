# id3_classifier.py

from id3 import Id3Estimator
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = data.data
y = data.target

# ID3 expects data as list of dictionaries
features = [dict(zip(data.feature_names, sample)) for sample in X]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.3, random_state=42)

# ID3 model
model = Id3Estimator()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print("Accuracy (ID3):", accuracy_score(y_test, y_pred))
