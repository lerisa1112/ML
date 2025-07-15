# Install ordinal regression model first (if not already)
# pip install mord

import mord as m
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score

# For ordinal data, use a dataset with ordered categories (e.g., student performance)
data = fetch_openml("student-mat", version=1, as_frame=True)
X = data.data.select_dtypes(include=["number"])
y = data.frame["G3"]  # Final grade (0 to 20)

# Convert scores to ordinal labels (Low, Medium, High)
y = y.apply(lambda x: 0 if x < 10 else (1 if x < 15 else 2))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train ordinal logistic regression model
model = m.LogisticIT()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Ordinal Accuracy:", accuracy_score(y_test, y_pred))
