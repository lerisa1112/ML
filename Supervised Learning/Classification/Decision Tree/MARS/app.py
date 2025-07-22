from pyearth import Earth
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 🎯 Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# 📚 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 🌍 Fit MARS model
model = Earth()
model.fit(X_train, y_train)

# 📈 Predict
y_pred = model.predict(X_test)

# 🧮 Evaluation
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# 📊 Plot results
plt.scatter(X_test, y_test, label='Actual', color='blue')
plt.scatter(X_test, y_pred, label='Predicted', color='red')
plt.plot(X_test, y_pred, color='green')
plt.title("MARS Regression")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
