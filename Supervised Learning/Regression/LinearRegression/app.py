# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
# X = independent variable (hours studied)
# Y = dependent variable (marks scored)
X = np.array([[1], [2], [3], [4], [5], [6]])
Y = np.array([35, 40, 50, 55, 65, 70])

# Create and train model
model = LinearRegression()
model.fit(X, Y)

# Predict marks for 7 hours of study
hours = np.array([[7]])
predicted_marks = model.predict(hours)

print(f"Predicted Marks for {hours[0][0]} hours of study: {predicted_marks[0]:.2f}")

# Plot the data
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Marks Scored')
plt.title('Linear Regression Example')
plt.legend()
plt.show()
