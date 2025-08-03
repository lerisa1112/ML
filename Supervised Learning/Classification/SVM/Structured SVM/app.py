from pystruct.models import ChainCRF
from pystruct.learners import StructuredSVM
from sklearn.model_selection import train_test_split
import numpy as np

# Example: 3 sequences with 4 features per token
X = [np.array([[1, 0], [0, 1], [1, 1], [0, 0]]),
     np.array([[0, 0], [1, 1], [0, 1], [1, 0]]),
     np.array([[1, 1], [1, 0], [0, 1], [0, 0]])]

# Corresponding labels (sequence of labels)
Y = [np.array([0, 1, 1, 0]),
     np.array([0, 1, 1, 0]),
     np.array([1, 0, 1, 0])]

# Create ChainCRF model
model = ChainCRF()

# Structured SVM
ssvm = StructuredSVM(model=model, max_iter=100, C=1.0)
ssvm.fit(X, Y)

# Predict
Y_pred = ssvm.predict(X)

print("Predictions:")
for pred in Y_pred:
    print(pred)
