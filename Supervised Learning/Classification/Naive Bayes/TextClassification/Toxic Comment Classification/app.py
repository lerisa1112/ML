# toxic_comment_classifier.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load Dataset
# You can download the dataset from: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
data = pd.read_csv("train.csv")

# 2. Check basic info
print("Sample data:\n", data.head())
print("Null values:\n", data.isnull().sum())

# 3. Keep only 'comment_text' and 'toxic' columns for binary classification
data = data[['comment_text', 'toxic']]

# Drop nulls
data.dropna(inplace=True)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(data['comment_text'], data['toxic'], test_size=0.2, random_state=42)

# 5. TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Model Training
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 7. Prediction
y_pred = model.predict(X_test_vec)

# 8. Evaluation
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 9. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Toxic', 'Toxic'], yticklabels=['Non-Toxic', 'Toxic'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 10. Predict on Custom Comments
def predict_comment(comment):
    comment_vec = vectorizer.transform([comment])
    prediction = model.predict(comment_vec)[0]
    return "Toxic" if prediction == 1 else "Non-Toxic"

# Test custom prediction
sample = "You're so stupid and ugly!"
print(f"Comment: '{sample}' → Prediction: {predict_comment(sample)}")
