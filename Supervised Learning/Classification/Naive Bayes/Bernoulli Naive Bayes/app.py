from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset (messages and labels)
messages = [
    "Free entry in a contest", "Win cash prizes now", "Call me later",
    "Are you coming to the party?", "Claim your free prize", "Let's catch up soon",
    "Congratulations, you won!", "Don't forget our meeting"
]
labels = [1, 1, 0, 0, 1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam

# 1. Vectorize text to binary features (presence/absence of words)
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(messages)

# 2. Split dataset into train/test
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# 3. Train Bernoulli Naive Bayes model
bnb = BernoulliNB()
bnb.fit(X_train, y_train)

# 4. Predict on test data
y_pred = bnb.predict(X_test)

# 5. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
