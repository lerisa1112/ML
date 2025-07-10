from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Sample text dataset
texts = [
    "India won the cricket match against Australia",
    "The new smartphone was launched by Samsung",
    "Politics is heating up before the elections",
    "The football match was exciting and intense",
    "Google releases a new AI model",
    "The president gave a speech on economic policy",
    "Manchester United signed a new striker",
    "Apple unveiled its latest MacBook",
    "The government passed a new education bill",
    "Barcelona defeated Real Madrid in El Clasico"
]

# Corresponding labels
labels = [
    "sports",
    "tech",
    "politics",
    "sports",
    "tech",
    "politics",
    "sports",
    "tech",
    "politics",
    "sports"
]

# Step 1: Convert text to feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Step 2: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Step 3: Train Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 4: Predict on test data
y_pred = model.predict(X_test)

# Step 5: Evaluate model
accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", metrics.classification_report(y_test, y_pred))
