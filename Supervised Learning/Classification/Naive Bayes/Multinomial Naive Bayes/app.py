from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample dataset
texts = [
    "Free money now", "Call this number to claim your prize",
    "Meeting schedule for tomorrow", "Let's have lunch",
    "Win cash instantly", "Project report attached",
    "Earn extra income from home", "Budget discussion on Monday"
]

labels = [1, 1, 0, 0, 1, 0, 1, 0]  # 1 = spam, 0 = ham

# Step 1: Convert text to numeric features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Step 2: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Step 3: Train Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 4: Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Optional: Test on new data
new_texts = ["Win big prizes", "Team meeting at 10am"]
new_X = vectorizer.transform(new_texts)
predictions = model.predict(new_X)

for text, label in zip(new_texts, predictions):
    print(f"'{text}' => {'Spam' if label == 1 else 'Ham'}")
