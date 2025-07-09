import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------
# STEP 1: CREATE SAMPLE DATASET
# ------------------------------
data = {
    "text": [
        "The moon landing in 1969 was real and televised live.",
        "Breaking: Aliens have landed in Paris and are dancing!",
        "NASA confirms water found on Mars.",
        "The president of the world is a robot in disguise!",
        "COVID-19 vaccines are effective and safe.",
        "Scientists claim the Earth is flat and spinning backward!",
        "Stock markets see growth after economic recovery.",
        "Bill Gates admits to putting microchips in vaccines!",
        "Local elections held peacefully in all districts.",
        "Government bans oxygen to control the population!"
    ],
    "label": [
        "REAL", "FAKE", "REAL", "FAKE", "REAL",
        "FAKE", "REAL", "FAKE", "REAL", "FAKE"
    ]
}

df = pd.DataFrame(data)

# Save to CSV for reference
df.to_csv("fake_or_real_news.csv", index=False)
print("Sample dataset saved as 'fakkorreal.csv'.")

# ------------------------------
# STEP 2: PREPROCESS & SPLIT
# ------------------------------
df.dropna(inplace=True)

X = df['text']
y = df['label'].map({'FAKE': 0, 'REAL': 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# STEP 3: TEXT VECTORIZATION
# ------------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ------------------------------
# STEP 4: TRAIN MODEL
# ------------------------------
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# ------------------------------
# STEP 5: EVALUATE
# ------------------------------
y_pred = model.predict(X_test_tfidf)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------
# STEP 6: CUSTOM INPUT PREDICTION
# ------------------------------
def detect_news(news_text):
    input_tfidf = vectorizer.transform([news_text])
    pred = model.predict(input_tfidf)
    return "REAL" if pred[0] == 1 else "FAKE"

# Example Usage
print("\nTry custom input:")
test_news = "Government announces new tax reforms for middle class."
print(f"News: {test_news}")
print("Prediction:", detect_news(test_news))
