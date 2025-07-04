# spam_detector.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1️⃣ Load data ──────────────────────────────────────────────────────────────
# Expect a CSV with two columns: "label" (spam|ham) and "text"
# Example: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
df = pd.read_csv("spam.csv", encoding="latin‑1")[['v1', 'v2']]
df.columns = ['label', 'text']          # rename for clarity
df['label']  = df.label.map({'ham':0, 'spam':1})

# 2️⃣ Split data ─────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df.text, df.label, test_size=0.2, random_state=42, stratify=df.label)

# 3️⃣ Build pipeline: TF‑IDF → Multinomial NB ───────────────────────────────
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        ngram_range=(1,2),        # unigrams + bigrams
        min_df=2,                 # ignore very rare tokens
        sublinear_tf=True)),
    ('nb', MultinomialNB(alpha=0.3))
])

# 4️⃣ Train ─────────────────────────────────────────────────────────────────
model.fit(X_train, y_train)

# 5️⃣ Evaluate ──────────────────────────────────────────────────────────────
preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.4f}\n")
print(classification_report(y_test, preds, target_names=['ham', 'spam']))

# 6️⃣ Save the trained model ────────────────────────────────────────────────
joblib.dump(model, "spam_detector.joblib")
print("✅ Model saved to spam_detector.joblib")

# 7️⃣ Quick demo on new e‑mails ─────────────────────────────────────────────
sample_emails = [
    "Congratulations! You’ve won a free cruise to the Bahamas. Call now!",
    "Hi Mom, can you send me the recipe for tonight’s dinner?"
]
print("\nSample predictions:")
for mail, pred in zip(sample_emails, model.predict(sample_emails)):
    print(f"📝 {mail[:50]}…  →  {'SPAM' if pred else 'HAM'}")
