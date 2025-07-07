from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Sample dataset: (Text, Language)
data = [
    ("Hello, how are you?", "English"),
    ("I am fine, thank you!", "English"),
    ("What is your name?", "English"),
    
    ("Bonjour, comment ça va?", "French"),
    ("Je vais bien, merci!", "French"),
    ("Quel est ton nom?", "French"),
    
    ("Hola, ¿cómo estás?", "Spanish"),
    ("Estoy bien, ¡gracias!", "Spanish"),
    ("¿Cuál es tu nombre?", "Spanish"),
    
    ("Hallo, wie geht es dir?", "German"),
    ("Mir geht es gut, danke!", "German"),
    ("Wie heißt du?", "German"),
]

# Step 1: Split data
texts, labels = zip(*data)
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

# Step 2: Convert text to numerical data
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Step 3: Train Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# Step 4: Predict on test set
y_pred = clf.predict(X_test_counts)

# Step 5: Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Try new sentences
test_sentences = [
    "Comment tu t'appelles?",
    "Where are you going?",
    "¿Dónde está el baño?",
    "Wie spät ist es?"
]
test_counts = vectorizer.transform(test_sentences)
predictions = clf.predict(test_counts)

for sentence, lang in zip(test_sentences, predictions):
    print(f"'{sentence}' => {lang}")
