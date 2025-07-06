from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Sample dataset
texts = [
    "India wins the world cup", 
    "Election campaigns are heating up", 
    "New iPhone launched", 
    "Football league results",
    "Budget announced in parliament",
    "Google unveils new AI chip"
]
labels = ["sports", "politics", "tech", "sports", "politics", "tech"]

# Build and train the model
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(texts, labels)

# 🔸 Get user input
user_input = input("Enter a news headline: ")

# 🔹 Predict category
prediction = model.predict([user_input])
print(f"\nPredicted Category: {prediction[0].capitalize()}")
