import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Force NLTK to look in the correct folder
nltk.data.path.append("C:/Users/Lerisa/nltk_data")  # Use your username if different

# Download lexicon if not already done
nltk.download('vader_lexicon')

# Function
def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    score = sid.polarity_scores(text)['compound']
    if score >= 0.5:
        return "😄 Positive"
    elif score <= -0.5:
        return "😠 Negative"
    else:
        return "😐 Neutral"

# Test
texts = [
    "I love this product!",
    "This is terrible!",
    "It's okay."
]

for text in texts:
    print(f"Text: {text} → Sentiment: {analyze_sentiment(text)}")
