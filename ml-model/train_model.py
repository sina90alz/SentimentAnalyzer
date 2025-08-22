# ml-model/train_model.py
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Example training data (you can expand later)
texts = [
    "I love programming in C#",
    "This project is terrible",
    "AI is amazing",
    "I hate bugs in code",
    "Learning is fun",
    "This is frustrating"
]
labels = ["positive", "negative", "positive", "negative", "positive", "negative"]

# Vectorize and train
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, labels)

# Save model + vectorizer
joblib.dump((model, vectorizer), "sentiment_model.pkl")
print("Model trained and saved as sentiment_model.pkl")
