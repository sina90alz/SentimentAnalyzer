# ml-model/predict.py
import sys
import joblib

# Load model + vectorizer
model, vectorizer = joblib.load("sentiment_model.pkl")

def predict_sentiment(text: str):
    X = vectorizer.transform([text])
    return model.predict(X)[0]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py 'your text here'")
    else:
        text = sys.argv[1]
        sentiment = predict_sentiment(text)
        print(sentiment)
