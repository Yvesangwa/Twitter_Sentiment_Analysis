from flask import Flask, request, jsonify
import joblib
import re
from pathlib import Path

app = Flask(__name__)

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load trained model and TF-IDF vectorizer
model = joblib.load(
    BASE_DIR / "models" / "logistic_regression_model.pkl"
)

tfidf = joblib.load(
    BASE_DIR / "models" / "tfidf_vectorizer.pkl"
)

# Clean tweet
def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Home endpoint
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Twitter Sentiment Analysis API is running"
    })

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict) or not isinstance(data.get("tweet"), str):
        return jsonify({
            "error": "Please provide a tweet as a string"
        }), 400

    tweet = data["tweet"]

    cleaned = clean_tweet(tweet)

    if not cleaned:
        return jsonify({
            "error": "Tweet contains no usable text"
        }), 400

    features = tfidf.transform([cleaned])
    prediction = int(model.predict(features)[0])

    sentiment = "Positive" if prediction == 1 else "Negative"

    return jsonify({
        "tweet": tweet,
        "sentiment": sentiment
    })

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=5000)
