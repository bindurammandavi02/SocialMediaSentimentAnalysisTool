
from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# Load trained ML model
model = joblib.load("sentiment_model.pkl")

# Store total sentiment analysis
sentiment_counts = {
    "Positive": 0,
    "Negative": 0,
    "Neutral": 0
}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


def get_emoji(sentiment):
    if sentiment == "Positive":
        return "😊"
    elif sentiment == "Negative":
        return "😞"
    else:
        return "😐"


@app.route("/")
def home():

    total = sum(sentiment_counts.values())

    percentages = {
        key: round((value / total) * 100, 2)
        if total > 0 else 0
        for key, value in sentiment_counts.items()
    }

    return render_template(
        "index.html",
        counts=sentiment_counts,
        percentages=percentages,
        total=total
    )


@app.route("/analyze", methods=["POST"])
def analyze():

    user_text = request.form.get("text", "")

    if not user_text.strip():
        return render_template(
            "index.html",
            error="Please enter some text!",
            counts=sentiment_counts
        )

    cleaned_text = clean_text(user_text)

    # ML Prediction
    prediction = model.predict([cleaned_text])[0]
    sentiment = str(prediction).strip().capitalize()

    # Check supported sentiment
    if sentiment not in sentiment_counts:
        return render_template(
            "index.html",
            error="Model returned an unsupported sentiment: " + sentiment,
            counts=sentiment_counts
        )

    # Confidence score
    probabilities = model.predict_proba([cleaned_text])[0]
    confidence = max(probabilities) * 100

    # Update sentiment count
    sentiment_counts[sentiment] += 1

    total = sum(sentiment_counts.values())

    # Calculate percentages
    percentages = {
        key: round((value / total) * 100, 2)
        if total > 0 else 0
        for key, value in sentiment_counts.items()
    }

    return render_template(
        "index.html",
        sentiment=sentiment,
        emoji=get_emoji(sentiment),
        confidence=round(confidence, 2),
        user_text=user_text,
        counts=sentiment_counts,
        percentages=percentages,
        total=total
    )


if __name__ == "__main__":
    app.run(debug=True)
