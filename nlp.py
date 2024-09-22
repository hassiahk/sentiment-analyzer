import re

import nltk
import pandas as pd
import plotly.express as px
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download("vader_lexicon")
nltk.download("punkt_tab")

vader = SentimentIntensityAnalyzer()
ps = PorterStemmer()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)

    tokens = word_tokenize(text)
    stemmed_tokens = [ps.stem(token) for token in tokens]

    doc = nlp(" ".join(stemmed_tokens))
    lemmatized_tokens = [token.lemma_ for token in doc]

    return " ".join(lemmatized_tokens)


# Function to perform sentiment analysis using VADER
def analyze_sentiment_vader(text):
    preprocessed_text = preprocess_text(text)
    scores = vader.polarity_scores(preprocessed_text)

    if scores["compound"] >= 0.05:
        sentiment = "Positive"
    elif scores["compound"] <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment, scores


# Function to display sentiment using Plotly
def display_sentiment_plotly(sentiment, scores):
    df = pd.DataFrame(
        {
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Score": [scores["pos"], scores["neu"], scores["neg"]],
        }
    )

    color_sequence = ["green", "gray", "red"]

    fig = px.bar(
        df,
        x="Sentiment",
        y="Score",
        color="Sentiment",
        title="Sentiment Analysis Result",
        labels={"Score": "VADER Sentiment Score"},
        color_discrete_sequence=color_sequence,
    )

    return fig
