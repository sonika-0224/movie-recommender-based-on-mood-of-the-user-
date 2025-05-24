import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

def train_text_emotion_model():
    df = pd.read_csv("data/text/train.txt", sep=";", names=["text", "emotion"])
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(max_iter=1000))
    ])
    model.fit(df['text'], df['emotion'])
    joblib.dump(model, 'models/text_emotion.pkl')
    print("Text emotion model trained and saved.")

def predict_emotion(text):
    model = joblib.load('models/text_emotion.pkl')
    return model.predict([text])[0]
