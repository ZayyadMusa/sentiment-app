"""
Training script — run this once to build and save the sentiment model.

What this teaches:
  - Loading a real NLP dataset
  - TF-IDF: converts text into numerical feature vectors
  - Logistic Regression: a linear classifier well-suited for text
  - Serializing a trained model so an API can reload it later

Usage:
    cd backend
    python train.py
"""

import os
import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from preprocess import clean_text

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_model")


def load_imdb():
    print("Downloading IMDB dataset (this may take a moment the first time)...")
    dataset = load_dataset("imdb")
    train_texts = [clean_text(t) for t in dataset["train"]["text"]]
    train_labels = dataset["train"]["label"]
    return train_texts, train_labels


def train(texts, labels):
    print(f"Fitting TF-IDF vectorizer on {len(texts):,} reviews...")
    vectorizer = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), sublinear_tf=True)
    X = vectorizer.fit_transform(texts)

    print("Training Logistic Regression classifier...")
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(X, labels)

    train_acc = accuracy_score(labels, clf.predict(X))
    print(f"Training accuracy: {train_acc:.4f}")
    return vectorizer, clf


def save(vectorizer, clf):
    os.makedirs(SAVE_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(SAVE_DIR, "vectorizer.pkl"))
    joblib.dump(clf, os.path.join(SAVE_DIR, "classifier.pkl"))
    print(f"Model saved to {SAVE_DIR}/")


if __name__ == "__main__":
    texts, labels = load_imdb()
    vectorizer, clf = train(texts, labels)
    save(vectorizer, clf)
    print("Done! Run evaluate.py next to check test-set accuracy.")
