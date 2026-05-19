# run this once to train and save the model before starting the server
# cd backend && python train.py

import os
import joblib
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from preprocess import clean_text

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_model")


def load_imdb():
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")
    train_texts = [clean_text(t) for t in dataset["train"]["text"]]
    train_labels = dataset["train"]["label"]
    return train_texts, train_labels


def train(texts, labels):
    print(f"Fitting TF-IDF on {len(texts):,} reviews...")
    # ngram_range=(1,2) includes word pairs - "not good" means something different to just "good"
    vectorizer = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2), sublinear_tf=True)
    X = vectorizer.fit_transform(texts)

    print("Training classifier...")
    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    clf.fit(X, labels)

    print(f"Training accuracy: {accuracy_score(labels, clf.predict(X)):.4f}")
    return vectorizer, clf


def save(vectorizer, clf):
    os.makedirs(SAVE_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(SAVE_DIR, "vectorizer.pkl"))
    joblib.dump(clf, os.path.join(SAVE_DIR, "classifier.pkl"))
    print(f"Saved to {SAVE_DIR}/")


if __name__ == "__main__":
    texts, labels = load_imdb()
    vectorizer, clf = train(texts, labels)
    save(vectorizer, clf)
    print("Done - run evaluate.py next to check test accuracy")
