"""
Evaluation script — run after train.py to measure real-world performance.

What this teaches:
  - Why train accuracy alone is misleading (overfitting)
  - Test-set evaluation: accuracy, precision, recall, F1
  - Confusion matrix: visualizes false positives vs false negatives

Usage:
    cd backend
    python evaluate.py
"""

import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from preprocess import clean_text

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_model")


def load_test_data():
    print("Loading IMDB test split (25,000 reviews)...")
    dataset = load_dataset("imdb")
    texts = [clean_text(t) for t in dataset["test"]["text"]]
    labels = dataset["test"]["label"]
    return texts, labels


def evaluate():
    vectorizer = joblib.load(os.path.join(SAVE_DIR, "vectorizer.pkl"))
    clf = joblib.load(os.path.join(SAVE_DIR, "classifier.pkl"))

    texts, labels = load_test_data()
    X = vectorizer.transform(texts)
    predictions = clf.predict(X)

    acc = accuracy_score(labels, predictions)
    print(f"\nTest accuracy: {acc:.4f}  ({acc*100:.1f}%)")
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=["Negative", "Positive"]))

    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.title("Confusion Matrix — IMDB Sentiment")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"))
    print("\nConfusion matrix saved to saved_model/confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    evaluate()
