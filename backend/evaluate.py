# checks how the model actually performs on data it has never seen
# run after train.py - cd backend && python evaluate.py

import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocess import clean_text

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_model")


def load_test_data():
    print("Loading test split...")
    dataset = load_dataset("imdb")
    texts = [clean_text(t) for t in dataset["test"]["text"]]
    labels = dataset["test"]["label"]
    return texts, labels


def evaluate():
    vectorizer = joblib.load(os.path.join(SAVE_DIR, "vectorizer.pkl"))
    clf = joblib.load(os.path.join(SAVE_DIR, "classifier.pkl"))

    texts, labels = load_test_data()
    X = vectorizer.transform(texts)
    preds = clf.predict(X)

    print(f"\nTest accuracy: {accuracy_score(labels, preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Negative", "Positive"]))

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    out = os.path.join(SAVE_DIR, "confusion_matrix.png")
    plt.savefig(out)
    print(f"\nConfusion matrix saved to {out}")
    plt.show()


if __name__ == "__main__":
    evaluate()
