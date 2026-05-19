# FastAPI app - handles predictions and stores reviews in the database
# start with: python -m uvicorn main:app --reload
# docs auto-generated at http://127.0.0.1:8000/docs

import os
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from preprocess import clean_text
from database import init_db, save_review, get_reviews, delete_review

app = FastAPI(title="Movie Sentiment API", version="2.0")

# needed so the HTML frontend can talk to this API without CORS errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_model")
_vectorizer = None
_clf = None


@app.on_event("startup")
def startup():
    init_db()


def get_model():
    global _vectorizer, _clf
    if _vectorizer is None:
        vec_path = os.path.join(SAVE_DIR, "vectorizer.pkl")
        clf_path = os.path.join(SAVE_DIR, "classifier.pkl")
        if not os.path.exists(vec_path) or not os.path.exists(clf_path):
            raise FileNotFoundError("Model not found - run train.py first")
        _vectorizer = joblib.load(vec_path)
        _clf = joblib.load(clf_path)
    return _vectorizer, _clf


class ReviewInput(BaseModel):
    movie: str
    review: str


class PredictionResult(BaseModel):
    id: int
    movie: str
    sentiment: str
    confidence: float
    cleaned_text: str


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResult)
def predict(input: ReviewInput):
    if not input.movie.strip():
        raise HTTPException(status_code=400, detail="Movie title can't be empty")
    if not input.review.strip():
        raise HTTPException(status_code=400, detail="Review can't be empty")

    try:
        vectorizer, clf = get_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    cleaned = clean_text(input.review)
    features = vectorizer.transform([cleaned])
    label = clf.predict(features)[0]
    confidence = round(float(clf.predict_proba(features).max()) * 100, 1)
    sentiment = "Positive" if label == 1 else "Negative"

    review_id = save_review(input.movie, input.review, sentiment, confidence)

    return PredictionResult(
        id=review_id,
        movie=input.movie,
        sentiment=sentiment,
        confidence=confidence,
        cleaned_text=cleaned,
    )


@app.get("/reviews")
def list_reviews():
    return get_reviews(limit=50)


@app.delete("/reviews/{review_id}")
def remove_review(review_id: int):
    deleted = delete_review(review_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Review not found")
    return {"deleted": review_id}
