"""
FastAPI app — serves the trained sentiment model as a REST API.

Endpoints:
  POST /predict  — accepts text, returns sentiment + confidence
  GET  /         — health check

Run with:
    cd backend
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""

import os
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from preprocess import clean_text

app = FastAPI(title="Sentiment Analysis API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_model")
_vectorizer = None
_clf = None


def get_model():
    global _vectorizer, _clf
    if _vectorizer is None:
        vec_path = os.path.join(SAVE_DIR, "vectorizer.pkl")
        clf_path = os.path.join(SAVE_DIR, "classifier.pkl")
        if not os.path.exists(vec_path) or not os.path.exists(clf_path):
            raise FileNotFoundError("Model not found. Run train.py first.")
        _vectorizer = joblib.load(vec_path)
        _clf = joblib.load(clf_path)
    return _vectorizer, _clf


class TextInput(BaseModel):
    text: str


class PredictionResult(BaseModel):
    sentiment: str
    confidence: float
    cleaned_text: str


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Sentiment API is running"}


@app.post("/predict", response_model=PredictionResult)
def predict(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        vectorizer, clf = get_model()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    cleaned = clean_text(input.text)
    features = vectorizer.transform([cleaned])
    label = clf.predict(features)[0]
    confidence = float(clf.predict_proba(features).max())

    return PredictionResult(
        sentiment="Positive" if label == 1 else "Negative",
        confidence=round(confidence * 100, 1),
        cleaned_text=cleaned,
    )
