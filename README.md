# Sentiment Analysis Web App

An end-to-end NLP project: trains a sentiment classifier on 50,000 IMDB movie
reviews and serves it through a FastAPI backend + plain HTML/CSS/JS frontend.

## What I built and learned

| Step | What it teaches |
|---|---|
| EDA notebook | Always explore your data before modeling |
| TF-IDF vectorizer | How text becomes numbers a model can learn from |
| Logistic Regression | A linear classifier well-suited for high-dimensional sparse features |
| Train/test split | Why training accuracy alone is misleading |
| FastAPI endpoint | How to serve an ML model as a REST API |
| HTML + fetch() | How a frontend talks to a backend API |

## Project structure

```
sentiment-app/
├── backend/
│   ├── main.py          # FastAPI app (prediction endpoint)
│   ├── train.py         # Training script — run once
│   ├── evaluate.py      # Metrics: accuracy, F1, confusion matrix
│   ├── preprocess.py    # Text cleaning helpers
│   ├── saved_model/     # Serialized model files (not in git)
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── notebooks/
│   └── exploration.ipynb   # EDA on the IMDB dataset
└── README.md
```

## Quick start

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Train the model (one-time, ~2 min)

```bash
cd backend
python train.py
```

Expected output: `Training accuracy: ~0.998` (high — the model fits the training data)

### 3. Evaluate on the test set

```bash
python evaluate.py
```

Expected: ~88–90% test accuracy. Saves a confusion matrix to `saved_model/confusion_matrix.png`.

### 4. Start the API server

```bash
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000/docs` — interactive Swagger UI to test the endpoint.

### 5. Open the frontend

Open `frontend/index.html` directly in your browser. Type a review and click Analyse.

**Tip:** Press `Ctrl+Enter` to submit without clicking.

## API reference

### `POST /predict`

**Request body:**
```json
{ "text": "This film was an absolute masterpiece." }
```

**Response:**
```json
{
  "sentiment": "Positive",
  "confidence": 96.3,
  "cleaned_text": "this film was an absolute masterpiece"
}
```

## Model performance

| Metric | Score |
|---|---|
| Test accuracy | ~89% |
| Positive F1 | ~89% |
| Negative F1 | ~89% |

## Upgrade path

To push accuracy to ~93%, fine-tune `distilbert-base-uncased` using
HuggingFace `Trainer` API — the `preprocess.py` cleaning step stays the same.
