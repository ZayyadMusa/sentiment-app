# Movie Sentiment Tracker

I built this as one of my first proper ML projects while studying for my MSc in AI. The goal was to actually go through the full pipeline — not just train a model in a notebook and call it done, but deploy it properly with an API and a UI.

The idea is simple: you type a movie title, write what you thought about it, and the app tells you whether your review is positive or negative (and how confident it is). Every review you submit gets saved to a database, so you end up with a log of your opinions over time.

It's trained on 50,000 IMDB movie reviews using TF-IDF + Logistic Regression, which sounds basic but hits ~89% accuracy on unseen data. Good enough to feel real.

---

## How to run it

**Install dependencies** (first time only):

```bash
cd backend
pip install -r requirements.txt
```

**Train the model** (first time only, takes ~2 minutes):

```bash
python train.py
```

**Start the API:**

```bash
python -m uvicorn main:app --reload
```

**Open the app:**

Open `frontend/index.html` in your browser. The API needs to be running for it to work.

You can also go to `http://127.0.0.1:8000/docs` to test the API directly in the browser — FastAPI generates that automatically which is pretty handy.

---

## How it works

When you submit a review, the text goes through a few steps:

1. **Cleaning** — strips HTML tags, lowercases everything, removes punctuation
2. **TF-IDF vectorization** — converts the words into numbers based on how often they appear across all reviews
3. **Logistic Regression** — predicts positive or negative based on those numbers
4. **Storage** — the result (movie, review, sentiment, confidence, timestamp) gets saved to a local SQLite database

The history table on the page loads directly from that database, so it persists even after you close and reopen the page.

---

## What I learned building this

- You need to explore your data before touching any model — the EDA notebook (`notebooks/exploration.ipynb`) showed me that IMDB reviews contain raw HTML and that the class split is perfectly 50/50, both of which actually matter
- Training accuracy being high (~93%) doesn't mean much on its own — test accuracy (~89%) is what you care about
- Serving an ML model as an API is straightforward with FastAPI once the model is saved — it just loads the `.pkl` files and runs predictions on request
- CORS is the thing that trips everyone up when a frontend tries to talk to a local API for the first time

---

## What's next

The plan is to swap out the Logistic Regression for a fine-tuned DistilBERT model, which should push accuracy closer to 93%. The rest of the code stays the same — that's the point of keeping the preprocessing and API separate.
