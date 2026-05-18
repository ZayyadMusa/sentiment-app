import re


def clean_text(text: str) -> str:
    """Remove HTML tags, punctuation, and extra whitespace from review text."""
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)       # strip HTML tags (common in IMDB reviews)
    text = re.sub(r"[^a-z\s]", "", text)    # keep only letters and spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text
