import re


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)      # IMDB reviews have raw HTML like <br />
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
