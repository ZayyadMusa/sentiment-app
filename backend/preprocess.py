import re


def clean_text(text: str) -> str:
    # IMDB reviews have raw HTML in them like <br /> so I strip those out first
    # then I lowercase everything so "Great" and "great" aren't treated differently
    # I only keep letters - punctuation and numbers don't really help with sentiment
    # finally collapse any extra spaces left behind after all the removal
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
