import re
import string

import nltk
import spacy

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

try:

    NLP = spacy.load("en_core_web_sm")

except OSError:

    # Fallback: download model automatically (happens once)

    import subprocess, sys

    # Runs the spaCy download command in the current Python environment

    subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        check=True,
    )

    NLP = spacy.load("en_core_web_sm")


def remove_urls(
        text: str
        ) -> str:
    
    """Strip http/https URLs from text."""

    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_emails(
        text: str
        ) -> str:

    """Strip email addresses from text."""

    return re.sub(r"\S+@\S+", " ", text)


def remove_special_characters(
        text: str
        ) -> str:
    """
    Keep only alphanumeric characters and spaces.
    Replaces punctuation and other symbols with a space.
    """

    # Allow hyphens inside words (e.g. 'full-stack') but remove standalone ones
    text = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text


def tokenize(
        text: str
        ) -> list[str]:
    """Split text into lowercase word tokens."""

    return text.lower().split()


def remove_stopwords(
        tokens: list[str]
        ) -> list[str]:
    """Filter out NLTK English stopwords."""

    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def lemmatize(
        tokens: list[str]
        ) -> list[str]:
    """
    Lemmatise tokens using spaCy.
    Processes the joined string for better POS tagging context.
    """

    doc = NLP(" ".join(tokens))

    return [token.lemma_ for token in doc if not token.is_space]


def preprocess_text(
        text: str
        ) -> str:
    """
    Full preprocessing pipeline.

    Parameters:
    text : str
        Raw input text (resume or job description).

    Returns:
    str
        Cleaned, lemmatised, stopword-free string ready for embedding.
    """
    if not text or not text.strip():
        return ""

    text = remove_urls(text)
    text = remove_emails(text)
    text = remove_special_characters(text)
    text = text.lower()

    tokens = tokenize(text)

    tokens = remove_stopwords(tokens)

    tokens = lemmatize(tokens)

    tokens = [t for t in tokens if len(t) > 1]

    return " ".join(tokens)
