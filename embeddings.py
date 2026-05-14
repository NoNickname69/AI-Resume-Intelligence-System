"""
embeddings.py
==============
Provides two embedding strategies:

1. TF-IDF  — classic sparse vector representation (scikit-learn)
2. Sentence Transformers — dense semantic embeddings (all-MiniLM-L6-v2)

Both functions accept two pre-processed text strings and return
numpy array vectors ready for cosine similarity computation.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# ── Sentence Transformer model (loaded once, cached by Python) ────────────────
# all-MiniLM-L6-v2 is a lightweight, high-quality model (~80 MB)
_MODEL_NAME = "all-MiniLM-L6-v2"
_transformer_model: SentenceTransformer | None = None


def _get_transformer_model() -> SentenceTransformer:
    """Lazy-load the Sentence Transformer model (singleton)."""
    global _transformer_model
    if _transformer_model is None:
        print(f"[embeddings] Loading model '{_MODEL_NAME}'… (first run only)")
        _transformer_model = SentenceTransformer(_MODEL_NAME)
    return _transformer_model


# ── Method 1 : TF-IDF ────────────────────────────────────────────────────────

def get_tfidf_embeddings(
    resume_text: str,
    jd_text: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute TF-IDF sparse vectors for the resume and job description.

    Parameters
    ----------
    resume_text : str
        Preprocessed resume text.
    jd_text : str
        Preprocessed job description text.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (resume_vector, jd_vector) as dense numpy arrays.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams for richer context
        max_features=10_000,  # cap vocabulary size
        sublinear_tf=True,    # apply log normalisation
    )

    # Fit on both documents so the vocabulary covers both
    tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])

    resume_vec = tfidf_matrix[0].toarray().flatten()
    jd_vec     = tfidf_matrix[1].toarray().flatten()

    return resume_vec, jd_vec


# ── Method 2 : Sentence Transformers ─────────────────────────────────────────

def get_transformer_embeddings(
    resume_text: str,
    jd_text: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute dense semantic embeddings using all-MiniLM-L6-v2.

    Parameters
    ----------
    resume_text : str
        Preprocessed resume text.
    jd_text : str
        Preprocessed job description text.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (resume_vector, jd_vector) as 384-dim numpy arrays.
    """
    model = _get_transformer_model()

    # Encode both at once (batched → faster)
    embeddings = model.encode(
        [resume_text, jd_text],
        normalize_embeddings=True,   # unit-normalise → cosine = dot product
        show_progress_bar=False,
    )

    resume_vec = embeddings[0]
    jd_vec     = embeddings[1]

    return resume_vec, jd_vec
