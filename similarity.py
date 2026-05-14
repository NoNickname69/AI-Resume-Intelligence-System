"""
similarity.py
==============
Cosine similarity calculation between two embedding vectors.

Works with both:
  - Sparse TF-IDF dense arrays  (sklearn)
  - Dense Sentence Transformer arrays (numpy)
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


def compute_cosine_similarity(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
) -> float:
    """
    Compute cosine similarity between two 1-D numpy vectors.

    Formula
    -------
    cosine_similarity = (A · B) / (||A|| * ||B||)

    Parameters
    ----------
    vec_a : np.ndarray
        First embedding vector (resume).
    vec_b : np.ndarray
        Second embedding vector (job description).

    Returns
    -------
    float
        Similarity score in the range [0.0, 1.0].
        Returns 0.0 if either vector has zero norm.
    """
    # Reshape to (1, N) for sklearn compatibility
    a = vec_a.reshape(1, -1)
    b = vec_b.reshape(1, -1)

    try:
        score = sklearn_cosine(a, b)[0][0]
        # Clamp to [0, 1] to handle floating-point edge cases
        return float(np.clip(score, 0.0, 1.0))
    except Exception as e:
        print(f"[similarity] Cosine similarity failed: {e}")
        return 0.0


def interpret_score(score_pct: float) -> str:
    """
    Convert a numeric match percentage to a human-readable label.

    Parameters
    ----------
    score_pct : float
        Match percentage (0 – 100).

    Returns
    -------
    str
        Label such as 'Excellent', 'Good', 'Fair', 'Weak'.
    """
    if score_pct >= 80:
        return "Excellent"
    elif score_pct >= 60:
        return "Good"
    elif score_pct >= 40:
        return "Fair"
    else:
        return "Weak"
