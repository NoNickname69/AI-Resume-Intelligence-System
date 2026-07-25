"""
utils.py
=========
General-purpose utility functions used across the project.
"""

import datetime


def validate_inputs(uploaded_file, job_description: str) -> tuple[bool, str]:
    """
    Validate user inputs before running the NLP pipeline.

    Parameters
    ----------
    uploaded_file : streamlit UploadedFile or None
        The uploaded PDF resume.
    job_description : str
        The pasted job description text.

    Returns
    -------
    tuple[bool, str]
        (is_valid, error_message)
        is_valid is True when all checks pass; error_message is '' in that case.
    """
    if uploaded_file is None:
        return False, "Please upload a resume PDF before analysing."

    if not job_description or not job_description.strip():
        return False, "Please paste a job description before analysing."

    if len(job_description.strip()) < 50:
        return False, (
            "The job description seems too short (< 50 characters). "
            "Please paste the full description for accurate results."
        )

    return True, ""


def log_info(message: str) -> None:
    """
    Print a timestamped info log to stdout.

    Parameters
    ----------
    message : str
        Message to log.
    """
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] INFO  {message}")


def truncate_text(text: str, max_chars: int = 5000) -> str:
    """
    Truncate text to a maximum character count.

    Sentence Transformers have a token limit (~256 tokens for MiniLM).
    Very long documents are truncated to keep embedding quality high.

    Parameters
    ----------
    text : str
        Input text.
    max_chars : int
        Maximum allowed characters.

    Returns
    -------
    str
        Truncated text.
    """
    if len(text) <= max_chars:
        return text
    log_info(f"Text truncated from {len(text)} to {max_chars} characters.")
    return text[:max_chars]


def percentage_overlap(set_a: set, set_b: set) -> float:
    """
    Compute Jaccard-style overlap percentage between two sets.

    Returns the percentage of set_b that is covered by set_a.

    Parameters
    ----------
    set_a : set
        Reference set (e.g. resume skills).
    set_b : set
        Target set (e.g. JD skills).

    Returns
    -------
    float
        Coverage percentage (0.0 - 100.0). Returns 0.0 if set_b is empty.
    """
    if not set_b:
        return 0.0
    overlap = len(set_a & set_b)
    return round((overlap / len(set_b)) * 100, 1)