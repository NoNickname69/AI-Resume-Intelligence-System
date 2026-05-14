"""
analysis.py
============
Keyword / skill extraction and overlap analysis.

Responsibilities:
  - Maintain a curated tech-skill vocabulary
  - Extract skills present in a given text
  - Identify matched and missing skills
  - Generate a plain-English recommendation
"""

import re

# ── Curated skill vocabulary ──────────────────────────────────────────────────
# Extend this list to cover more domains / roles.

SKILL_VOCABULARY: list[str] = [
    # Programming languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "kotlin", "swift", "r", "scala", "matlab", "php", "ruby", "bash", "shell",

    # ML / AI
    "machine learning", "deep learning", "neural network", "nlp",
    "natural language processing", "computer vision", "reinforcement learning",
    "transfer learning", "llm", "large language model", "generative ai",
    "transformers", "bert", "gpt", "diffusion model",

    # ML frameworks / libraries
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
    "hugging face", "sentence transformers", "spacy", "nltk", "gensim",
    "xgboost", "lightgbm", "catboost", "fastai",

    # Data / analytics
    "pandas", "numpy", "matplotlib", "seaborn", "plotly", "tableau",
    "power bi", "sql", "mysql", "postgresql", "sqlite", "mongodb",
    "elasticsearch", "spark", "hadoop", "dbt", "airflow",

    # Web / backend
    "fastapi", "flask", "django", "node.js", "express", "react",
    "vue", "angular", "next.js", "graphql", "rest api", "restful",
    "html", "css", "tailwind",

    # Cloud / DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "terraform", "ci/cd", "github actions", "jenkins", "linux",
    "git", "github", "gitlab",

    # Vector / search
    "pinecone", "weaviate", "chroma", "faiss", "vector database",
    "semantic search", "rag", "retrieval augmented generation",

    # General engineering
    "api", "microservices", "agile", "scrum", "object oriented",
    "data structures", "algorithms", "system design",

    # Soft skills
    "communication", "teamwork", "leadership", "problem solving",
    "critical thinking", "time management",
]

# Pre-compile lowercase set for fast lookup
_SKILL_SET_LOWER = {s.lower() for s in SKILL_VOCABULARY}


def _normalise(text: str) -> str:
    """Lowercase and strip extra whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def extract_skills(text: str) -> set[str]:
    """
    Scan `text` for known skills from SKILL_VOCABULARY.

    Uses whole-word matching to avoid false positives
    (e.g. 'r' inside 'learning').

    Parameters
    ----------
    text : str
        Raw (non-preprocessed) text — preserves multi-word phrases.

    Returns
    -------
    set[str]
        Set of identified skill strings (lowercase).
    """
    text_lower = _normalise(text)
    found: set[str] = set()

    for skill in SKILL_VOCABULARY:
        skill_lower = skill.lower()
        # Word-boundary match
        pattern = r"\b" + re.escape(skill_lower) + r"\b"
        if re.search(pattern, text_lower):
            found.add(skill_lower)

    return found


def get_matched_skills(
    resume_skills: set[str],
    jd_skills: set[str],
) -> list[str]:
    """
    Return skills that appear in BOTH the resume and job description.

    Parameters
    ----------
    resume_skills : set[str]
        Skills extracted from the resume.
    jd_skills : set[str]
        Skills extracted from the job description.

    Returns
    -------
    list[str]
        Sorted list of matched skills (title-cased for display).
    """
    matched = resume_skills & jd_skills
    return sorted(s.title() for s in matched)


def get_missing_skills(
    resume_skills: set[str],
    jd_skills: set[str],
) -> list[str]:
    """
    Return skills required by the JD but absent from the resume.

    Parameters
    ----------
    resume_skills : set[str]
        Skills extracted from the resume.
    jd_skills : set[str]
        Skills extracted from the job description.

    Returns
    -------
    list[str]
        Sorted list of missing skills (title-cased for display).
    """
    missing = jd_skills - resume_skills
    return sorted(s.title() for s in missing)


def generate_recommendation(
    score_pct: float,
    matched: list[str],
    missing: list[str],
) -> str:
    """
    Produce a concise plain-English recommendation based on the match score
    and skill overlap.

    Parameters
    ----------
    score_pct : float
        Similarity score as a percentage (0–100).
    matched : list[str]
        List of matched skills.
    missing : list[str]
        List of missing skills.

    Returns
    -------
    str
        Multi-sentence recommendation paragraph.
    """
    n_matched = len(matched)
    n_missing = len(missing)

    # ── Score-based opening ───────────────────────────────────────────────────
    if score_pct >= 80:
        opening = (
            f"🟢 **Excellent match!** Your resume aligns very strongly with "
            f"this role ({score_pct}% similarity)."
        )
    elif score_pct >= 60:
        opening = (
            f"🟡 **Good match.** Your resume shows solid alignment with this "
            f"role ({score_pct}% similarity), with room to strengthen a few areas."
        )
    elif score_pct >= 40:
        opening = (
            f"🟠 **Moderate match.** Your resume partially aligns with the job "
            f"description ({score_pct}% similarity). Consider tailoring it further."
        )
    else:
        opening = (
            f"🔴 **Weak match.** The semantic similarity is low ({score_pct}%). "
            f"The role may require significant upskilling or the resume needs rewriting."
        )

    # ── Skill summary ─────────────────────────────────────────────────────────
    if n_matched > 0:
        skill_line = (
            f"You matched **{n_matched}** required skill(s) "
            f"({', '.join(matched[:5])}{'…' if n_matched > 5 else ''})."
        )
    else:
        skill_line = "No direct skill matches were detected from the vocabulary."

    # ── Missing skill advice ──────────────────────────────────────────────────
    if n_missing == 0:
        missing_line = "You appear to cover all key skills mentioned in the job description. ✅"
    elif n_missing <= 3:
        missing_line = (
            f"You are missing **{n_missing}** skill(s) listed in the JD: "
            f"{', '.join(missing)}. Adding these to your resume (if you have them) "
            f"could boost your match score."
        )
    else:
        top_missing = missing[:5]
        missing_line = (
            f"There are **{n_missing}** skill gaps. Priority areas to develop: "
            f"{', '.join(top_missing)}{'…' if n_missing > 5 else ''}."
        )

    # ── Action prompt ─────────────────────────────────────────────────────────
    action = (
        "**Tip:** Tailor your resume to mirror the exact keywords in the job "
        "description. ATS systems often filter on keyword frequency before a "
        "human ever reads your application."
    )

    return "\n\n".join([opening, skill_line, missing_line, action])
