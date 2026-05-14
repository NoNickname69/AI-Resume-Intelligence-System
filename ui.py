"""
ui.py
======
Reusable Streamlit UI components.

Keeps app.py clean by separating rendering logic from pipeline logic.
"""

import streamlit as st
from similarity import interpret_score


def render_header() -> None:
    """Render the application title and subtitle."""
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0 0.5rem;">
            <h1 style="font-size: 2.4rem; font-weight: 800; color: #1e293b; margin-bottom: 0.2rem;">
                🧠 AI Resume Intelligence System
            </h1>
            <p style="font-size: 1.05rem; color: #64748b;">
                Semantic resume ↔ job description matching powered by NLP embeddings
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """
    Render the sidebar with method selector and project info.

    Returns
    -------
    str
        Selected embedding method: 'TF-IDF' or 'Sentence Transformer'.
    """
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/000000/artificial-intelligence.png",
            width=72,
        )
        st.title("Settings")
        st.markdown("---")

        method = st.radio(
            label="🔧 Embedding Method",
            options=["TF-IDF", "Sentence Transformer"],
            index=1,
            help=(
                "**TF-IDF** — Fast, keyword-based sparse vectors.\n\n"
                "**Sentence Transformer** — Slow (first run), semantically richer dense vectors."
            ),
        )

        st.markdown("---")
        st.markdown(
            """
            **About**

            This MVP demonstrates:
            - 📄 PDF text extraction
            - 🔤 NLP preprocessing
            - 🔢 Embedding generation
            - 📐 Cosine similarity
            - 🔍 Skill gap analysis

            Built with Python · Streamlit · spaCy ·
            sentence-transformers · scikit-learn
            """
        )

        st.markdown("---")
        st.caption("AI Resume Intelligence System · MVP v1.0")

    return method


def render_score_card(score_pct: float, method: str) -> None:
    """
    Display the main match score as a styled card.

    Parameters
    ----------
    score_pct : float
        Match percentage (0–100).
    method : str
        Embedding method used.
    """
    label = interpret_score(score_pct)

    # Choose colour based on score
    if score_pct >= 80:
        gradient = "linear-gradient(135deg, #10b981 0%, #059669 100%)"
    elif score_pct >= 60:
        gradient = "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)"
    elif score_pct >= 40:
        gradient = "linear-gradient(135deg, #f97316 0%, #ea580c 100%)"
    else:
        gradient = "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)"

    st.markdown(
        f"""
        <div class="score-card" style="background: {gradient};">
            <div class="score-number">{score_pct}%</div>
            <div class="score-label">Resume Match Score — <strong>{label}</strong></div>
            <div style="font-size: 0.8rem; opacity: 0.8; margin-top: 0.5rem;">
                Method: {method}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_skills_section(
    title: str,
    skills: list[str],
    kind: str = "match",
) -> None:
    """
    Display a labelled list of skill pills.

    Parameters
    ----------
    title : str
        Section heading.
    skills : list[str]
        Skill names to display.
    kind : str
        'match' → green pills | 'missing' → red pills.
    """
    st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)

    if not skills:
        st.info("None detected." if kind == "match" else "All required skills found! 🎉")
        return

    pill_class = "skill-pill-match" if kind == "match" else "skill-pill-missing"
    pills_html = " ".join(
        f"<span class='{pill_class}'>{skill}</span>" for skill in skills
    )
    st.markdown(f"<div>{pills_html}</div>", unsafe_allow_html=True)

    # Count badge
    count_label = "matched" if kind == "match" else "missing"
    st.caption(f"{len(skills)} skill(s) {count_label}")


def render_analysis_summary(
    recommendation: str,
    score_pct: float,
    resume_text: str,
    jd_text: str,
) -> None:
    """
    Render the overall analysis section with metrics and recommendation text.

    Parameters
    ----------
    recommendation : str
        Markdown-formatted recommendation paragraph.
    score_pct : float
        Match percentage.
    resume_text : str
        Raw resume text (for word-count stats).
    jd_text : str
        Raw job description text.
    """
    st.markdown("#### 📊 Overall Analysis")

    # Quick stats row
    m1, m2, m3 = st.columns(3)
    m1.metric("Match Score", f"{score_pct}%")
    m2.metric("Resume Words", f"{len(resume_text.split()):,}")
    m3.metric("JD Words", f"{len(jd_text.split()):,}")

    st.markdown("")  # spacer

    # Recommendation box
    st.markdown(
        f"""
        <div class="info-box">
            {recommendation.replace(chr(10), '<br>')}
        </div>
        """,
        unsafe_allow_html=True,
    )
