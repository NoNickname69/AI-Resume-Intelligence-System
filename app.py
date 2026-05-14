"""
AI Resume Intelligence System
==============================
Main Streamlit application — orchestrates the full NLP pipeline.
"""

import streamlit as st

# Local module imports
from parser import extract_text_from_pdf
from preprocess import preprocess_text
from embeddings import get_tfidf_embeddings, get_transformer_embeddings
from similarity import compute_cosine_similarity
from analysis import (
    extract_skills,
    get_matched_skills,
    get_missing_skills,
    generate_recommendation,
)
from ui import (
    render_header,
    render_score_card,
    render_skills_section,
    render_analysis_summary,
    render_sidebar,
)
from utils import validate_inputs, log_info

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Intelligence System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ──────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* General body */
        body { font-family: 'Segoe UI', sans-serif; }

        /* Main container padding */
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }

        /* Score card */
        .score-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 16px;
            padding: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.35);
        }
        .score-number {
            font-size: 4rem;
            font-weight: 800;
            line-height: 1;
        }
        .score-label { font-size: 1.1rem; opacity: 0.9; margin-top: 0.4rem; }

        /* Skill pills */
        .skill-pill-match {
            display: inline-block;
            background: #d1fae5;
            color: #065f46;
            border-radius: 999px;
            padding: 0.3rem 0.9rem;
            margin: 0.25rem;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .skill-pill-missing {
            display: inline-block;
            background: #fee2e2;
            color: #991b1b;
            border-radius: 999px;
            padding: 0.3rem 0.9rem;
            margin: 0.25rem;
            font-size: 0.85rem;
            font-weight: 600;
        }

        /* Section header */
        .section-header {
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
            color: #1e293b;
        }

        /* Info box */
        .info-box {
            background: #f0f9ff;
            border-left: 4px solid #0ea5e9;
            border-radius: 6px;
            padding: 1rem 1.25rem;
            margin-bottom: 1rem;
            font-size: 0.92rem;
            color: #0c4a6e;
        }

        /* Divider */
        hr { border: none; border-top: 1px solid #e2e8f0; margin: 1.5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    """Entry point — renders UI and runs the NLP pipeline on user action."""

    # ── Sidebar (method selector + about) ─────────────────────────────────────
    embedding_method = render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────────
    render_header()

    st.markdown("---")

    # ── Input columns ─────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("#### 📄 Upload Resume (PDF)")
        uploaded_file = st.file_uploader(
            label="Drag & drop or click to browse",
            type=["pdf"],
            help="Only PDF files are supported in this MVP.",
        )

    with col_right:
        st.markdown("#### 💼 Paste Job Description")
        job_description = st.text_area(
            label="Job Description",
            placeholder="Paste the full job description here…",
            height=220,
            label_visibility="collapsed",
        )

    st.markdown("---")

    # ── Analyse button ─────────────────────────────────────────────────────────
    analyse_btn = st.button(
        "🔍  Analyse Resume",
        type="primary",
        use_container_width=True,
    )

    # ── Pipeline ───────────────────────────────────────────────────────────────
    if analyse_btn:
        # 1. Validate
        is_valid, error_msg = validate_inputs(uploaded_file, job_description)
        if not is_valid:
            st.error(f"⚠️  {error_msg}")
            return

        with st.spinner("Running NLP pipeline… this may take a moment on first run."):

            # 2. Extract resume text from PDF
            log_info("Extracting text from PDF…")
            resume_text = extract_text_from_pdf(uploaded_file)
            if not resume_text:
                st.error("Could not extract text from the PDF. Try a text-based PDF.")
                return

            # 3. Preprocess both texts
            log_info("Preprocessing texts…")
            clean_resume = preprocess_text(resume_text)
            clean_jd = preprocess_text(job_description)

            # 4. Generate embeddings & compute similarity
            log_info(f"Generating embeddings using: {embedding_method}")
            if embedding_method == "TF-IDF":
                resume_vec, jd_vec = get_tfidf_embeddings(clean_resume, clean_jd)
            else:
                resume_vec, jd_vec = get_transformer_embeddings(clean_resume, clean_jd)

            # 5. Cosine similarity → match score
            score = compute_cosine_similarity(resume_vec, jd_vec)
            score_pct = round(score * 100, 1)

            # 6. Skill analysis
            log_info("Analysing skill overlap…")
            resume_skills = extract_skills(resume_text)
            jd_skills = extract_skills(job_description)
            matched = get_matched_skills(resume_skills, jd_skills)
            missing = get_missing_skills(resume_skills, jd_skills)
            recommendation = generate_recommendation(score_pct, matched, missing)

        # ── Results ────────────────────────────────────────────────────────────
        st.success("✅  Analysis complete!")
        st.markdown("---")

        # Score card + method badge
        render_score_card(score_pct, embedding_method)

        st.markdown("---")

        # Skills columns
        skill_col1, skill_col2 = st.columns(2, gap="large")
        with skill_col1:
            render_skills_section("✅ Matched Skills", matched, kind="match")
        with skill_col2:
            render_skills_section("❌ Missing Skills", missing, kind="missing")

        st.markdown("---")

        # Overall analysis
        render_analysis_summary(recommendation, score_pct, resume_text, job_description)

        # Raw text expanders (debug / transparency)
        with st.expander("🔎 View Extracted Resume Text"):
            st.text_area("Resume Text", resume_text, height=250, disabled=True)
        with st.expander("🔎 View Preprocessed Resume Text"):
            st.text_area("Cleaned Resume", clean_resume, height=150, disabled=True)


if __name__ == "__main__":
    main()
