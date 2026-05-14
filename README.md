# 🧠 AI Resume Intelligence System

> **Semantic resume ↔ job-description matching powered by NLP embeddings**
>
> A portfolio-quality MVP demonstrating real-world NLP engineering — built for AI/ML internship applications.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?logo=streamlit)
![spaCy](https://img.shields.io/badge/spaCy-3.7%2B-09a3d5?logo=spacy)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

The **AI Resume Intelligence System** is a local, browser-based tool that:

1. Accepts a **resume PDF**
2. Accepts a **job description** (plain text)
3. Runs a full **NLP preprocessing pipeline**
4. Generates **semantic embeddings** (TF-IDF *or* Sentence Transformers)
5. Computes **cosine similarity** between the two documents
6. Returns a **match score**, **matched skills**, **missing skills**, and an **overall recommendation**

The goal is to help job-seekers quickly identify gaps between their resume and a target role — similar to what commercial ATS (Applicant Tracking Systems) do internally.

---

## 🎯 Problem Statement

Job-seekers often apply to roles without knowing how well their resume actually aligns with what the employer is looking for. ATS systems silently reject thousands of applications based on keyword and semantic mismatches — before a human ever reads the resume.

This tool makes that process **transparent and actionable**:
- See your exact match score
- Know which skills are present vs. missing
- Get a plain-English recommendation to improve your resume

---

## ✨ Features

| Feature | Details |
|---|---|
| 📄 PDF Resume Upload | Extracts text from any text-based PDF |
| 💼 Job Description Input | Paste the full JD into a text area |
| 🔤 NLP Preprocessing | Lowercasing → Tokenisation → Stopword removal → Lemmatisation |
| 🔢 Dual Embedding Methods | TF-IDF (fast) or all-MiniLM-L6-v2 (semantic) |
| 📐 Cosine Similarity | Industry-standard vector similarity metric |
| ✅ Matched Skills | Green-pill display of overlapping skills |
| ❌ Missing Skills | Red-pill display of JD skills not in resume |
| 📊 Overall Recommendation | Score interpretation + actionable advice |
| 🎨 Clean Modern UI | Custom CSS, responsive layout, no clutter |

---

## 🏗️ Architecture

```
PDF Resume
    │
    ▼
[parser.py]  ──────────────────────────────────────
PDF → raw text extraction (PyPDF2)                 │
                                                    │
    ▼                                               │
[preprocess.py]                                 Job Description
URL/email removal → lowercase → tokenise →     (plain text)
stopword removal → lemmatise (spaCy)                │
                                                    ▼
    │                                       [preprocess.py]
    └─────────────────────────────────────────────► │
                                                    │
                            ▼               ▼
                        [embeddings.py]
                    TF-IDF  │  Sentence Transformer
                  (sklearn)   (all-MiniLM-L6-v2)
                            │
                            ▼
                      [similarity.py]
                    Cosine Similarity
                            │
                            ▼
                       [analysis.py]
              Skill extraction + gap analysis
                            │
                            ▼
                         [ui.py]
              Score card + pills + recommendation
```

---

## 📁 Folder Structure

```
AI-Resume-Intelligence-System/
│
├── data/                   # Scratch space for intermediate files
├── sample_resumes/         # Example PDFs for testing
├── screenshots/            # UI screenshots for README
│
├── src/
│   ├── app.py              # Main Streamlit entry point
│   ├── parser.py           # PDF text extraction
│   ├── preprocess.py       # NLP cleaning pipeline
│   ├── embeddings.py       # TF-IDF & Sentence Transformer embeddings
│   ├── similarity.py       # Cosine similarity computation
│   ├── analysis.py         # Skill extraction & gap analysis
│   ├── ui.py               # Streamlit UI components
│   └── utils.py            # Shared helper functions
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF Parsing | PyPDF2 |
| NLP Preprocessing | NLTK, spaCy (en_core_web_sm) |
| TF-IDF Embeddings | scikit-learn |
| Semantic Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Similarity | scikit-learn cosine_similarity |
| Data Handling | NumPy, Pandas |
| Language | Python 3.11+ |

---

## ⚙️ Installation

### 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Resume-Intelligence-System.git
cd AI-Resume-Intelligence-System
```

### 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
```

### 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4 — Download the spaCy language model

```bash
python -m spacy download en_core_web_sm
```

> The Sentence Transformer model (`all-MiniLM-L6-v2`, ~80 MB) is downloaded automatically on first use.

---

## 🚀 Usage

```bash
cd src
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### Step-by-step

1. Choose an **embedding method** in the left sidebar (`TF-IDF` or `Sentence Transformer`)
2. Upload your **resume PDF** using the file uploader
3. Paste the **job description** text into the text area
4. Click **🔍 Analyse Resume**
5. View your **match score**, **matched / missing skills**, and **recommendation**

---

## 📸 Screenshots

> *Add screenshots to the `screenshots/` folder and update the paths below.*

| Upload & Input | Results |
|---|---|
| ![Upload](./../screenshots/upload.png) | ![Results](./../screenshots/results.png) |

---

## 📊 Example Output

```
Match Score: 84%   (Excellent)

Matched Skills:
  Python  Machine Learning  NLP  Scikit-Learn  Pandas  NumPy  Git

Missing Skills:
  Docker  AWS  FastAPI  Kubernetes

Overall Analysis:
  🟢 Excellent match! Your resume aligns very strongly with this role (84% similarity).

  You matched 7 required skill(s) (Python, Machine Learning, Nlp, Scikit-Learn, Pandas…).

  You are missing 4 skill(s) listed in the JD: Docker, Aws, Fastapi, Kubernetes.
  Adding these to your resume (if you have them) could boost your match score.

  Tip: Tailor your resume to mirror the exact keywords in the job description.
  ATS systems often filter on keyword frequency before a human ever reads your application.
```

---

## 🔬 NLP Concepts Demonstrated

| Concept | Where |
|---|---|
| Text extraction | `parser.py` → PyPDF2 |
| Tokenisation & lemmatisation | `preprocess.py` → spaCy |
| Stopword removal | `preprocess.py` → NLTK |
| Sparse vector embeddings | `embeddings.py` → TF-IDF |
| Dense semantic embeddings | `embeddings.py` → Sentence Transformers |
| Cosine similarity | `similarity.py` |
| Keyword / entity extraction | `analysis.py` |
| Explainable AI output | `analysis.py` → skill gap report |

---

## 🔮 Future Improvements

| Feature | Description |
|---|---|
| 🤖 LLM Feedback | Use GPT / Claude to generate detailed, personalised resume feedback |
| 📋 ATS Optimisation | Suggest exact phrases to add for higher ATS pass-through rates |
| 🏆 Resume Ranking | Rank multiple resumes against a single JD (batch mode) |
| 🗄️ Vector Search | Store embeddings in Pinecone / Chroma for fast candidate retrieval |
| 📚 RAG Hiring Assistant | Build a retrieval-augmented QA system over a resume database |
| 🌍 Multilingual Support | Support non-English resumes using multilingual-e5-large |
| 📈 Analytics Dashboard | Track match score trends across multiple applications |
| 🔐 User Accounts | Save resume history and JD comparisons (authenticated) |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m 'Add my feature'`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👤 Author

Built as an MVP portfolio project demonstrating NLP engineering skills for AI/ML internship applications.

---

*If this project helped you, please ⭐ the repo!*
