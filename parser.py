"""
parser.py
==========
Handles PDF text extraction using PyPDF2.
Safely reads each page and concatenates the text.
"""

import io
import PyPDF2


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Extract plain text from a Streamlit UploadedFile object (PDF).

    Parameters
    ----------
    uploaded_file : streamlit.runtime.uploaded_file_manager.UploadedFile
        The PDF file uploaded by the user.

    Returns
    -------
    str
        Concatenated text from all pages, or empty string on failure.
    """
    text = ""
    try:
        # Read bytes from the uploaded file
        pdf_bytes = uploaded_file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))

        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except Exception as page_err:
                # Skip unreadable pages gracefully
                print(f"[parser] Could not read page {page_num}: {page_err}")
                continue

    except Exception as e:
        print(f"[parser] PDF extraction failed: {e}")
        return ""

    return text.strip()


def extract_text_from_string(text: str) -> str:
    """
    Passthrough helper for plain-text inputs (job description, etc.).

    Parameters
    ----------
    text : str
        Raw input string.

    Returns
    -------
    str
        Stripped text.
    """
    return text.strip() if text else ""
