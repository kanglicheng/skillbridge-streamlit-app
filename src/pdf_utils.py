"""PDF text extraction with empty-text detection for scanned PDFs."""
from __future__ import annotations

import logging
from typing import BinaryIO

import pdfplumber

log = logging.getLogger("skillbridge")

MIN_TEXT_LEN = 40  # Below this, treat as "no text extracted" (likely a scanned PDF).


class PDFReadError(ValueError):
    """Base class for PDF extraction failures. Subclasses let callers
    distinguish causes (corrupted vs. empty) without forcing them to."""


class EmptyPDFError(PDFReadError):
    """Raised when a PDF opens cleanly but yields too little text (likely scanned)."""


def extract_text(file: BinaryIO) -> str:
    """Extract concatenated text from a PDF file-like.

    Raises:
        PDFReadError: if the PDF can't be opened (corrupted/encrypted/malformed).
        EmptyPDFError: if the PDF opens but yields too little text (likely scanned).
    """
    chunks: list[str] = []
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    chunks.append(text)
    except Exception as e:
        log.warning("pdfplumber failed to read PDF: %s: %s", type(e).__name__, e)
        raise PDFReadError(
            "Could not read this PDF — it may be corrupted or password-protected. "
            "Please paste your text instead."
        ) from e
    combined = "\n".join(chunks).strip()
    if len(combined) < MIN_TEXT_LEN:
        log.warning("PDF extraction produced <%d chars; likely scanned/image PDF", MIN_TEXT_LEN)
        raise EmptyPDFError(
            "Could not extract text from this PDF. It may be a scanned image — "
            "please paste your resume text manually."
        )
    return combined
