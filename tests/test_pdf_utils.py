"""Covers both error branches of pdf_utils.extract_text with hand-crafted PDFs.

Avoids a reportlab/fpdf dev dependency by constructing minimal valid PDF bytes
inline — one with a text-drawing content stream, one with an empty-resources
page and no content stream at all. The corrupt-bytes blob exercises the
pdfplumber open failure path.
"""
from __future__ import annotations

import io

import pytest

from src.pdf_utils import EmptyPDFError, PDFReadError, extract_text


def _build_pdf(parts: list[bytes]) -> bytes:
    """Assemble an N-object PDF with a correct xref from a list of object bytes.

    parts[0] must be the header; parts[1:] must be the N numbered objects in
    order. This computes byte offsets from the concatenated body so the xref
    table is always correct, regardless of minor edits.
    """
    offsets = [0]
    for p in parts:
        offsets.append(offsets[-1] + len(p))
    body = b"".join(parts)
    xref_pos = offsets[-1]
    n_objs = len(parts) - 1
    xref = f"xref\n0 {n_objs + 1}\n".encode() + b"0000000000 65535 f \n"
    for i in range(1, n_objs + 1):
        xref += f"{offsets[i]:010d} 00000 n \n".encode()
    xref += f"trailer\n<</Size {n_objs + 1}/Root 1 0 R>>\nstartxref\n{xref_pos}\n%%EOF\n".encode()
    return body + xref


def _blank_pdf() -> bytes:
    """Valid PDF, one page, no content stream — opens cleanly, yields no text."""
    return _build_pdf([
        b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n",
        b"1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n",
        b"2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n",
        b"3 0 obj\n<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Resources<<>>>>\nendobj\n",
    ])


def _text_pdf(text: str) -> bytes:
    """Valid PDF with a single Helvetica line — extraction should return `text`."""
    stream = f"BT /F1 12 Tf 50 750 Td ({text}) Tj ET".encode("latin-1")
    content_obj = (
        b"4 0 obj\n<</Length " + str(len(stream)).encode() + b">>\nstream\n"
        + stream + b"\nendstream\nendobj\n"
    )
    return _build_pdf([
        b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n",
        b"1 0 obj\n<</Type/Catalog/Pages 2 0 R>>\nendobj\n",
        b"2 0 obj\n<</Type/Pages/Kids[3 0 R]/Count 1>>\nendobj\n",
        b"3 0 obj\n<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Resources<</Font<</F1 5 0 R>>>>/Contents 4 0 R>>\nendobj\n",
        content_obj,
        b"5 0 obj\n<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>\nendobj\n",
    ])


def test_extract_text_happy_path():
    """A well-formed PDF with ample text returns the extracted string."""
    payload = "Python Docker Kubernetes AWS Terraform Senior Engineer"
    result = extract_text(io.BytesIO(_text_pdf(payload)))
    assert payload in result


def test_extract_text_empty_pdf_raises_empty_error():
    """Valid PDF that produces <MIN_TEXT_LEN chars raises EmptyPDFError (scanned-PDF case)."""
    with pytest.raises(EmptyPDFError):
        extract_text(io.BytesIO(_blank_pdf()))


def test_extract_text_empty_pdf_error_is_pdf_read_error():
    """EmptyPDFError is a PDFReadError subclass — callers can catch one base."""
    with pytest.raises(PDFReadError):
        extract_text(io.BytesIO(_blank_pdf()))


def test_extract_text_corrupt_bytes_raises_pdf_read_error():
    """Non-PDF bytes fail at pdfplumber.open and surface as PDFReadError."""
    with pytest.raises(PDFReadError):
        extract_text(io.BytesIO(b"definitely not a pdf"))
