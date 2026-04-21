"""Render-time safety helpers for untrusted text.

Anything that originates from a resume, portfolio, or LLM response and is
displayed via st.markdown must pass through escape_markdown() first — otherwise
markdown link syntax (`[label](url)`) in the input renders as a real clickable
link, which is a phishing vector even though Streamlit blocks raw HTML.
"""
from __future__ import annotations

import re

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~=])")


def escape_markdown(text: str) -> str:
    """Backslash-escape every Markdown/HTML metacharacter.

    Use at the render boundary for any string that originated from untrusted
    input (resume text, portfolio text, LLM output). Newlines collapse to
    spaces so a multi-line snippet can't break out of an inline context.
    """
    if not text:
        return ""
    return _MD_SPECIAL.sub(r"\\\1", str(text)).replace("\n", " ").replace("\r", " ")
