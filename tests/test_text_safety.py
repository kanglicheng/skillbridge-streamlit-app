"""Tests for the markdown render-time escape — closes the resume-as-XSS vector."""
from __future__ import annotations

from src.text_safety import escape_markdown


def test_escape_neutralises_markdown_link():
    # Phishing payload a candidate could put in their resume.
    payload = "see [my portfolio](https://evil.example/phish)"
    out = escape_markdown(payload)
    assert "[" not in out.replace("\\[", "")
    assert "](" not in out.replace("\\]\\(", "")


def test_escape_neutralises_image_syntax():
    payload = "![pwn](https://evil.example/x.png)"
    out = escape_markdown(payload)
    assert out.startswith("\\!")


def test_escape_neutralises_emphasis_and_code():
    payload = "*bold* _italic_ `code` ~strike~"
    out = escape_markdown(payload)
    for ch in ("*", "_", "`", "~"):
        # Every special char must be backslash-prefixed.
        assert f"\\{ch}" in out


def test_escape_collapses_newlines():
    payload = "line1\nline2\rline3"
    out = escape_markdown(payload)
    assert "\n" not in out
    assert "\r" not in out


def test_escape_handles_empty_and_none_like():
    assert escape_markdown("") == ""
    assert escape_markdown(None) == ""  # type: ignore[arg-type]


def test_escape_preserves_plain_text():
    payload = "Built APIs in Python and deployed on AWS"
    out = escape_markdown(payload)
    # No metachars in input → output should equal input.
    assert out == payload
