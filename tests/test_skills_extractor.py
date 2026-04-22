"""Deterministic tests for the fallback extractor + canonicalization.
The OpenAI path is not tested here — we verify the code works end-to-end without a key.
"""
from __future__ import annotations

from pathlib import Path

from src.config import Settings
from src.job_data import load_taxonomy
from src.skills_extractor import canonicalize, extract_skills

FIXTURES = Path(__file__).parent / "fixtures"


def _no_openai_settings() -> Settings:
    return Settings(openai_api_key=None, openai_model="gpt-4o-mini", use_fallbacks_only=True)


def test_fallback_extractor_finds_expected_skills():
    taxonomy = load_taxonomy()
    text = (FIXTURES / "resume_sample.txt").read_text()
    result = extract_skills(text, taxonomy, _no_openai_settings())
    assert result.source == "fallback"
    expected = {"python", "docker", "kubernetes", "terraform", "aws", "postgresql",
                "fastapi", "react", "typescript", "github-actions", "git", "linux",
                "mentoring", "agile", "javascript", "sql", "mongodb", "express"}
    missing = expected - set(result.skills)
    assert not missing, f"Missing expected skills: {missing}"


def test_canonicalize_maps_aliases_to_canonical():
    taxonomy = load_taxonomy()
    result = canonicalize(["Amazon Web Services", "k8s", "PY", "Next.js"], taxonomy)
    assert result == ["aws", "kubernetes", "python", "nextjs"]


def test_canonicalize_drops_unknown():
    taxonomy = load_taxonomy()
    result = canonicalize(["cobol", "python", "zzzzznotaskill"], taxonomy)
    assert result == ["python"]


def test_canonicalize_short_skill_does_not_fuzzy_match():
    """Pins the SHORT_SKILL_LEN guard: strings at/under the length cap bypass the
    fuzzy matcher. Verified by a 3-char exact hit ('sql') resolving and a 3-char
    non-taxonomy string ('xyz') dropping — where fuzzy matching without the
    guard would risk a false positive against a longer canonical ID.

    Note: this doesn't pin a specific historical collision (the motivating case
    'sql' → 'sqlalchemy' requires 'sqlalchemy' in the taxonomy, which isn't
    present today). The assertion is on the guard's behavior, not the example.
    """
    taxonomy = load_taxonomy()
    # 'sql' is exact in the taxonomy → should resolve.
    assert canonicalize(["sql"], taxonomy) == ["sql"]
    # A similar-looking 3-char non-taxonomy string must NOT fuzzy-match (would fail the short-skill guard).
    assert canonicalize(["xyz"], taxonomy) == []


def test_canonicalize_dedupes_preserving_order():
    taxonomy = load_taxonomy()
    result = canonicalize(["python", "py", "PYTHON", "docker"], taxonomy)
    assert result == ["python", "docker"]


def test_empty_text_returns_empty():
    taxonomy = load_taxonomy()
    assert extract_skills("", taxonomy, _no_openai_settings()).skills == []
    assert extract_skills("   ", taxonomy, _no_openai_settings()).skills == []


def test_evidence_retained_when_fallback_snippet_spans_lines():
    """Regression guard: fallback snippets are built by slicing the resume with a ±40-char
    window and collapsing newlines to spaces. The grounding check must tolerate that
    whitespace rewrite, or multi-line-spanning snippets get dropped and the evidence
    expander shows empty for most skills. Whitespace-normalizing both sides fixes it.
    """
    taxonomy = load_taxonomy()
    text = (FIXTURES / "resume_sample.txt").read_text()
    result = extract_skills(text, taxonomy, _no_openai_settings())
    assert "python" in result.evidence


def test_extract_skills_unions_openai_with_rule_based_for_missed_tokens(monkeypatch):
    """Regression guard: a verbatim resume token (e.g., a one-line 'Kafka' bullet)
    must surface even when OpenAI's extraction skips it. Previously the rule-based
    pass only ran if OpenAI returned nothing; observed in practice that OpenAI
    would emit a long skills list while silently dropping a brand name on its own
    line, so the gap analysis then flagged that skill as missing despite it being
    in the resume. The union ensures the rule-based pass is always a floor.
    """
    import src.skills_extractor as extractor_module

    taxonomy = load_taxonomy()
    text = "Built event pipelines with Kafka and ran services in Python on AWS."
    settings = Settings(openai_api_key="sk-fake", openai_model="gpt-4o-mini", use_fallbacks_only=False)

    # Stub OpenAI to return a partial, kafka-less list.
    monkeypatch.setattr(
        extractor_module,
        "_openai_extract",
        lambda *a, **kw: (
            ["python", "aws"],
            {"python": "ran services in Python", "aws": "on AWS"},
        ),
    )

    result = extract_skills(text, taxonomy, settings)
    assert "kafka" in result.skills, "rule-based pass must catch verbatim 'Kafka'"
    assert "python" in result.skills and "aws" in result.skills
    # OpenAI contributed → source still labels it as openai.
    assert result.source == "openai"


def test_fallback_evidence_recovers_when_openai_snippet_is_fabricated(monkeypatch):
    """If OpenAI's snippet for a skill fails the grounding check (not present in
    the resume), but the rule-based pass found that same skill with a real
    resume slice, the UI must show the fallback snippet — not empty evidence.
    Pre-fix, OpenAI's snippet won the merge via setdefault, then grounding
    dropped it, and the recoverable rule-based snippet was silently lost.
    """
    import src.skills_extractor as extractor_module

    taxonomy = load_taxonomy()
    text = "Built event pipelines with Kafka and ran services in Python on AWS."
    settings = Settings(openai_api_key="sk-fake", openai_model="gpt-4o-mini", use_fallbacks_only=False)

    # OpenAI returns python with a fabricated snippet not present in the resume.
    monkeypatch.setattr(
        extractor_module,
        "_openai_extract",
        lambda *a, **kw: (
            ["python"],
            {"python": "I am an expert Python engineer with 10 years experience"},
        ),
    )

    result = extract_skills(text, taxonomy, settings)
    assert "python" in result.skills
    assert "python" in result.evidence, "fallback snippet should backfill after grounding drop"
    # The retained snippet must be the resume slice, not OpenAI's fabrication.
    assert "10 years experience" not in result.evidence["python"]
    assert "python" in result.evidence["python"].lower()


def test_extract_skills_falls_back_when_openai_returns_none(monkeypatch):
    """When _openai_extract returns None (malformed JSON, API error, etc.), the
    public entrypoint must fall through to the deterministic rule-based path and
    mark the result as 'fallback'. We stub _openai_extract rather than the API
    itself — the OpenAI call is never invoked, so this stays within the
    'deterministic paths only' test policy.
    """
    import src.skills_extractor as extractor_module

    taxonomy = load_taxonomy()
    text = (FIXTURES / "resume_sample.txt").read_text()

    # Settings with a fake key so extract_skills takes the OpenAI branch before stubbing.
    settings = Settings(openai_api_key="sk-fake", openai_model="gpt-4o-mini", use_fallbacks_only=False)

    monkeypatch.setattr(extractor_module, "_openai_extract", lambda *a, **kw: None)

    result = extract_skills(text, taxonomy, settings)
    assert result.source == "fallback"
    # Rule-based extractor should still produce a non-empty canonical list for this resume.
    assert "python" in result.skills
