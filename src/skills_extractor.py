"""Skill extraction with canonicalization. Both OpenAI and fallback paths emit
canonical IDs — the rest of the app never sees raw strings.

Security note: resume text is untrusted input. Canonicalizing LLM output to a
known-ID allowlist is the containment boundary — any prompt-injection that
coerces the model into emitting unexpected strings is dropped at canonicalize().
See docs/ARCHITECTURE.md §8.3 for the full prompt-injection threat model.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

from .config import CANONICALIZE_FUZZY_THRESHOLD, SHORT_SKILL_LEN, Settings
from .openai_client import get_client
from .prompts import SKILLS_EXTRACTION_SYSTEM, SKILLS_EXTRACTION_USER_TEMPLATE

log = logging.getLogger("skillbridge")


@dataclass(frozen=True)
class ExtractionResult:
    skills: list[str]
    source: str  # "openai" | "fallback"
    evidence: dict[str, str] = field(default_factory=dict)


def _build_alias_index(taxonomy: list[dict]) -> dict[str, str]:
    """Flat dict: lowered canonical + every lowered alias → canonical ID."""
    index: dict[str, str] = {}
    for entry in taxonomy:
        canonical = entry["canonical"]
        index[canonical.lower()] = canonical
        for alias in entry.get("aliases", []):
            index[alias.lower()] = canonical
    return index


def canonicalize(raw_skills: list[str], taxonomy: list[dict]) -> list[str]:
    """Map arbitrary strings to canonical IDs. Exact → alias → fuzzy (with short-skill guard).

    Unmatched skills are dropped and logged. Deduplicated preserving first-seen order.
    """
    index = _build_alias_index(taxonomy)
    choices = list(index.keys())
    seen: set[str] = set()
    out: list[str] = []
    for raw in raw_skills:
        if not raw:
            continue
        key = raw.strip().lower()
        canonical: str | None = None
        if key in index:
            canonical = index[key]
        elif len(key) > SHORT_SKILL_LEN:
            match = process.extractOne(key, choices, scorer=fuzz.token_set_ratio)
            if match and match[1] >= CANONICALIZE_FUZZY_THRESHOLD:
                canonical = index[match[0]]
        if canonical is None:
            log.info("Dropping unmatched skill: %r", raw)
            continue
        if canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


def _fallback_extract(text: str, taxonomy: list[dict]) -> tuple[list[str], dict[str, str]]:
    """Rule-based: scan resume text for any alias or canonical, collect snippets."""
    index = _build_alias_index(taxonomy)
    text_lower = text.lower()
    found: dict[str, str] = {}
    for phrase, canonical in index.items():
        # Word-boundary match for multi-char skills; short skills need exact-token hit too.
        pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
        match = re.search(pattern, text_lower)
        if match and canonical not in found:
            start = max(0, match.start() - 40)
            end = min(len(text), match.end() + 40)
            found[canonical] = text[start:end].strip().replace("\n", " ")
    return list(found.keys()), found


def _openai_extract(text: str, taxonomy: list[dict], settings: Settings) -> tuple[list[str], dict[str, str]] | None:
    client = get_client(settings)
    if client is None:
        return None
    hints = ", ".join(sorted({e["canonical"] for e in taxonomy}))
    try:
        resp = client.chat.completions.create(
            model=settings.openai_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SKILLS_EXTRACTION_SYSTEM},
                {
                    "role": "user",
                    "content": SKILLS_EXTRACTION_USER_TEMPLATE.format(hints=hints, resume_text=text[:8000]),
                },
            ],
        )
        payload = json.loads(resp.choices[0].message.content or "{}")
        raw_skills = payload.get("skills") or []
        evidence = payload.get("evidence") or {}
        if not isinstance(raw_skills, list) or not isinstance(evidence, dict):
            log.warning("OpenAI returned unexpected JSON shape; falling back")
            return None
        return [str(s) for s in raw_skills], {str(k): str(v) for k, v in evidence.items()}
    except Exception:
        log.exception("OpenAI skill extraction failed; falling back")
        return None


def extract_skills(text: str, taxonomy: list[dict], settings: Settings) -> ExtractionResult:
    """Try OpenAI; fall back to rule-based. Always returns canonical IDs."""
    if not text or not text.strip():
        return ExtractionResult(skills=[], source="fallback", evidence={})

    source = "fallback"
    raw_skills: list[str] = []
    raw_evidence: dict[str, str] = {}

    oai = _openai_extract(text, taxonomy, settings)
    if oai is not None:
        raw_skills, raw_evidence = oai
        source = "openai"

    if not raw_skills:
        raw_skills, raw_evidence = _fallback_extract(text, taxonomy)
        source = "fallback"

    canonical_skills = canonicalize(raw_skills, taxonomy)

    # Map evidence keys through canonicalization too (OpenAI evidence may use aliases).
    # Snippets are only retained if they literally appear in the resume — the model is
    # *claiming* these come from the text, but the prompt doesn't constrain it to copy
    # verbatim. Dropping non-present snippets prevents surfacing fabricated evidence in
    # the UI (belt-and-suspenders alongside escape_markdown at render time).
    canon_evidence: dict[str, str] = {}
    if raw_evidence:
        index = _build_alias_index(taxonomy)
        text_lower = text.lower()
        for key, snippet in raw_evidence.items():
            canon = index.get(key.strip().lower())
            if not canon or canon not in canonical_skills or canon in canon_evidence:
                continue
            if not snippet or snippet.strip().lower() not in text_lower:
                log.info("Dropping evidence snippet for %r: not present in resume text", canon)
                continue
            canon_evidence[canon] = snippet

    return ExtractionResult(skills=canonical_skills, source=source, evidence=canon_evidence)
