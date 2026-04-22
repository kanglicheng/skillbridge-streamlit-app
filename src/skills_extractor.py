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
    """Run OpenAI (when available) and the rule-based scanner; union both.

    The rule-based pass is a floor — OpenAI sometimes drops a verbatim token
    (e.g. a one-line `Kafka` bullet) even when it returns 20+ others. Both
    paths feed `canonicalize()` so the allowlist invariant holds.

    `source` is "openai" only when OpenAI contributed at least one skill.
    """
    if not text or not text.strip():
        return ExtractionResult(skills=[], source="fallback", evidence={})

    oai = _openai_extract(text, taxonomy, settings)
    oai_skills, oai_evidence = oai if oai is not None else ([], {})

    fb_skills, fb_evidence = _fallback_extract(text, taxonomy)

    raw_skills: list[str] = list(oai_skills)
    seen = set(raw_skills)
    for s in fb_skills:
        if s not in seen:
            raw_skills.append(s)
            seen.add(s)

    raw_evidence: dict[str, str] = dict(oai_evidence)
    for s, snippet in fb_evidence.items():
        raw_evidence.setdefault(s, snippet)

    source = "openai" if oai is not None and oai_skills else "fallback"

    canonical_skills = canonicalize(raw_skills, taxonomy)

    # Grounding check: drop snippets that don't appear in the resume (OpenAI can
    # fabricate). Whitespace-normalized on both sides so fallback slices that
    # collapsed newlines still match.
    canon_evidence: dict[str, str] = {}
    if raw_evidence:
        index = _build_alias_index(taxonomy)
        text_flat = re.sub(r"\s+", " ", text.lower())
        for key, snippet in raw_evidence.items():
            canon = index.get(key.strip().lower())
            if not canon or canon not in canonical_skills or canon in canon_evidence:
                continue
            flat = re.sub(r"\s+", " ", (snippet or "").lower()).strip()
            if not flat or flat not in text_flat:
                log.info("Dropping evidence snippet for %r: not present in resume text", canon)
                continue
            canon_evidence[canon] = snippet

    # Backfill from fb_evidence after grounding — fallback snippets are always
    # grounded by construction (sliced from the resume).
    for canon, snippet in fb_evidence.items():
        if canon in canonical_skills and canon not in canon_evidence:
            canon_evidence[canon] = snippet

    return ExtractionResult(skills=canonical_skills, source=source, evidence=canon_evidence)
