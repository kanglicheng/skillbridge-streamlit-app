"""Learning roadmap. Resources come from the curated data file; the model only
orders the skills and writes rationale — it never invents URLs.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field

import streamlit as st

from .config import Settings
from .job_data import load_resources, load_taxonomy
from .openai_client import get_client
from .prompts import ROADMAP_SYSTEM, ROADMAP_USER_TEMPLATE

log = logging.getLogger("skillbridge")

ASSUMED_WEEKLY_HOURS = 10


@dataclass(frozen=True)
class Roadmap:
    priority_order: list[str]
    rationale: dict[str, str]
    estimated_weeks: int
    source: str  # "openai" | "fallback"
    resources: dict[str, list[dict]] = field(default_factory=dict)


def _categorize(skill: str, taxonomy: list[dict]) -> str:
    for entry in taxonomy:
        if entry["canonical"] == skill:
            return entry["category"]
    return "other"


def _weeks_from_resources(priority: list[str], resources: dict[str, list[dict]]) -> int:
    total_hours = 0
    for skill in priority:
        for r in resources.get(skill, []):
            total_hours += int(r.get("est_hours", 0))
    return max(1, math.ceil(total_hours / ASSUMED_WEEKLY_HOURS))


def _fallback_roadmap(
    missing: list[str],
    missing_freq: dict[str, int],
    resources: dict[str, list[dict]],
    taxonomy: list[dict],
) -> Roadmap:
    ordered = sorted(missing, key=lambda s: (-missing_freq.get(s, 0), _categorize(s, taxonomy), s))
    rationale = {s: f"Appears in {missing_freq.get(s, 0)} job(s) for this role." for s in ordered}
    return Roadmap(
        priority_order=ordered,
        rationale=rationale,
        estimated_weeks=_weeks_from_resources(ordered, resources),
        resources={s: resources.get(s, []) for s in ordered},
        source="fallback",
    )


def _openai_roadmap(
    target_role: str,
    current_skills: tuple[str, ...],
    missing_skills_ordered: tuple[str, ...],
    portfolio_text: str,
    resources: dict[str, list[dict]],
    settings: Settings,
) -> Roadmap | None:
    client = get_client(settings)
    if client is None:
        return None
    try:
        resp = client.chat.completions.create(
            model=settings.openai_model,
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": ROADMAP_SYSTEM},
                {
                    "role": "user",
                    "content": ROADMAP_USER_TEMPLATE.format(
                        target_role=target_role,
                        current_skills=", ".join(current_skills) or "(none)",
                        missing_skills=", ".join(missing_skills_ordered),
                        portfolio_text=(portfolio_text or "")[:2000],
                    ),
                },
            ],
        )
        payload = json.loads(resp.choices[0].message.content or "{}")
        priority = payload.get("priority_order") or []
        rationale = payload.get("rationale") or {}
        weeks = payload.get("estimated_weeks")
        if not isinstance(priority, list) or not isinstance(rationale, dict):
            log.warning("OpenAI roadmap returned unexpected shape; falling back")
            return None
        # Constrain to the skills we asked about — guard against hallucinated additions.
        allowed = set(missing_skills_ordered)
        priority_clean = [s for s in priority if s in allowed]
        for s in missing_skills_ordered:
            if s not in priority_clean:
                priority_clean.append(s)
        rationale_clean = {s: str(rationale.get(s, "")) for s in priority_clean}
        if not isinstance(weeks, int) or weeks <= 0:
            weeks = _weeks_from_resources(priority_clean, resources)
        return Roadmap(
            priority_order=priority_clean,
            rationale=rationale_clean,
            estimated_weeks=int(weeks),
            resources={s: resources.get(s, []) for s in priority_clean},
            source="openai",
        )
    except Exception:
        log.exception("OpenAI roadmap failed; falling back")
        return None


# Two caches, one per path: a transient OpenAI failure can't poison the
# "OpenAI succeeded" line. TTL on the OpenAI cache lets a cached None
# (st.cache_data caches None too) self-heal. Cache keys exclude the API
# key — we key on model + inputs.
_OPENAI_CACHE_TTL_SECONDS = 300


@st.cache_data(show_spinner=False, ttl=_OPENAI_CACHE_TTL_SECONDS)
def _cached_openai_roadmap(
    target_role: str,
    current: tuple[str, ...],
    missing: tuple[str, ...],
    portfolio_text: str,
    openai_model: str,
) -> Roadmap | None:
    """OpenAI path. Returns None if OpenAI is unavailable or the call fails.

    Nones are cached; TTL self-heals. API key is loaded inside and
    deliberately excluded from the cache key.
    """
    live = Settings.load(use_fallbacks_only=False)
    if not live.openai_api_key:
        return None
    call_settings = Settings(
        openai_api_key=live.openai_api_key,
        openai_model=openai_model,
        use_fallbacks_only=False,
    )
    resources = load_resources()
    return _openai_roadmap(target_role, current, missing, portfolio_text, resources, call_settings)


@st.cache_data(show_spinner=False)
def _cached_fallback_roadmap(
    missing: tuple[str, ...],
    missing_freq_items: tuple[tuple[str, int], ...],
) -> Roadmap:
    """Deterministic fallback path. No TTL — pure function of inputs."""
    resources = load_resources()
    taxonomy = load_taxonomy()
    return _fallback_roadmap(list(missing), dict(missing_freq_items), resources, taxonomy)


def generate_roadmap(
    target_role: str,
    current_skills: list[str],
    missing_skills_ranked: list[tuple[str, int]],
    portfolio_text: str,
    settings: Settings,
) -> Roadmap:
    """Public entry. Tries OpenAI when available, falls back otherwise. Each
    path has an independent cache so failures in one don't contaminate the other.
    """
    current = tuple(sorted(current_skills))
    missing = tuple(s for s, _ in missing_skills_ranked)
    missing_freq_items = tuple(missing_skills_ranked)

    if settings.openai_available:
        result = _cached_openai_roadmap(
            target_role=target_role,
            current=current,
            missing=missing,
            portfolio_text=portfolio_text or "",
            openai_model=settings.openai_model,
        )
        if result is not None:
            return result
        # None means OpenAI was unavailable or the call failed. The cached None
        # self-heals via TTL; fall through to the fallback cache for this call.

    return _cached_fallback_roadmap(missing=missing, missing_freq_items=missing_freq_items)
