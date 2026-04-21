"""Thin OpenAI client wrapper. Returns None when unavailable so callers fall back cleanly."""
from __future__ import annotations

import logging

from openai import OpenAI

from .config import Settings

log = logging.getLogger("skillbridge")

_logged_reason: str | None = None


def get_client(settings: Settings) -> OpenAI | None:
    """Return an OpenAI client or None. Logs the reason once per process."""
    global _logged_reason
    if settings.use_fallbacks_only:
        if _logged_reason != "toggle":
            log.info("OpenAI disabled by 'Use fallbacks only' toggle")
            _logged_reason = "toggle"
        return None
    if not settings.openai_api_key:
        if _logged_reason != "no_key":
            log.info("OpenAI disabled: no OPENAI_API_KEY in env")
            _logged_reason = "no_key"
        return None
    _logged_reason = None
    return OpenAI(api_key=settings.openai_api_key)
