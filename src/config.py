"""Configuration: paths, constants, and the Settings dataclass.

Settings is frozen and passed explicitly (not a module global) so behavior
under Streamlit reruns is predictable and testable.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
JOBS_CSV = DATA_DIR / "jobs.csv"
TAXONOMY_JSON = DATA_DIR / "skills_taxonomy.json"
RESOURCES_JSON = DATA_DIR / "resources.json"

TARGET_ROLES: tuple[str, ...] = (
    "Cloud Engineer",
    "Security Analyst",
    "Backend Engineer",
    "Data Analyst",
    "ML Engineer",
    "Frontend Engineer",
    "DevOps Engineer",
    "Data Engineer",
)

# Composite score weights — overlap is more interpretable and less noisy on a
# 100-row synthetic dataset than the classifier, so it carries more weight.
CLASSIFIER_WEIGHT = 0.4
OVERLAP_WEIGHT = 0.6

CANONICALIZE_FUZZY_THRESHOLD = 88
SHORT_SKILL_LEN = 4  # skills ≤ this many chars must exact-match (avoids sql→sqlalchemy)


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    openai_model: str
    use_fallbacks_only: bool

    @classmethod
    def load(cls, use_fallbacks_only: bool = False) -> "Settings":
        return cls(
            openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
            openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            use_fallbacks_only=use_fallbacks_only,
        )

    @property
    def openai_available(self) -> bool:
        return bool(self.openai_api_key) and not self.use_fallbacks_only


def configure_logging() -> logging.Logger:
    """Set up a module-level logger suitable for Streamlit's debug expander."""
    logger = logging.getLogger("skillbridge")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    return logger
