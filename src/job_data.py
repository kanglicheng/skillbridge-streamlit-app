"""Load and validate jobs.csv and the skills taxonomy."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import JOBS_CSV, RESOURCES_JSON, TAXONOMY_JSON

REQUIRED_COLUMNS = {
    "job_id",
    "title",
    "role_category",
    "seniority",
    "description",
    "required_skills",
    "nice_to_have_skills",
    "location",
}


def load_jobs(path: Path = JOBS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"jobs.csv missing columns: {sorted(missing)}")
    df["required_skills"] = df["required_skills"].fillna("")
    df["nice_to_have_skills"] = df["nice_to_have_skills"].fillna("")
    df["description"] = df["description"].fillna("")
    return df


def split_skills(cell: str) -> list[str]:
    return [s.strip() for s in (cell or "").split("|") if s.strip()]


@lru_cache(maxsize=1)
def load_taxonomy(path: Path = TAXONOMY_JSON) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data["skills"]


@lru_cache(maxsize=1)
def load_resources(path: Path = RESOURCES_JSON) -> dict[str, list[dict]]:
    with open(path) as f:
        raw = json.load(f)
    return raw
