"""Matcher tests: overlap math, composite bounds, what-if monotonicity."""
from __future__ import annotations

import pandas as pd
import pytest

from src.matcher import (
    _freq_weighted_overlap,
    _required_skill_frequencies,
    missing_skills,
    score,
)


@pytest.fixture
def tiny_jobs_df() -> pd.DataFrame:
    """Hand-built 3-job Backend Engineer dataset. Easy to reason about by hand."""
    return pd.DataFrame([
        {"job_id": "A", "title": "BE", "role_category": "Backend Engineer", "seniority": "Mid",
         "description": "python fastapi postgres service", "required_skills": "python|fastapi|postgresql",
         "nice_to_have_skills": "", "location": "Remote"},
        {"job_id": "B", "title": "BE", "role_category": "Backend Engineer", "seniority": "Mid",
         "description": "python service postgres", "required_skills": "python|postgresql|docker",
         "nice_to_have_skills": "", "location": "Remote"},
        {"job_id": "C", "title": "BE", "role_category": "Backend Engineer", "seniority": "Senior",
         "description": "python postgres service", "required_skills": "python|postgresql|redis|docker",
         "nice_to_have_skills": "", "location": "Remote"},
    ])


def test_skill_frequencies(tiny_jobs_df):
    freq = _required_skill_frequencies(tiny_jobs_df, "Backend Engineer")
    assert freq["python"] == 3
    assert freq["postgresql"] == 3
    assert freq["docker"] == 2
    assert freq["redis"] == 1
    assert freq["fastapi"] == 1
    assert "go" not in freq


def test_overlap_math_fully_covered(tiny_jobs_df):
    freq = _required_skill_frequencies(tiny_jobs_df, "Backend Engineer")
    total = sum(freq.values())
    have = {"python", "fastapi", "postgresql", "docker", "redis"}
    covered = freq["python"] + freq["fastapi"] + freq["postgresql"] + freq["docker"] + freq["redis"]
    assert _freq_weighted_overlap(have, freq) == pytest.approx(covered / total)


def test_overlap_math_empty(tiny_jobs_df):
    freq = _required_skill_frequencies(tiny_jobs_df, "Backend Engineer")
    assert _freq_weighted_overlap(set(), freq) == 0.0


def test_missing_skills_ranked_by_frequency(tiny_jobs_df):
    gaps = missing_skills({"python"}, "Backend Engineer", tiny_jobs_df, limit=10)
    # postgresql (3) beats docker (2) beats redis/fastapi (1).
    assert gaps[0] == ("postgresql", 3)
    assert gaps[1][0] == "docker" and gaps[1][1] == 2


class _FakeClassifier:
    classes = ("Backend Engineer", "Frontend Engineer")

    class _FakeModel:
        classes_ = ("Backend Engineer", "Frontend Engineer")

        def predict_proba(self, X):
            # Return a fixed, deterministic probability independent of input.
            return [[0.7, 0.3]]

    class _FakeVec:
        def transform(self, texts):
            return texts

    def __init__(self):
        self.vectorizer = self._FakeVec()
        self.model = self._FakeModel()


def test_composite_in_unit_range(tiny_jobs_df):
    clf = _FakeClassifier()
    s = score({"python", "postgresql"}, "Backend Engineer", clf, tiny_jobs_df)
    assert 0.0 <= s.composite <= 1.0
    assert 0.0 <= s.skill_overlap_pct <= 1.0
    assert 0.0 <= s.classifier_prob <= 1.0


def test_whatif_monotonicity_adding_required_skill(tiny_jobs_df):
    """Adding a required skill the candidate didn't have must never decrease composite."""
    clf = _FakeClassifier()
    base = score({"python"}, "Backend Engineer", clf, tiny_jobs_df)
    improved = score({"python", "postgresql"}, "Backend Engineer", clf, tiny_jobs_df)
    assert improved.skill_overlap_pct >= base.skill_overlap_pct
    assert improved.composite >= base.composite - 1e-9  # floating-point slack
