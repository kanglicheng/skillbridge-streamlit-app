"""Skill overlap scoring, composite score, missing-skills ranking, what-if simulation."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .classifier import Classifier, predict_proba
from .config import CLASSIFIER_WEIGHT, OVERLAP_WEIGHT
from .job_data import split_skills


def _role_jobs(jobs_df: pd.DataFrame, target_role: str) -> pd.DataFrame:
    return jobs_df[jobs_df["role_category"] == target_role]


def _required_skill_frequencies(jobs_df: pd.DataFrame, target_role: str) -> Counter:
    """Counter of required skills across this role's jobs (higher = more demanded)."""
    subset = _role_jobs(jobs_df, target_role)
    counter: Counter = Counter()
    for cell in subset["required_skills"]:
        for s in split_skills(cell):
            counter[s] += 1
    return counter


def _freq_weighted_overlap(candidate_skills: set[str], freq: Counter) -> float:
    """Fraction of the role's total 'skill weight' the candidate covers.

    Each required skill is weighted by how often it appears across the role's jobs —
    a rare-but-required skill counts more than a ubiquitous one.
    """
    total_weight = sum(freq.values())
    if total_weight == 0:
        return 0.0
    covered = sum(w for skill, w in freq.items() if skill in candidate_skills)
    return covered / total_weight


@dataclass(frozen=True)
class Score:
    classifier_prob: float
    skill_overlap_pct: float
    composite: float

    def as_dict(self) -> dict[str, float]:
        return {
            "classifier_prob": self.classifier_prob,
            "skill_overlap_pct": self.skill_overlap_pct,
            "composite": self.composite,
        }


def _score_with_freq(
    skills_set: set[str],
    target_role: str,
    clf: Classifier,
    freq: Counter,
    portfolio_text: str,
) -> Score:
    """Internal: score given a precomputed freq Counter. Lets what_if reuse freq
    across the base/simulated pair without recomputing per-role stats twice."""
    probs = predict_proba(clf, skills_set, portfolio_text)
    classifier_prob = float(probs.get(target_role, 0.0))
    overlap = _freq_weighted_overlap(skills_set, freq)
    composite = CLASSIFIER_WEIGHT * classifier_prob + OVERLAP_WEIGHT * overlap
    return Score(classifier_prob=classifier_prob, skill_overlap_pct=overlap, composite=composite)


def score(
    skills: Iterable[str],
    target_role: str,
    clf: Classifier,
    jobs_df: pd.DataFrame,
    portfolio_text: str = "",
) -> Score:
    freq = _required_skill_frequencies(jobs_df, target_role)
    return _score_with_freq(set(skills), target_role, clf, freq, portfolio_text)


def missing_skills(
    skills: Iterable[str],
    target_role: str,
    jobs_df: pd.DataFrame,
    limit: int = 20,
) -> list[tuple[str, int]]:
    """Skills required by this role but not held, ranked by frequency (desc)."""
    have = set(skills)
    freq = _required_skill_frequencies(jobs_df, target_role)
    gaps = [(s, c) for s, c in freq.items() if s not in have]
    gaps.sort(key=lambda x: (-x[1], x[0]))
    return gaps[:limit]


def probabilities_table(clf: Classifier, skills: Iterable[str], portfolio_text: str = "") -> list[tuple[str, float]]:
    """Return class → prob sorted desc — used for the analysis bar chart."""
    probs = predict_proba(clf, skills, portfolio_text)
    return sorted(probs.items(), key=lambda kv: -kv[1])


def what_if(
    skills: Iterable[str],
    added_skills: Iterable[str],
    target_role: str,
    clf: Classifier,
    jobs_df: pd.DataFrame,
    portfolio_text: str = "",
) -> dict:
    freq = _required_skill_frequencies(jobs_df, target_role)
    base_set = set(skills)
    simulated_set = base_set | set(added_skills)
    base = _score_with_freq(base_set, target_role, clf, freq, portfolio_text)
    simulated = _score_with_freq(simulated_set, target_role, clf, freq, portfolio_text)
    base_d, sim_d = base.as_dict(), simulated.as_dict()
    return {
        "base": base_d,
        "simulated": sim_d,
        "delta": {k: sim_d[k] - base_d[k] for k in base_d},
    }
