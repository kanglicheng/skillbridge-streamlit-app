"""Roadmap fallback tests: ordering, resource integrity, weeks math."""
from __future__ import annotations

import math

from src.job_data import load_resources, load_taxonomy
from src.roadmap import ASSUMED_WEEKLY_HOURS, _fallback_roadmap


def test_fallback_orders_by_frequency_desc():
    missing = ["kubernetes", "terraform", "helm"]
    freq = {"kubernetes": 5, "terraform": 10, "helm": 2}
    result = _fallback_roadmap(missing, freq, load_resources(), load_taxonomy())
    assert result.source == "fallback"
    assert result.priority_order[0] == "terraform"
    assert result.priority_order[-1] == "helm"


def test_fallback_returns_only_curated_resources():
    missing = ["python", "aws", "kubernetes"]
    freq = {"python": 10, "aws": 8, "kubernetes": 6}
    resources = load_resources()
    result = _fallback_roadmap(missing, freq, resources, load_taxonomy())
    for skill, res_list in result.resources.items():
        for r in res_list:
            # Every returned resource must exist in the curated file (no invention).
            assert r in resources[skill]


def test_fallback_weeks_matches_resource_hours():
    missing = ["python"]
    freq = {"python": 1}
    resources = load_resources()
    total_hours = sum(int(r["est_hours"]) for r in resources["python"])
    expected_weeks = max(1, math.ceil(total_hours / ASSUMED_WEEKLY_HOURS))
    result = _fallback_roadmap(missing, freq, resources, load_taxonomy())
    assert result.estimated_weeks == expected_weeks


def test_fallback_handles_skill_without_resources():
    # A canonical skill that exists in taxonomy but is not in resources.json is OK.
    missing = ["powerbi"]  # in taxonomy, but may not have curated resources
    freq = {"powerbi": 3}
    result = _fallback_roadmap(missing, freq, load_resources(), load_taxonomy())
    assert "powerbi" in result.priority_order
    # Resources list may be empty, but key should still be present.
    assert "powerbi" in result.resources
