"""Cross-file data-consistency checks.

These tests catch drift between jobs.csv, skills_taxonomy.json, and resources.json.
A skill referenced in jobs.csv that isn't in the taxonomy would silently fail
canonicalization, so resumes mentioning it get no credit and gap analysis never
surfaces it. Pin this invariant.
"""
from __future__ import annotations

from src.config import TARGET_ROLES
from src.job_data import load_jobs, load_resources, load_taxonomy, split_skills


def _taxonomy_token_set() -> set[str]:
    """Every canonical + every alias, lowercased. Exactly what canonicalize() accepts."""
    tokens: set[str] = set()
    for entry in load_taxonomy():
        tokens.add(entry["canonical"].lower())
        for alias in entry.get("aliases", []):
            tokens.add(alias.lower())
    return tokens


def test_all_csv_skills_are_in_taxonomy():
    jobs = load_jobs()
    tokens = _taxonomy_token_set()
    unknown: set[str] = set()
    for col in ("required_skills", "nice_to_have_skills"):
        for cell in jobs[col]:
            for s in split_skills(cell):
                if s.strip().lower() not in tokens:
                    unknown.add(s)
    assert not unknown, (
        f"jobs.csv references skills missing from skills_taxonomy.json: {sorted(unknown)}. "
        "Add them to the taxonomy or remove them from the CSV."
    )


def test_all_resource_keys_are_canonical_skills():
    canonicals = {e["canonical"] for e in load_taxonomy()}
    orphaned = [k for k in load_resources() if k not in canonicals]
    assert not orphaned, (
        f"resources.json references non-canonical skills: {orphaned}. "
        "Roadmap lookups for these keys would silently return empty lists."
    )


def test_all_csv_role_categories_are_target_roles():
    jobs = load_jobs()
    csv_roles = set(jobs["role_category"].unique())
    unknown = csv_roles - set(TARGET_ROLES)
    assert not unknown, (
        f"jobs.csv contains role_category values not in TARGET_ROLES: {sorted(unknown)}"
    )


def test_job_ids_are_unique():
    jobs = load_jobs()
    dupes = jobs["job_id"][jobs["job_id"].duplicated()].tolist()
    assert not dupes, f"Duplicate job_ids in jobs.csv: {dupes}"


def test_canonicals_have_no_whitespace():
    """Canonical skill IDs must be single tokens (no spaces).

    The classifier's TfidfVectorizer uses `token_pattern=r"(?u)\\b\\w[\\w-]+\\b"`,
    which treats a whitespace-containing canonical as multiple tokens — the
    multi-word signal would silently split at training time. Hyphenated IDs
    like `incident-response` are fine; spaced IDs would not be.
    """
    spaced = [e["canonical"] for e in load_taxonomy() if any(c.isspace() for c in e["canonical"])]
    assert not spaced, (
        f"Canonical skill IDs contain whitespace: {spaced}. "
        "Use hyphens instead so the classifier treats each canonical as one token."
    )


def test_no_ambiguous_aliases():
    """Every token (canonical or alias, lowercased) must resolve to exactly one canonical.

    _build_alias_index silently overwrites on collision, so an ambiguous alias
    like "tf" (terraform vs tensorflow) would misclassify whichever entry
    happens to come later in the JSON.
    """
    token_to_canonicals: dict[str, set[str]] = {}
    for entry in load_taxonomy():
        canonical = entry["canonical"]
        for token in [canonical.lower(), *(a.lower() for a in entry.get("aliases", []))]:
            token_to_canonicals.setdefault(token, set()).add(canonical)
    collisions = {t: sorted(cs) for t, cs in token_to_canonicals.items() if len(cs) > 1}
    assert not collisions, (
        f"Ambiguous taxonomy tokens resolve to multiple canonicals: {collisions}. "
        "Remove the alias from one or both entries."
    )
