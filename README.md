# SkillBridge Career Navigator

A Streamlit prototype that helps students and early-career candidates see how
their skills align with a target role. End-to-end flow:

**Resume upload → skill extraction → gap analysis → role-fit prediction → what-if simulation → learning roadmap.**

OpenAI is used where it adds value (skill extraction, roadmap rationale) but
every AI step has a deterministic fallback, so the app is fully usable without
an API key.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # optional — add your OPENAI_API_KEY if you have one
```

## Run

```bash
streamlit run app.py
```

## Test

```bash
pytest
```

Tests cover deterministic paths only (fallback extractor, canonicalization,
matcher math, what-if monotonicity, roadmap ordering) — no OpenAI mocking.

## How it works

- **Taxonomy-first canonicalization** — resume skills from the OpenAI
  extractor and the rule-based scanner are unioned and mapped to canonical IDs
  before scoring. The rest of the app never sees raw strings, so output is
  consistent regardless of which extractor contributed.
- **Composite score** — `0.4 × classifier_prob + 0.6 × frequency-weighted
  skill overlap`. Overlap carries more weight because it's more interpretable
  and less noisy on a 100-row synthetic dataset than a TF-IDF + LogReg
  classifier.
- **Gap analysis uses `required_skills` only.** `nice_to_have_skills` are
  deliberately excluded from both the overlap denominator and the missing-skills
  list — gaps reflect what's actually required for the role, not the superset of
  everything nice to have. The classifier also trains on `description +
  required_skills` only, so nice-to-haves are carried in the dataset for
  extensibility but don't currently influence scoring or gap detection.
- **Roadmap** — the model never invents URLs. Missing skills are looked up in
  a curated `data/resources.json`; OpenAI's only job is to order the skills
  and write one sentence of rationale each. Fallback path uses frequency
  ranking with the same resource data.
- **Caching** — the classifier is built once per CSV mtime; the roadmap is
  cached by content so What-If interactions don't re-hit OpenAI on every
  multiselect toggle.

## Security considerations

This is a prototype, but the design choices reflect the threat model a
production deployment would face. Resume text is untrusted input flowing into
an LLM, and the app surfaces a few concrete defenses:

- **Allowlist canonicalization** — LLM-extracted skills are mapped to a fixed
  taxonomy of canonical IDs in `src/skills_extractor.py`. Any string the model
  emits that isn't in the taxonomy is dropped, so a prompt-injected resume
  can't propagate arbitrary text into downstream UI or scoring.
- **Render-time markdown escape** — every untrusted string (resume snippets,
  LLM rationale) passes through `src/text_safety.py::escape_markdown` before
  reaching `st.markdown`. Closes the "resume contains
  `[click](https://evil.com)`" phishing vector. Streamlit blocks raw HTML by
  default; this closes the markdown-link gap on top.
- **No model-invented URLs** — the roadmap model orders skills and writes
  rationale only; resource URLs come from `data/resources.json`. Removes the
  obvious "inject a phishing link via the resume" path.
- **Strict JSON response format** — both OpenAI calls use
  `response_format={"type": "json_object"}` with explicit schemas in
  `src/prompts.py`. Malformed responses fall back deterministically.
- **Bounded inputs (length only)** — resume text capped at 8000 chars and
  portfolio at 2000 chars before reaching the model; uploaded PDFs capped at
  5 MB to bound parser memory. This is length-capping only — full input
  sanitization (non-printable stripping, zero-width-unicode handling) is
  described in `docs/ARCHITECTURE.md` §8.3 as a production step and is not
  implemented in the prototype.

The full production threat model — prompt injection, audit logging, tenant
isolation, secret handling — lives in `docs/ARCHITECTURE.md` §8.

## Limitations

This is a prototype. These are conscious tradeoffs, not oversights:

- **Synthetic ~100-row dataset.** TF-IDF + Logistic Regression per the
  assignment brief, trained on a synthetic corpus. The classifier is
  illustrative, not production-calibrated; the composite score leans on
  overlap for this reason. Retraining on a real job corpus with held-out
  metrics is a production step, not a prototype one.
- **No seniority modeling.** The classifier predicts `role_category` only.
  Seniority is in the data but not used as a target.
- **Curated resource seed.** `resources.json` covers the top ~50 skills with
  2–4 handpicked links each. Production would wire to a real catalog
  (Coursera/Udemy/LinkedIn Learning) rather than bundling URLs.
- **Qualitative evaluation.** Spot-checked against a handful of resumes;
  there is no held-out benchmark.

## Structure

```
app.py                # Streamlit UI (no business logic)
src/
  config.py           # Settings dataclass, constants, paths
  pdf_utils.py        # PDF text extraction with empty-text detection
  openai_client.py    # Returns None if no key or toggle disabled
  prompts.py          # OpenAI prompts + JSON schemas as first-class artifacts
  skills_extractor.py # OpenAI + rule-based fallback; both canonicalized
  job_data.py         # Load/validate jobs, taxonomy, resources
  classifier.py       # TF-IDF + LogisticRegression, Streamlit-cached
  matcher.py          # Overlap, composite score, missing skills, what-if
  roadmap.py          # Sequences curated resources (model orders, doesn't invent)
data/
  jobs.csv            # Synthetic job postings
  skills_taxonomy.json
  resources.json
tests/                # pytest; deterministic paths only
```
