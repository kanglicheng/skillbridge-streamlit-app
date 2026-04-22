"""All OpenAI prompts, isolated from business logic for reviewability.

Each prompt targets strict JSON output via response_format={"type":"json_object"}
so downstream parsers don't have to be defensive about prose.
"""
from __future__ import annotations

# Extraction: asks the model to pick only from the provided taxonomy hints.
# Free-form extraction (without the hint list) makes canonicalization noisier.
SKILLS_EXTRACTION_SYSTEM = (
    "You extract professional skills from a candidate's resume. "
    "You ONLY return skills that appear in, or are direct synonyms of, the provided taxonomy hint list. "
    "Do not invent skills. Return strict JSON."
)

SKILLS_EXTRACTION_USER_TEMPLATE = """Taxonomy hints (canonical IDs, pick from these or their common synonyms):
{hints}

Candidate text (resume, with optional portfolio appended):
\"\"\"
{resume_text}
\"\"\"

Return JSON with exactly this shape:
{{
  "skills": ["<canonical_or_synonym>", ...],
  "evidence": {{ "<skill>": "<short snippet from the text showing this skill>", ... }}
}}
Include evidence only when you can cite a snippet. Max 30 skills."""


# Roadmap: the model orders and rationalizes ONLY. Resources come from resources.json
# after the model returns — the model is never asked to produce URLs.
ROADMAP_SYSTEM = (
    "You are a career coach helping a candidate close skill gaps. "
    "You will be given a target role, the candidate's current skills, and the missing skills. "
    "Your job is to prioritize the missing skills (not invent resources). Return strict JSON."
)

ROADMAP_USER_TEMPLATE = """Target role: {target_role}
Current skills: {current_skills}
Missing skills (must appear in priority_order exactly as given, no additions): {missing_skills}
Portfolio/projects summary (may be empty): \"\"\"{portfolio_text}\"\"\"

Return JSON with exactly this shape:
{{
  "priority_order": [<missing skill ids in recommended learning order>],
  "rationale": {{ "<skill>": "<1 short sentence on why this order>", ... }},
  "estimated_weeks": <integer, rough total calendar time assuming ~10 focused hours/week>
}}"""
