"""SkillBridge Career Navigator — Streamlit UI.

UI-only by design; all business logic lives in src/. The top-level error boundary
catches any uncaught exception so the demo never shows a raw traceback.
"""
from __future__ import annotations

import hashlib
import io
import logging
import threading
import traceback
from collections import deque

import pandas as pd
import streamlit as st

from src.classifier import get_classifier
from src.config import TARGET_ROLES, Settings, configure_logging
from src.job_data import load_jobs, load_taxonomy
from src.matcher import missing_skills, probabilities_table, score, what_if
from src.pdf_utils import PDFReadError, extract_text
from src.roadmap import generate_roadmap
from src.skills_extractor import extract_skills
from src.text_safety import escape_markdown

MAX_UPLOAD_BYTES = 5 * 1_000_000  # 5 MB (decimal) — matches README and the displayed label.

STEPS = ("1. Input", "2. Analysis", "3. Roadmap & What-If")

log = configure_logging()

st.set_page_config(page_title="SkillBridge Career Navigator", layout="wide")


# ---------- Sidebar ----------
def render_sidebar() -> Settings:
    """Render the sidebar. Debug log is rendered FIRST so it stays visible even
    if Settings.load() or anything below it raises. Settings failures are caught
    here rather than bubbling to main()'s error boundary, because that boundary
    should only protect the step rendering — the user needs the sidebar controls
    and debug log to recover from a broken config.
    """
    st.sidebar.title("SkillBridge")

    # Always-visible debug log: rendered before any code that could fail.
    with st.sidebar.expander("Debug log", expanded=False):
        st.caption("Tail of the most recent log lines (if any). Full logs are printed to stderr.")
        st.code(_tail_log(), language="text")

    try:
        has_key = bool(Settings.load().openai_api_key)
    except Exception:
        log.exception("Settings load failed in sidebar")
        st.sidebar.error(
            "Could not load configuration — check your .env. "
            "Running in fallbacks-only mode."
        )
        return Settings(openai_api_key=None, openai_model="gpt-4o-mini", use_fallbacks_only=True)

    st.sidebar.markdown(
        f"**OpenAI key:** {'✅ detected' if has_key else '⚠️ not set'}"
    )
    st.session_state.setdefault("use_fallbacks_only", False)
    st.sidebar.checkbox(
        "Use fallbacks only",
        key="use_fallbacks_only",
        help="Forces the rule-based extractor and curated-only roadmap even if a key is set.",
    )

    return Settings.load(use_fallbacks_only=st.session_state["use_fallbacks_only"])


# deque(maxlen=200) makes append + bounded trim atomic under the GIL, but
# iterating it (e.g., list(_LOG_BUFFER) in _tail_log) can still raise
# RuntimeError if another thread appends mid-iteration. _LOG_LOCK serializes
# write vs. snapshot so both sides are safe under Streamlit's worker threads.
_LOG_BUFFER: deque[str] = deque(maxlen=200)
_LOG_LOCK = threading.Lock()


class _BufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        with _LOG_LOCK:
            _LOG_BUFFER.append(msg)


def _install_buffer_handler() -> None:
    logger = logging.getLogger("skillbridge")
    if any(isinstance(h, _BufferHandler) for h in logger.handlers):
        return
    handler = _BufferHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(handler)


def _tail_log(n: int = 30) -> str:
    # Snapshot under the lock so a concurrent emit() can't mutate the deque
    # mid-iteration (CPython raises RuntimeError on that). Slicing happens
    # outside the critical section.
    with _LOG_LOCK:
        lines = list(_LOG_BUFFER)
    return "\n".join(lines[-n:]) or "(no log lines yet)"


# ---------- Input hashing ----------
def _input_hash(
    resume_text: str,
    portfolio_text: str,
    target_role: str,
    use_fallbacks_only: bool,
    openai_model: str,
) -> str:
    # openai_model is in the key so a model flip invalidates cached analysis.
    h = hashlib.sha256()
    for part in (resume_text, portfolio_text, target_role, str(use_fallbacks_only), openai_model):
        h.update(part.encode("utf-8", errors="ignore"))
        h.update(b"\x00")
    return h.hexdigest()


# ---------- Tabs ----------
def render_input_tab(settings: Settings) -> None:
    st.subheader("Step 1 · Your resume")
    st.caption("Upload or paste your resume, pick a target role, then **Analyze**.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Resume**")
        resume_pdf = st.file_uploader(
            "Resume PDF", type=["pdf"], key="resume_pdf", label_visibility="collapsed"
        )
        resume_text_input = st.text_area("…or paste resume text", height=220, key="resume_text_input")
    with col_b:
        st.markdown("**Portfolio** &nbsp; _(optional — adds to extracted skills)_")
        portfolio_pdf = st.file_uploader(
            "Portfolio PDF", type=["pdf"], key="portfolio_pdf", label_visibility="collapsed"
        )
        portfolio_text_input = st.text_area("…or paste portfolio text", height=220, key="portfolio_text_input")

    target_role = st.selectbox(
        "Target role", TARGET_ROLES, key="target_role",
        help="We'll score your fit for this role and list the missing skills.",
    )

    if st.button("Analyze →", type="primary"):
        resume_text = _resolve_text(resume_pdf, resume_text_input, "resume")
        portfolio_text = _resolve_text(portfolio_pdf, portfolio_text_input, "portfolio")
        if not resume_text:
            st.warning("Please upload a resume PDF or paste resume text.")
            return
        _run_analysis(resume_text, portfolio_text, target_role, settings)


def _resolve_text(file, text: str, label: str) -> str:
    if file is not None:
        size = getattr(file, "size", 0) or 0
        if size > MAX_UPLOAD_BYTES:
            st.warning(
                f"{label.capitalize()} PDF is {size / 1_000_000:.1f} MB — "
                f"max is {MAX_UPLOAD_BYTES // 1_000_000} MB. Using pasted text if provided."
            )
            return text.strip()
        try:
            return extract_text(io.BytesIO(file.read()))
        except PDFReadError as e:
            st.warning(f"{label.capitalize()} PDF: {e}")
            return text.strip()
        except Exception:
            # Defense-in-depth for unanticipated failures (e.g., file.read() itself).
            log.exception("Unexpected PDF failure for %s", label)
            st.warning(f"Could not read {label} PDF; using pasted text if provided.")
            return text.strip()
    return text.strip()


def _run_analysis(resume_text: str, portfolio_text: str, target_role: str, settings: Settings) -> None:
    new_hash = _input_hash(
        resume_text, portfolio_text, target_role, settings.use_fallbacks_only, settings.openai_model
    )
    if st.session_state.get("analysis_hash") != new_hash:
        st.session_state.pop("analysis", None)
    st.session_state["analysis_hash"] = new_hash

    with st.spinner("Extracting skills and scoring..."):
        taxonomy = load_taxonomy()
        # Feed both texts to extraction so portfolio-only skills don't surface as gaps.
        candidate_text = resume_text + (("\n\n" + portfolio_text) if portfolio_text else "")
        extraction = extract_skills(candidate_text, taxonomy, settings)
        clf = get_classifier()
        jobs_df = load_jobs()
        scoring = score(extraction.skills, target_role, clf, jobs_df, resume_text, portfolio_text)
        probs = probabilities_table(clf, extraction.skills, resume_text, portfolio_text)
        gaps = missing_skills(extraction.skills, target_role, jobs_df)

    st.session_state["analysis"] = {
        "resume_text": resume_text,
        "portfolio_text": portfolio_text,
        "target_role": target_role,
        "extraction": extraction,
        "score": scoring.as_dict(),
        "probs": probs,
        "gaps": gaps,
    }
    st.toast(
        f"Analyzed — {len(extraction.skills)} skills extracted ({extraction.source})",
        icon="✅",
    )
    # `_pending_nav` is consumed before segmented_control instantiates —
    # Streamlit forbids writing a widget's own key after it renders.
    st.session_state["_pending_nav"] = STEPS[1]
    st.rerun()


def render_analysis_tab() -> None:
    analysis = st.session_state.get("analysis")
    if not analysis:
        st.info("Run **Step 1 · Your resume** first.")
        return

    st.subheader("Step 2 · Your fit")
    st.caption("Skills we found, your role match, and what's missing. Step 3 turns the gaps into a plan.")

    extraction = analysis["extraction"]
    st.markdown("**Extracted skills**")
    badge = "🟢 OpenAI + rule-based" if extraction.source == "openai" else "🟡 rule-based only"
    st.caption(
        f"Source: {badge}",
        help="OpenAI infers skills from context; the rule-based scanner is always run as a safety net.",
    )
    if extraction.skills:
        st.write(" ".join(f"`{s}`" for s in extraction.skills))
    else:
        st.warning("No skills matched the taxonomy.")
    if extraction.evidence:
        with st.expander("Evidence (matched snippets)"):
            for skill, snippet in extraction.evidence.items():
                # `skill` is canonical (allowlisted); `snippet` is raw resume text.
                st.markdown(f"**{skill}** — _{escape_markdown(snippet)}_")

    st.divider()
    st.markdown("**Role probabilities**")
    st.caption("How the classifier ranks you across all roles.")
    probs_df = pd.DataFrame(analysis["probs"], columns=["role", "probability"]).set_index("role")
    st.bar_chart(probs_df)

    st.divider()
    st.markdown(f"**Fit vs {analysis['target_role']}**")
    s = analysis["score"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall fit", f"{s['composite']:.2f}",
                help="0–1. Composite of classifier probability (0.4) and skill overlap (0.6).")
    col2.metric("Role match (model)", f"{s['classifier_prob']:.2f}",
                help="0–1. Classifier probability for the target role.")
    col3.metric("Skills covered", f"{s['skill_overlap_pct']:.2f}",
                help="0–1. Fraction of the role's weighted skill demand your resume covers.")

    st.divider()
    st.markdown("**Missing skills for this role**")
    if analysis["gaps"]:
        st.caption("Required by jobs in this role but not found in your resume — head to Step 3 to plan them.")
        gap_df = pd.DataFrame(analysis["gaps"], columns=["skill", "jobs requiring it"])
        st.dataframe(gap_df, width="stretch", hide_index=True)
    else:
        st.success("You match all required skills for this role.")


def render_roadmap_tab(settings: Settings) -> None:
    analysis = st.session_state.get("analysis")
    if not analysis:
        st.info("Run **Step 1 · Your resume** first.")
        return

    gaps = analysis["gaps"]
    current_skills: list[str] = analysis["extraction"].skills
    target_role: str = analysis["target_role"]
    resume_text: str = analysis["resume_text"]
    portfolio_text: str = analysis["portfolio_text"]

    st.subheader("Step 3 · Plan your gaps")
    if not gaps:
        st.success("You already match every required skill for this role — nothing to plan.")
        return

    st.caption("Try the what-if to see how learning specific skills moves your fit, then follow the roadmap below.")

    st.markdown("**What-If simulation**")
    selected = st.multiselect(
        "Pick skills to simulate learning — your fit scores update below.",
        options=[s for s, _ in gaps],
        default=[s for s, _ in gaps[:3]],
        key="whatif_selection",
    )

    jobs_df = load_jobs()
    clf = get_classifier()
    wi = what_if(current_skills, selected, target_role, clf, jobs_df, resume_text, portfolio_text)
    c1, c2, c3 = st.columns(3)
    c1.metric("Fit now", f"{wi['base']['composite']:.2f}")
    c2.metric("Fit with additions", f"{wi['simulated']['composite']:.2f}",
              delta=f"{wi['delta']['composite']:+.2f}")
    c3.metric("Skills-covered Δ", f"{wi['delta']['skill_overlap_pct']:+.2f}")

    st.divider()
    st.markdown("**Learning roadmap**")
    st.caption("Ordered by impact for this role. Each skill links to curated resources.")
    with st.spinner("Building roadmap..."):
        roadmap = generate_roadmap(
            target_role, current_skills, gaps, portfolio_text, settings
        )
    st.caption(
        f"Roadmap source: {'🟢 openai' if roadmap.source == 'openai' else '🟡 fallback'} · "
        f"Estimated effort: ~{roadmap.estimated_weeks} weeks at ~10 hrs/week"
    )
    for i, skill in enumerate(roadmap.priority_order, start=1):
        with st.container(border=True):
            # `skill` is canonical; `rationale` is LLM-generated and may reflect untrusted portfolio text.
            rationale = escape_markdown(roadmap.rationale.get(skill, ""))
            st.markdown(f"**{i}. `{skill}`** — {rationale}")
            resources = roadmap.resources.get(skill, [])
            if not resources:
                st.caption("_No curated resources for this skill yet._")
                continue
            for r in resources:
                kind = r.get("kind", "docs")
                hours = r.get("est_hours", "?")
                st.markdown(
                    f"- [{r['title']}]({r['url']}) · _{kind}_ · ~{hours}h"
                )


# ---------- Main ----------
def _render_step(settings: Settings) -> None:
    """Nav widget + step rendering. The only part wrapped by the error boundary
    in main() — everything else (sidebar, debug log, title) must render first
    so the user can recover from step-level failures.
    """
    # Programmatic nav (queued from _run_analysis) applied before the widget
    # instantiates, so we can legally write to its backing session-state key.
    if "_pending_nav" in st.session_state:
        st.session_state["nav"] = st.session_state.pop("_pending_nav")
    st.session_state.setdefault("nav", STEPS[0])

    st.segmented_control(
        "Step", STEPS, key="nav", label_visibility="collapsed"
    )
    st.divider()

    current = st.session_state["nav"]
    if current == STEPS[0]:
        render_input_tab(settings)
    elif current == STEPS[1]:
        render_analysis_tab()
    else:
        render_roadmap_tab(settings)


def main() -> None:
    _install_buffer_handler()
    settings = render_sidebar()
    st.title("SkillBridge Career Navigator")
    st.caption("Resume → canonical skills → role fit → what-if → learning roadmap.")

    try:
        _render_step(settings)
    except Exception as e:
        log.exception("Error during step rendering")
        st.error(
            f"Something went wrong ({type(e).__name__}). "
            "Check the debug log in the sidebar."
        )
        with st.expander("Traceback (debug)"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
