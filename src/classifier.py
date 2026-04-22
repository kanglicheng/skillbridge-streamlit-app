"""TF-IDF + LogisticRegression role classifier. Cached via Streamlit."""
from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from .config import JOBS_CSV
from .job_data import load_jobs

log = logging.getLogger("skillbridge")


@dataclass(frozen=True)
class Classifier:
    vectorizer: TfidfVectorizer
    model: LogisticRegression
    classes: tuple[str, ...]


def _build(csv_mtime: float) -> Classifier:
    del csv_mtime  # only used as a cache key; load_jobs reads the current file
    df = load_jobs()
    texts = (df["description"].astype(str) + " " + df["required_skills"].astype(str).str.replace("|", " ", regex=False)).tolist()
    labels = df["role_category"].tolist()
    # Custom token_pattern keeps hyphens inside tokens so multi-word canonicals
    # ("incident-response", "github-actions") stay atomic features; sklearn's
    # default would split on `-`.
    vectorizer_kwargs = dict(
        min_df=1,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b\w[\w-]+\b",
    )
    lr_kwargs = dict(max_iter=1000, class_weight="balanced", random_state=42)

    # Fresh Pipeline per fold — fitting the vectorizer once upfront would leak
    # test-fold vocabulary into training features.
    try:
        min_class = min(Counter(labels).values())
        k = min(5, min_class)
        if k >= 2:
            cv_pipe = Pipeline([
                ("tfidf", TfidfVectorizer(**vectorizer_kwargs)),
                ("lr", LogisticRegression(**lr_kwargs)),
            ])
            scores = cross_val_score(cv_pipe, texts, labels, cv=k, scoring="accuracy")
            log.info(
                "Classifier %d-fold CV accuracy: %.2f ± %.2f (n=%d, classes=%d)",
                k, scores.mean(), scores.std(), len(labels), len(set(labels)),
            )
        else:
            log.info("Skipping CV: smallest class has %d sample(s)", min_class)
    except Exception:
        log.exception("CV evaluation failed; continuing with training")

    vectorizer = TfidfVectorizer(**vectorizer_kwargs)
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression(**lr_kwargs)
    model.fit(X, labels)
    return Classifier(vectorizer=vectorizer, model=model, classes=tuple(model.classes_))


# Streamlit caches; csv_mtime is an explicit arg so edits invalidate the cache.
@st.cache_resource(show_spinner=False)
def build_classifier_cached(csv_mtime: float) -> Classifier:
    return _build(csv_mtime)


def get_classifier() -> Classifier:
    return build_classifier_cached(os.path.getmtime(JOBS_CSV))


def predict_proba(clf: Classifier, skills: Iterable[str], extra_text: str = "") -> dict[str, float]:
    """Return class_name → probability for a skills list + optional free text.

    `extra_text` is concatenated before vectorization so inference sees prose
    like training did (descriptions + required_skills), not skill tokens alone.
    """
    query = " ".join(skills) + " " + extra_text
    X = clf.vectorizer.transform([query])
    probs = clf.model.predict_proba(X)[0]
    return {cls: float(p) for cls, p in zip(clf.classes, probs)}
