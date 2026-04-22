"""Real-classifier behavior pin: resume_text actually influences predict_proba.

Existing matcher tests use a FakeClassifier whose predict_proba ignores its input,
so a regression that silently drops resume_text on the floor (or stops passing it
through to the vectorizer) would go uncaught. This file uses the real
TF-IDF + LR classifier so the plumbing is end-to-end real.
"""
from __future__ import annotations

from src.classifier import get_classifier, predict_proba


def test_resume_text_shifts_classifier_probs():
    """Backend-shaped prose passed via extra_text must bump Backend Engineer prob.

    Pins both: (1) the plumbing — predict_proba's extra_text reaches the
    vectorizer rather than being silently dropped — and (2) the monotonic
    expectation — that prose vocabulary overlapping the training corpus
    moves the classifier in the expected direction.
    """
    clf = get_classifier()
    base = predict_proba(clf, ["python"], "")
    rich = predict_proba(
        clf, ["python"], "django fastapi postgres rest api backend services"
    )
    assert rich["Backend Engineer"] > base["Backend Engineer"]
