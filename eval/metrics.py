"""Eval metrics: section-level token F1 for SOAP notes, top-k accuracy for codes."""

from __future__ import annotations

import re
import string
from collections import Counter

from open_scribe.note_gen import SoapNote

SECTIONS = ("subjective", "objective", "assessment", "plan")

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.'/-][a-z0-9]+)*")


def _tokenize(text: str) -> list[str]:
    """Lowercase + alphanumeric token split. Drops punctuation, keeps hyphenated terms intact."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _f1(pred_tokens: list[str], gold_tokens: list[str]) -> dict[str, float]:
    """Token-level F1, treating tokens as a multiset (preserves repeated terms)."""
    if not pred_tokens and not gold_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())

    precision = overlap / sum(pred_counts.values())
    recall = overlap / sum(gold_counts.values())
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def section_f1(predicted: SoapNote, gold: SoapNote) -> dict[str, dict[str, float]]:
    """Per-section precision / recall / F1 between two SOAP notes."""
    return {
        section: _f1(_tokenize(getattr(predicted, section)), _tokenize(getattr(gold, section)))
        for section in SECTIONS
    }


def aggregate_section_f1(per_example: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """Macro-average per-section metrics across N examples."""
    if not per_example:
        return {}
    out: dict[str, dict[str, float]] = {}
    for section in SECTIONS:
        out[section] = {
            metric: sum(e[section][metric] for e in per_example) / len(per_example)
            for metric in ("precision", "recall", "f1")
        }
    return out


def whole_note_f1(predicted_text: str, gold_text: str) -> dict[str, float]:
    """Token-level F1 over the entire note as a single string.

    For datasets like PriMock57 where gold notes are free-form prose rather
    than structured SOAP, this is the most honest metric.
    """
    return _f1(_tokenize(predicted_text), _tokenize(gold_text))


def icd10_top_k_accuracy(
    predicted_codes: list[str], gold_codes: list[str], k: int = 5
) -> float:
    """Fraction of gold codes appearing in the top-k predicted codes."""
    if not gold_codes:
        return 0.0
    top_k = set(predicted_codes[:k])
    return sum(1 for g in gold_codes if g in top_k) / len(gold_codes)
