"""Reading AVeriTeC records: the gold verdict, the annotators' questions, and their answers.

A record is one entry of the AVeriTeC json files (claim, label, questions with answers).
"""
from __future__ import annotations

import re

LABELS = {"Supported": "supported", "Refuted": "refuted",
          "Not Enough Evidence": "cannot_determine",
          "Conflicting Evidence/Cherrypicking": "conflicting"}

_TOK = re.compile(r"[0-9a-z]+(?:'[a-z]+)?", re.I)
_STOP = {  # words that do not count toward an answer's content
    'a', 'about', 'also', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'being', 'but', 'by',
    'can', 'could', 'did', 'do', 'does', 'for', 'from', 'had', 'has', 'have', 'he', 'if', 'in',
    'into', 'is', 'it', 'its', 'may', 'might', 'more', 'most', 'must', 'no', 'not', 'of', 'on',
    'or', 'over', 's', 'said', 'says', 'she', 'should', 'so', 't', 'than', 'that', 'the', 'their',
    'then', 'there', 'these', 'they', 'this', 'those', 'to', 'under', 'was', 'we', 'were', 'what',
    'when', 'where', 'which', 'who', 'whom', 'will', 'with', 'would', 'you'
}


def gold_verdict(record: dict) -> str:
    """The record's label as a verifier verdict."""
    return LABELS[record["label"]]


def claim_date(record: dict) -> str:
    """The claim's date as YYYY-MM-DD (AVeriTeC writes day-month-year), or "" if it cannot be read."""
    m = re.fullmatch(r"(\d{1,2})-(\d{1,2})-(\d{4})", (record.get("claim_date") or "").strip())
    return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}" if m else ""


def questions(record: dict) -> list[str]:
    """The annotators' questions, without repeats (case-insensitive), in order. What the judge sees."""
    seen, out = set(), []
    for q in record.get("questions") or []:
        t = (q.get("question") or "").strip()
        if t and t.casefold() not in seen:
            seen.add(t.casefold())
            out.append(t)
    return out


def _content_words(s: str) -> set[str]:
    return {w for w in (m.group(0).casefold() for m in _TOK.finditer(s or "")) if len(w) > 1 and w not in _STOP}


def qa_pairs(record: dict, max_questions: int = 10) -> list[tuple[str, str]]:
    """(question, answer) pairs for Ev2R: the first `max_questions` questions, unanswerable ones
    skipped, and a bare yes/no answer replaced by its explanation."""
    out = []
    for q in (record.get("questions") or [])[:max_questions]:
        qt = (q.get("question") or "").strip()
        for a in q.get("answers") or []:
            if a.get("answer_type") == "Unanswerable" or "no answer could be found" in (a.get("answer") or "").casefold():
                continue
            s = (a.get("answer") or "").strip()
            if len(_content_words(s)) < 2:
                s = (a.get("boolean_explanation") or "").strip()
            if s and len(_content_words(s)) >= 2:
                out.append((qt, s))
    return out
