"""Ev2R recall: how many of the annotators' answers the pool states.

For each gold (question, answer) pair, a grader model grades whether the evidence states the
answer (1), part of it (0.5) or not (0). A claim's recall is the mean over its pairs. Ev2R is the
mean over claims.
"""
from __future__ import annotations

import json
import re

SYSTEM = 'You judge whether retrieved evidence covers a gold answer. Answer only with JSON.'

TEMPLATE = """Gold question: {question}
Gold answer: {answer}

Retrieved evidence:
{evidence}

Does the retrieved evidence contain the information in the gold answer?
- "yes": the evidence states the gold answer's information (wording may differ)
- "partial": part of it, or only via a weaker/vaguer statement
- "no": the evidence does not contain it

Reply ONLY JSON: {{"coverage": "yes"|"partial"|"no"}}"""

POINTS = {"yes": 1.0, "partial": 0.5, "no": 0.0}


def evidence(documents: list[dict], cap: int | None = None) -> str:
    """The pool as the grader sees it: every non-empty document, prefixed with [?]."""
    text = "\n\n".join(f"[?] {d['text'].strip()}" for d in documents if (d.get("text") or "").strip())
    return text if cap is None else text[:cap]


def grade(llm, question: str, answer: str, evidence_text: str) -> str:
    reply = llm(SYSTEM, TEMPLATE.format(question=question, answer=answer[:600],
                                        evidence=evidence_text or "(empty)"))
    m = re.search(r"\{.*\}", reply or "", re.S)
    try:
        cov = json.loads(m.group(0)).get("coverage", "") if m else ""
    except json.JSONDecodeError:
        cov = ""
    return cov if cov in POINTS else "parse_fail"


def recall(llm, documents: list[dict], qa_pairs: list[tuple[str, str]], cap: int | None = None) -> dict:
    ev = evidence(documents, cap)
    grades = [grade(llm, q, a, ev) for q, a in qa_pairs]
    score = sum(POINTS.get(g, 0.0) for g in grades) / len(grades) if grades else 0.0
    return {"recall": score, "grades": [{"question": q, "answer": a, "coverage": g}
                                        for (q, a), g in zip(qa_pairs, grades)]}
