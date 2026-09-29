"""The blind judge: sufficiency, coverage and relevance precision of an evidence pool.

The judge sees the claim, the annotators' questions and every document of the pool, shuffled
when a seed is given and without the method's name. It grades each document's relevance from
0 to 3, marks each question answered, partial or absent, and says whether the pool is
sufficient to settle the claim. Per claim:

    sufficiency  1 if the judge says the pool is sufficient, else 0
    coverage     questions marked answered / questions
    precision    documents graded 2 or 3 / documents graded
"""
from __future__ import annotations

import json
import random
import re

TASK = """You are auditing an evidence pool built by a fact-checking system.

You get a claim, the questions a professional fact-checker needed to answer
about it, and every document the system retrieved. Answer three separate
questions. They are at different levels and a pool can pass one and fail
another — judge each on its own.

1. RELEVANCE, per document. Grade EVERY document:
     3 = directly answers part of the claim
     2 = relevant evidence bearing on the claim
     1 = background only: on-topic, establishes nothing
     0 = not relevant to this claim
   Give a short reason, and for grade 2 or 3 quote VERBATIM the passage that
   earns it.

2. COVERAGE, over the whole set. For EACH numbered question, say whether the
   set as a whole answers it: 'answered' (some document states the answer),
   'partial' (the set gets close but does not state it), or 'absent'. Name
   the document and quote the text when it is answered. Ten documents that
   all say the same thing cover one question, not ten.

3. SUFFICIENCY, over the whole set. Could a careful reader settle this claim
   from this set ALONE, without fetching anything else? Answer yes or no,
   say why in one sentence, and name what is still missing if anything is.

Judge only what is in front of you. Do not use what you already know about
the claim to fill a gap the documents leave — if the pool does not contain
it, it is missing."""

SHAPE = '{"documents": [{"ref": "D1", "relevance": 0, "why": "...", "quote": "..."}], "coverage": [{"q": 1, "status": "answered|partial|absent", "ref": "D1 or empty", "quote": "..."}], "sufficiency": "yes|no", "sufficiency_reason": "...", "missing": "..."}'

DOC_CAP_MAX = 12_000        # characters of each document the judge sees
DOC_CAP_MIN = 3_000
BUDGET = 1_200_000          # characters for the whole pool; documents are cut to fit


def render(claim: str, questions: list[str], documents: list[dict], seed: str | None = None,
           budget: int = BUDGET) -> str:
    """The judge's prompt. A document is a dict with `text` and optionally `title`, `url` and
    `snippet` (True for a search-engine snippet rather than a page). With `seed`, the documents
    are shuffled; the paper used the seed "20260820|<claim id>|<method>"."""
    docs = [d for d in documents if d.get("text")]
    order = list(range(len(docs)))
    if seed is not None:
        random.Random(seed).shuffle(order)
    docs = [docs[i] for i in order]
    cap = DOC_CAP_MAX if not docs else max(DOC_CAP_MIN, min(DOC_CAP_MAX, budget // len(docs)))
    b = [TASK, "", f"CLAIM: {claim}", "", "QUESTIONS A FACT-CHECKER NEEDED TO ANSWER:"]
    b += [f"  Q{i}. {q}" for i, q in enumerate(questions, 1)]
    if not questions:
        b.append("  (none recorded — judge coverage as an empty list)")
    b += ["", f"THE POOL — {len(docs)} documents, in no particular order:"]
    for i, d in enumerate(docs, 1):
        t = d["text"]
        body = t[:cap]
        flag = (f" | SHOWN: first {len(body):,} of {len(t):,} chars" if len(t) > cap
                else f" | {len(t):,} chars")
        if d.get("snippet"):
            flag += (" | SEARCH-ENGINE SNIPPET ONLY — the page could not be "
                     "retrieved; this is the engine's short extract, not the document")
        b += ["", f"--- D{i} | {(d.get('title') or '')[:110]} | {(d.get('url') or '')[:110]}{flag} ---", body]
    b += ["", f"Reply with ONLY this JSON shape:\n{SHAPE}"]
    return "\n".join(b)


def parse(reply: str) -> dict | None:
    """The judge's JSON reply, or None if there is none (the claim then counts as not judged)."""
    m = re.search(r"\{.*\}", reply or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def scores(verdict: dict | None) -> dict:
    """Sufficiency, coverage and precision from a parsed reply."""
    v = verdict or {}
    grades = [d.get("relevance") for d in v.get("documents") or [] if isinstance(d.get("relevance"), int)]
    cov = v.get("coverage") or []
    answered = sum(1 for c in cov if c.get("status") == "answered")
    return {"sufficiency": 1 if v.get("sufficiency") == "yes" else 0,
            "coverage": answered / len(cov) if cov else 0.0,
            "precision": sum(1 for g in grades if g >= 2) / len(grades) if grades else 0.0}


def judge(llm, claim: str, questions: list[str], documents: list[dict], seed: str | None = None) -> dict | None:
    """Judge one pool. `llm(system, prompt)` returns the reply text; the judge uses no system
    prompt. Returns the three scores plus the parsed reply, or None if the reply has no JSON."""
    if not any(d.get("text") for d in documents):
        return {"sufficiency": 0, "coverage": 0.0, "precision": 0.0, "reply": None}
    verdict = parse(llm("", render(claim, questions, documents, seed)))
    if verdict is None:
        return None
    return {**scores(verdict), "reply": verdict}
