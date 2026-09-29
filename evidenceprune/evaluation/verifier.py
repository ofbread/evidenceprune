from __future__ import annotations

import json
import re

SYSTEM = (
    "You are a careful fact-checker. You may use ONLY the documents "
    "provided — not your own knowledge of the topic. If the documents "
    "do not decide the claim, say cannot_determine.")

TEMPLATE = """Claim to check:
{claim}
The claim was made on: {date}

Documents (excerpts; use only these):
{docs}

Decide the claim from these documents alone.
Reply with ONLY JSON:
{{"verdict": "supported" | "refuted" | "conflicting" | "cannot_determine",
"citations": ["<COPY 1-3 sentences EXACTLY, character for character, from the documents that decide it>"],
"because": "<one short sentence>"}}"""

NO_DOCUMENTS = '(no documents were retrieved)'
DOC_CAP = 12_000
PROMPT_BUDGET = 600_000
VERDICTS = ("supported", "refuted", "conflicting", "cannot_determine")

_FOLD = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
                       "\u2013": "-", "\u2014": "-", "\u2010": "-", "\u2011": "-", "\u00a0": " "})


def norm(s: str) -> str:
    """Fold curly quotes and dashes, collapse whitespace, lower-case."""
    return re.sub(r"\s+", " ", (s or "").translate(_FOLD)).strip().lower()


def render(claim: str, claim_date: str, documents: list[dict], doc_cap: int = DOC_CAP,
           prompt_budget: int = PROMPT_BUDGET) -> tuple[str, list[str]]:
    """The verifier's prompt, and the exact text of each document as shown (citations are checked
    against these). A document is a dict with `text` and optionally `title`, `domain`, `tier`
    and `published` (YYYY-MM-DD)."""
    cap = doc_cap
    total = sum(min(len(d.get("text") or ""), cap) for d in documents)
    if total > prompt_budget:
        cap = max(400, prompt_budget // max(1, len(documents)))
    blocks, shown = [], []
    for i, d in enumerate(documents):
        text = (d.get("text") or "")[:cap]
        shown.append(text)
        title = (d.get("title") or "").strip()[:120]
        head = (f"[D{i}] " + (f"{title} — " if title else "")
                + f"{d.get('domain', '?')} ({d.get('tier', '?')}, published "
                  f"{(d.get('published') or 'unknown')[:10]})")
        blocks.append(f"{head}\n{text}")
    docs = "\n\n".join(blocks) if blocks else NO_DOCUMENTS
    return TEMPLATE.format(claim=claim, date=claim_date or "unknown", docs=docs), shown


def parse(reply: str) -> dict:
    """The JSON object in the model's reply, or {} if there is none."""
    m = re.search(r"\{.*\}", reply or "", re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}


def citation_found(citation: str, shown: list[str]) -> bool:
    """A citation counts if its first 400 characters (at least 40) appear in a shown document."""
    c = norm(citation)[:400]
    return len(c) >= 40 and any(c in norm(s) for s in shown)


def score(verdict: str, n_verified: int, gold: str) -> str:
    """right, wrong, ungrounded (a verdict with no verified citation), cannot (no verdict when the
    gold label has one) or parse_failed. `gold` is one of VERDICTS."""
    if verdict not in VERDICTS:
        return "parse_failed"
    if verdict == "cannot_determine":
        return "right" if gold == "cannot_determine" else "cannot"
    if n_verified < 1:
        return "ungrounded"
    return "right" if verdict == gold else "wrong"


def verify(llm, claim: str, documents: list[dict], claim_date: str = "", gold: str | None = None,
           doc_cap: int = DOC_CAP) -> dict:
    """Run the verifier on one claim. `llm(system, prompt)` returns the model's reply text.
    With `gold`, the result also carries `score` (see `score`)."""
    prompt, shown = render(claim, claim_date, documents, doc_cap)
    reply = llm(SYSTEM, prompt)
    j = parse(reply)
    verdict = str(j.get("verdict", "")).strip().lower()
    citations = [c for c in (j.get("citations") or []) if isinstance(c, str)]
    n_ok = sum(citation_found(c, shown) for c in citations)
    out = {"verdict": verdict, "citations": citations, "n_verified": n_ok,
           "because": str(j.get("because", "")), "reply": reply}
    if gold is not None:
        out["score"] = score(verdict, n_ok, gold)
    return out


def accuracy(scores: list[str]) -> float:
    """The share of claims scored right."""
    return sum(s == "right" for s in scores) / len(scores) if scores else 0.0
