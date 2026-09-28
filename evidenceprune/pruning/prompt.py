"""Builds prompt for pruning"""

from __future__ import annotations

import re

SYSTEM = ("You mark which sentences of a document a fact-checker must "
          "keep. You reply only with JSON.")

TEMPLATE = """Claim under check:
{claim}{card}
{doc}
What must be established:
{points}

The document, as numbered sentences:
{sents}

Decide for EVERY sentence whether a fact-checker verifying this claim would
need it. Keep a sentence if it states a fact bearing on the claim or any point
above, or if it supplies the speaker, date, place or units that another kept
sentence depends on. Drop navigation, boilerplate, other stories, and text
about a different topic. Judge each sentence in the context of the whole
document, not on its own.

If the document bears on nothing above, keep nothing.

Reply with ONLY JSON: {{"keep": [<numbers>]}}"""

HEAD_MARK = "The document, as numbered sentences:\n"
TAIL_MARK = "\n\nDecide for EVERY sentence"
SENT_RE = re.compile(r"^S(\d+): ", re.M)
SENT_CAP = 400
NO_POINTS = "  (nothing specific was recorded — judge against the claim itself)"

_FILLER = re.compile(r'^\s*(anything incompatible with the claim|'
                     r'anything that contradicts (the )?claim|n/?a|none)\s*$', re.I)


_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December"]


def spell_date(date: str) -> str:
    """'2020-10-30' -> '30 October 2020'"""
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", (date or "").strip())
    if not m or not 1 <= int(m.group(2)) <= 12:
        return (date or "").strip()
    return f"{int(m.group(3))} {_MONTHS[int(m.group(2)) - 1]} {m.group(1)}"


def claim_card(claim_date: str = "", speaker: str = "") -> str:
    bits = []
    if claim_date:
        bits.append(f"made on {spell_date(claim_date)}")
    if speaker:
        bits.append(f"by {speaker}")
    return ("\n" + ", ".join(bits)) if bits else ""


def doc_card(title: str = "", url: str = "", domain: str = "", part: int = 0, of: int = 0) -> str:
    if not (title or url):
        return ""
    out = ["", f"DOCUMENT: {(title or '(untitled)')[:140]}"]
    src = domain or ""
    u = (url or "")[:130]
    if src or u:
        out.append(f"SOURCE:   {src}{'  ·  ' if src and u else ''}{u}")
    if of > 1:
        out.append(f"(part {part} of {of} of this document)")
    return "\n".join(out) + "\n"


def points_block(requirements: list[str] | None) -> str:
    reqs = [r for r in (requirements or []) if r and not _FILLER.match(str(r))]
    return "\n".join(f"  R{i + 1}. {r}" for i, r in enumerate(reqs)) or NO_POINTS


def render_sentences(text: str, spans: list[tuple[int, int]], base: int) -> str:
    """S<global index>: <sentence>, whitespace collapsed, cut at 400 chars."""
    return "\n".join(f"S{base + i}: " + " ".join(text[a:b].split())[:SENT_CAP]
                     for i, (a, b) in enumerate(spans))


def render(claim: str, text: str, spans: list[tuple[int, int]], base: int, *,
           requirements: list[str] | None = None, claim_date: str = "", speaker: str = "",
           title: str = "", url: str = "", domain: str = "", part: int = 0, of: int = 0) -> str:
    return TEMPLATE.format(claim=claim, card=claim_card(claim_date, speaker),
                           doc=doc_card(title, url, domain, part, of),
                           points=points_block(requirements),
                           sents=render_sentences(text, spans, base))


def window_ids(prompt: str) -> list[int]:
    """The global sentence ids rendered in a prompt."""
    body = prompt.split(HEAD_MARK, 1)[-1].split(TAIL_MARK, 1)[0]
    return [int(m.group(1)) for m in SENT_RE.finditer(body)]


def split_prompt(prompt: str) -> tuple[str, list[str]]:
    """(head, sentence texts) for encoder."""
    head, _, body = prompt.partition(HEAD_MARK)
    body = body.split(TAIL_MARK, 1)[0]
    return head.rstrip(), [m.group(1) for m in re.finditer(r"^S\d+: (.*)$", body, re.M)]
