"""Sentence splitting and windowing.

sentences(text) returns (start, end) offsets into the original text.
"""
from __future__ import annotations

import re

WINDOW_SENTS = 48
WINDOW_CHARS = 10_000

_ABBREV = re.compile(
    r"(?:\b(?:Mr|Mrs|Ms|Dr|Prof|Sen|Rep|Gov|Gen|Col|Sgt|St|Jr|Sr|vs|etc|"
    r"No|Inc|Ltd|Co|Corp|Fig|Vol|Op|Jan|Feb|Mar|Apr|Jun|Jul|"
    r"Aug|Sep|Sept|Oct|Nov|Dec)|(?:[A-Za-z]\.){1,3}[A-Za-z])\.$")
_END = re.compile(r"[.!?][\"'”’)\]]*\s")


def sentences(text: str) -> list[tuple[int, int]]:
    spans = []
    pos = 0
    for line in text.split("\n"):
        if line.strip():
            start = pos
            for m in _END.finditer(line):
                end = pos + m.end() - 1
                head = text[start:end].rstrip(")\"'”’]")
                if _ABBREV.search(head.rstrip()):
                    continue
                if end - start >= 2:
                    spans.append((start, end))
                start = end
                while start < pos + len(line) and text[start] == " ":
                    start += 1
            if pos + len(line) - start >= 2:
                spans.append((start, pos + len(line)))
        pos += len(line) + 1
    return spans


def windows(text: str, max_sents: int = WINDOW_SENTS, max_chars: int = WINDOW_CHARS) -> list[list[tuple[int, int]]]:
    """Consecutive windows of sentence spans"""
    out, cur, n = [], [], 0
    for sp in sentences(text):
        w = sp[1] - sp[0]
        if cur and (len(cur) >= max_sents or n + w > max_chars):
            out.append(cur)
            cur, n = [], 0
        cur.append(sp)
        n += w
    if cur:
        out.append(cur)
    return out
