"""Claim-conditioned context pruning with a distilled student.

    from claimprune import Pruner
    pruner = Pruner("<hf-id-or-local-dir>")           # the 2B decoder or the 0.4B encoder
    kept = pruner.prune(claim, documents)             # documents: list of str or {"text", "title", "url"}

Each document is cut into windows of 48 sentences / 10,000 characters, every window is
scored in one forward pass, and the sentences whose P(keep) reaches the model's threshold
are copied VERBATIM (as slices of the original text, joined with "[…]" where a gap was
cut). The deployed reading rule applies by default: at most 16 windows per document, and
reading stops after two consecutive windows in which nothing is kept.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import prompt as P
from .sentences import windows as _windows

Scorer = Callable[[str, str, list[int]], list[float]]   # (prompt, system, global ids) -> P(keep) per id

DEFAULT_THRESHOLDS = {"decoder": 0.3073580265045166, "encoder": 0.3458289667963982}
DOC_CEILING = 12_000        # characters one document may contribute (cut at a sentence boundary)
ELISION = "\n\n[…]\n\n"


@dataclass
class Document:
    text: str
    title: str = ""
    url: str = ""
    domain: str = ""

    @classmethod
    def of(cls, d) -> "Document":
        if isinstance(d, Document):
            return d
        if isinstance(d, str):
            return cls(text=d)
        return cls(text=d.get("text", ""), title=d.get("title", "") or "", url=d.get("url", "") or "",
                   domain=d.get("domain", "") or "")


@dataclass
class Pruned:
    """What was kept from one document."""
    text: str                                  # the kept sentences, verbatim slices joined with "[…]"
    kept: list[tuple[int, int]]                # (start, end) offsets of the kept sentences in the source
    probs: dict[int, float]                    # P(keep) per sentence index actually read
    n_sentences: int
    n_windows: int                             # windows in the document
    windows_read: int                          # windows the model actually read (cap / stop rule)
    stopped_early: bool
    source: Document = field(repr=False, default=None)

    @property
    def sentences(self) -> list[str]:
        return [self.source.text[a:b] for a, b in self.kept] if self.source else []


def _load_threshold(model_dir: str, kind: str) -> float:
    p = Path(model_dir) / "claimprune.json"
    if p.exists():
        try:
            return float(json.loads(p.read_text())["threshold"])
        except Exception:  # noqa: BLE001
            pass
    return DEFAULT_THRESHOLDS[kind]


def _resolve(model: str) -> str:
    """A local directory as given; otherwise a Hugging Face Hub id, downloaded once."""
    if Path(model).is_dir():
        return model
    from huggingface_hub import snapshot_download
    return snapshot_download(model)


def merge_intervals(spans: list[tuple[int, int]], picks: list[int]) -> list[tuple[int, int]]:
    """The picked sentences' spans, merged where adjacent (radius 0: the cited sentence only)."""
    out: list[list[int]] = []
    for i in sorted(set(picks)):
        a, b = spans[i]
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def cap_document(text: str, ceiling: int = DOC_CEILING) -> str:
    """Trim one document's emission at a sentence boundary, never mid-word."""
    if not ceiling or len(text) <= ceiling:
        return text
    cut = text[:ceiling]
    for sep in (". ", ".\n", "\n\n", "\n", " "):
        i = cut.rfind(sep)
        if i > ceiling * 0.6:
            return cut[:i + len(sep)].rstrip()
    return cut.rstrip()


class Pruner:
    """A distilled claim-conditioned sentence selector.

    model: a Hugging Face Hub id or a local checkpoint directory. The kind (decoder or
    encoder) is read from the checkpoint's config; the keep threshold from its
    `claimprune.json` when present, else the published default for that kind.
    scorer: for tests, a callable (prompt, system, ids) -> P(keep) list instead of a model.
    """

    def __init__(self, model: str | None = None, *, threshold: float | None = None,
                 scorer: Scorer | None = None, device: str | None = None):
        self.system = P.SYSTEM
        if scorer is not None:
            self.scorer = scorer
            self.kind = "custom"
            self.threshold = threshold if threshold is not None else DEFAULT_THRESHOLDS["decoder"]
            return
        if not model:
            raise ValueError("give a model id or directory, or a scorer")
        model_dir = _resolve(model)
        from .encoder import is_encoder_dir
        if is_encoder_dir(model_dir):
            from .encoder import EncoderScorer
            self.kind = "encoder"
            self.scorer = EncoderScorer(model_dir, device=device).score
        else:
            from .decoder import DecoderScorer
            self.kind = "decoder"
            self.scorer = DecoderScorer(model_dir, device=device).score
        self.threshold = threshold if threshold is not None else _load_threshold(model_dir, self.kind)

    # ------------------------------------------------------------------ one document --
    def prune_document(self, claim: str, doc, *, requirements: list[str] | None = None,
                       claim_date: str = "", speaker: str = "", cap_windows: int = 16,
                       stop_after_empty: int = 2, doc_ceiling: int = DOC_CEILING) -> Pruned:
        d = Document.of(doc)
        text = d.text or ""
        wins = _windows(text)
        n_all = len(wins)
        if cap_windows and len(wins) > cap_windows:
            wins = wins[:cap_windows]
        spans = [sp for w in wins for sp in w]
        picks: list[int] = []
        probs: dict[int, float] = {}
        base, empties, read, stopped = 0, 0, 0, False
        for n, w in enumerate(wins, 1):
            prompt = P.render(claim, text, w, base, requirements=requirements, claim_date=claim_date,
                              speaker=speaker, title=d.title, url=d.url, domain=d.domain, part=n, of=len(wins))
            ids = list(range(base, base + len(w)))
            ps = self.scorer(prompt, self.system, ids)
            read += 1
            got = []
            for i, pr in zip(ids, ps):
                probs[i] = float(pr)
                if pr >= self.threshold:
                    got.append(i)
            picks += got
            base += len(w)
            if stop_after_empty:
                empties = 0 if got else empties + 1
                if empties >= stop_after_empty and n < len(wins):
                    stopped = True
                    break
        ivals = merge_intervals(spans, picks) if picks else []
        out = ELISION.join(text[a:b].strip() for a, b in ivals if text[a:b].strip())
        out = cap_document(out, doc_ceiling)
        return Pruned(text=out, kept=[spans[i] for i in sorted(set(picks))], probs=probs,
                      n_sentences=len(spans), n_windows=n_all, windows_read=read, stopped_early=stopped, source=d)

    # ------------------------------------------------------------------ a pool --
    def prune(self, claim: str, documents, **kw) -> list[Pruned]:
        """Prune every document toward the claim; the same keyword options as prune_document."""
        return [self.prune_document(claim, d, **kw) for d in documents]
