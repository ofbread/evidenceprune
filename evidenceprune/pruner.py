"""Prune documents to the sentences that are relevant for a claim."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from . import prompt as P
from .sentences import windows as _windows

Scorer = Callable[[str, str, list[int]], list[float]]   # (prompt, system, global ids) -> P(keep) per id

DEFAULT_THRESHOLDS = {"decoder": 0.3073580265045166, "encoder": 0.3458289667963982}
DOC_CEILING = 12_000
ELISION = "\n\n[…]\n\n"


def domain_of(url: str) -> str:
    """The host of a url, lower-cased, without a leading www."""
    try:
        h = (urlparse(url).netloc or "").lower()
    except Exception:  
        return ""
    return h[4:] if h.startswith("www.") else h


@dataclass
class Document:
    text: str
    title: str = ""
    url: str = ""
    domain: str = ""

    def __post_init__(self):
        if not self.domain and self.url:
            self.domain = domain_of(self.url)

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
    text: str                                  # the kept sentences, gaps marked with […]
    kept: list[tuple[int, int]]                # (start, end) of each kept sentence in the document's text
    probs: dict[int, float]                    # sentence index -> P(keep), for the sentences that were read
    n_sentences: int                           # sentences in the document's text
    n_windows: int                             # windows in the document
    windows_read: int                         
    stopped_early: bool
    title_kept: bool = False                   # the title is scored too
    title_prob: float | None = None
    source: Document = field(repr=False, default=None)

    @property
    def sentences(self) -> list[str]:
        return [self.source.text[a:b] for a, b in self.kept] if self.source else []


def _load_threshold(model_dir: str, kind: str) -> float:
    p = Path(model_dir) / "evidenceprune.json"
    if p.exists():
        try:
            return float(json.loads(p.read_text())["threshold"])
        except Exception:  
            pass
    return DEFAULT_THRESHOLDS[kind]


def _resolve(model: str) -> str:
    d = Path(model).expanduser()
    if d.is_dir() and (d / "config.json").exists():
        return str(d)
    if re.fullmatch(r"[\w.-]+/[\w.-]+", model) and not d.exists():
        from huggingface_hub import snapshot_download
        try:
            return snapshot_download(model)
        except Exception as e:  
            raise FileNotFoundError(f"{model} is not a local checkpoint folder, and fetching it from the "
                                    f"Hugging Face Hub failed: {e}") from None
    raise FileNotFoundError(f"{model} is not a checkpoint folder (no config.json) and not a Hub id such as "
                            f"ofbread/evidenceprune-modernbert-large")


def merge_intervals(spans: list[tuple[int, int]], picks: list[int]) -> list[tuple[int, int]]:
    """Kept sentences as (start, end) intervals."""
    out: list[list[int]] = []
    for i in sorted(set(picks)):
        a, b = spans[i]
        if out and a <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def cap_document(text: str, ceiling: int = DOC_CEILING) -> str:

    if not ceiling or len(text) <= ceiling:
        return text
    cut = text[:ceiling]
    for sep in (". ", ".\n", "\n\n", "\n", " "):
        i = cut.rfind(sep)
        if i > ceiling * 0.6:
            return cut[:i + len(sep)].rstrip()
    return cut.rstrip()


class Pruner:

    def __init__(self, model: str | None = None, *, threshold: float | None = None,
                 scorer: Scorer | None = None, device: str | None = None):
        self.system = P.SYSTEM
        if scorer is not None:
            self.scorer = scorer
            self.kind = "custom"
            self.threshold = threshold if threshold is not None else DEFAULT_THRESHOLDS["decoder"]
            return
        if not model:
            raise ValueError("give the checkpoint folder, or a scorer")
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

    # prune one document
    def prune_document(self, claim: str, doc, *, requirements: list[str] | None = None,
                       claim_date: str = "", speaker: str = "", cap_windows: int = 16,
                       stop_after_empty: int = 2, doc_ceiling: int = DOC_CEILING) -> Pruned:
        d = Document.of(doc)
        text = d.text or ""
        # The model reads the title as the first line of the document, so the title is
        # sentence 0 and the page's sentences follow it. Offsets are mapped back below.
        full = f"{d.title}\n{text}"
        off = len(d.title) + 1
        wins = _windows(full)
        n_all = len(wins)
        all_spans = [sp for w in wins for sp in w]
        if cap_windows and len(wins) > cap_windows:
            wins = wins[:cap_windows]
        spans = [sp for w in wins for sp in w]
        picks: list[int] = []
        probs: dict[int, float] = {}
        base, empties, read, stopped = 0, 0, 0, False
        for n, w in enumerate(wins, 1):
            prompt = P.render(claim, full, w, base, requirements=requirements, claim_date=claim_date,
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
        out = ELISION.join(full[a:b].strip() for a, b in ivals if full[a:b].strip())
        out = cap_document(out, doc_ceiling)
        n_title = sum(1 for a, _ in all_spans if a < off)          # the title's sentences (0 or 1)
        kept = [(spans[i][0] - off, spans[i][1] - off) for i in sorted(set(picks)) if spans[i][0] >= off]
        title_probs = [probs[i] for i in range(n_title) if i in probs]
        return Pruned(text=out, kept=kept, probs={i - n_title: p for i, p in probs.items() if i >= n_title},
                      n_sentences=len(all_spans) - n_title, n_windows=n_all, windows_read=read,
                      stopped_early=stopped, title_kept=any(spans[i][0] < off for i in picks),
                      title_prob=max(title_probs) if title_probs else None, source=d)

    # prune a pool of documents
    def prune(self, claim: str, documents, **kw) -> list[Pruned]:
        """Prune every document toward the claim; the same keyword options as prune_document."""
        return [self.prune_document(claim, d, **kw) for d in documents]
