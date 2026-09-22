"""claimprune — prune documents toward a claim from the command line.

    claimprune --model <id-or-dir> --claim "…" --docs docs.jsonl [--out kept.jsonl]
    claimprune --model <id-or-dir> --claim "…" --text page.txt

docs.jsonl: one JSON object per line with "text" and optionally "title", "url", "domain".
Output: one JSON object per document with the kept text, the kept sentences with offsets,
the per-sentence P(keep) and the reading statistics.
"""
from __future__ import annotations

import argparse
import json
import sys

from .pruner import Pruner


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="claimprune", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="Hugging Face id or local checkpoint directory")
    ap.add_argument("--claim", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--docs", help="JSONL of documents ({text, title?, url?, domain?})")
    src.add_argument("--text", help="one document as a text file")
    ap.add_argument("--requirements", nargs="*", default=None, help="what must be established (optional)")
    ap.add_argument("--claim-date", default="")
    ap.add_argument("--speaker", default="")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--cap-windows", type=int, default=16)
    ap.add_argument("--stop-after-empty", type=int, default=2)
    ap.add_argument("--out", default="", help="JSONL output (default: stdout)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    if a.docs:
        docs = [json.loads(l) for l in open(a.docs, encoding="utf-8") if l.strip()]
    else:
        docs = [{"text": open(a.text, encoding="utf-8").read(), "title": a.text}]
    pr = Pruner(a.model, threshold=a.threshold, device=a.device)
    out = open(a.out, "w", encoding="utf-8") if a.out else sys.stdout
    for d, r in zip(docs, pr.prune(a.claim, docs, requirements=a.requirements, claim_date=a.claim_date,
                                   speaker=a.speaker, cap_windows=a.cap_windows, stop_after_empty=a.stop_after_empty)):
        rec = {"title": d.get("title", ""), "url": d.get("url", ""), "kept_text": r.text,
               "kept": [{"start": s, "end": e, "text": r.source.text[s:e]} for s, e in r.kept],
               "probs": {str(k): round(v, 4) for k, v in r.probs.items()},
               "n_sentences": r.n_sentences, "n_windows": r.n_windows, "windows_read": r.windows_read,
               "stopped_early": r.stopped_early}
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if a.out:
        out.close()
        print(f"wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
