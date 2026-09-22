# claimprune

Claim-conditioned context pruning with small distilled models. Give it a claim and a set of
web pages; it returns, for each page, the sentences a fact-checker would need, copied
verbatim from the page.

The two released models are students of a 27B LLM extractor: they were trained to reproduce
its keep-or-drop decision for every sentence of a page, and they reach its judged coverage
and sufficiency on AVeriTeC at a fraction of the cost (see the paper).

| model | size | how it reads | per window (H100) |
|---|---|---|---|
| `claimprune-qwen3.5-2b` | 2B decoder | the full prompt; one forward pass, no generation | 0.14 s |
| `claimprune-modernbert-large` | 0.4B encoder | the prompt's head + the sentences, token classification | 0.016 s |

## Install

```
pip install claimprune            # or: pip install -e .   from this directory
```

Requires Python 3.10+, `torch`, `transformers` (≥ 5.0 for the Qwen3.5 decoder; the encoder
also works on 4.48+) and `huggingface_hub`. A GPU is not required; the encoder is usable on a
CPU.

## Use

```python
from claimprune import Pruner

pruner = Pruner("<org>/claimprune-modernbert-large")   # or a local checkpoint directory
claim = "Wearing face masks will stop the spread of covid 19"
docs = [{"text": open("page1.txt").read(), "title": "Community Use of Masks", "url": "https://..."},
        {"text": open("page2.txt").read()}]

for r in pruner.prune(claim, docs):
    print(r.text)          # the kept sentences, verbatim, "[…]" where text was cut between them
    print(r.kept)          # (start, end) offsets of each kept sentence in the source text
    print(r.windows_read, "of", r.n_windows, "windows read")
```

Optional context, which the models saw in training when it was available:

```python
pruner.prune(claim, docs,
             requirements=["Did the CDC recommend masks in 2020?", "Do masks reduce transmission?"],
             claim_date="2020-10-29", speaker="a Facebook post")
```

Command line:

```
claimprune --model <org>/claimprune-qwen3.5-2b --claim "…" --docs docs.jsonl --out kept.jsonl
claimprune --model <org>/claimprune-modernbert-large --claim "…" --text page.txt
```

`docs.jsonl` holds one document per line: `{"text": ..., "title": ..., "url": ...}`. The output
holds the kept text, the kept sentences with offsets, the per-sentence probabilities and the
reading statistics for every document.

## How it works

1. **Windows.** A page is split into sentences (offsets into the original text; nothing is
   normalised) and cut into windows of at most 48 sentences or 10,000 characters.
2. **Prompt.** Each window is shown with the claim, an optional one-line card (date, speaker),
   an optional document card (title, source, part *n* of *m*), the points to establish, and the
   sentences numbered globally across the page (`S17: …`). The wording is exactly the one the
   teacher labelled with; do not change it.
3. **Scoring.** The decoder appends a decision scaffold for every sentence and reads P(keep)
   from the logits at each slot in one forward pass; the encoder classifies tokens and averages
   over each sentence. A sentence is kept when P(keep) reaches the model's threshold
   (`claimprune.json` in the checkpoint; 0.307 for the 2B, 0.346 for the encoder, set so that
   the student keeps as much as its teacher did on held-out pages).
4. **Reading rule.** At most 16 windows per page are read, and reading stops after two
   consecutive windows in which nothing was kept. Change with `cap_windows` and
   `stop_after_empty` (0 disables either).
5. **Output.** The kept sentences are merged into contiguous slices of the original text and
   joined with `[…]`; one page contributes at most 12,000 characters, cut at a sentence boundary.

Everything above is the code the paper's numbers were produced with, reduced to what
inference needs; `tests/` checks the windowing, the prompt and the reading rule with a stub
scorer and needs no model.

## Models

The checkpoints live on the Hugging Face Hub (`<org>/claimprune-qwen3.5-2b`,
`<org>/claimprune-modernbert-large`); `Pruner("<id>")` downloads them once. Each carries a
model card with the training recipe and a `claimprune.json` with its threshold. Base models:
`Qwen/Qwen3.5-2B` and `answerdotai/ModernBERT-large`.

## Citation

If you use this, cite the AAAI-27 student abstract (reference to be added on acceptance).

## License

Apache-2.0 for the code in this repository. The model weights carry the licences of their base
models (Apache-2.0 for both bases).
