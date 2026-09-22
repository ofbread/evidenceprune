---
license: apache-2.0
base_model: <Qwen/Qwen3.5-2B | answerdotai/ModernBERT-large>
language: en
tags: [fact-checking, context-pruning, evidence-retrieval, distillation]
---

# claimprune-<qwen3.5-2b | modernbert-large>

A claim-conditioned sentence selector for open-web fact-checking: given a claim and a web
page, it marks the sentences a fact-checker would need. It is a student of a 27B LLM
extractor (Qwen3.8-27B), trained to reproduce the teacher's keep-or-drop decision for every
sentence of a page. Use it through the `claimprune` package:

```python
from claimprune import Pruner
kept = Pruner("<this repo id>").prune(claim, documents)
```

## Training

- **Labels:** the 27B extractor's decisions under the production prompt over pages from the
  AVeriTeC training knowledge store (no evaluation claim among them): 4,289 pages / 15,653
  windows (the 2B); 4,923 pages / 23,827 windows (the encoder). Windows with nothing to keep
  were kept as training data at half weight.
- **2B decoder:** Qwen3.5-2B, full fine-tune, one epoch, lr 2e-5, effective batch 32, sequences
  to 4,096 tokens, bf16, thinking off; the loss is a two-way cross-entropy over the tokens
  " keep" / " drop" at each sentence's decision slot. Threshold 0.3074, set on held-out store
  claims so that the student keeps as much as the teacher.
- **0.4B encoder:** ModernBERT-large, token classification, three epochs, lr 3e-5, batch 32,
  sequences to 8,192 tokens; a sentence's P(keep) is the mean over its tokens. Threshold 0.3458.

## Evaluation (AVeriTeC dev, 120 claims, the benchmark's evidence-date rule)

| pool | verdicts right / 120 | Ev2R | judged sufficiency | coverage | relevance |
|---|---|---|---|---|---|
| teacher (27B, cap & stop) | 77 | 0.483 | 0.425 | 0.527 | 0.716 |
| 2B student | 73 | 0.477 | 0.458 | 0.555 | 0.643 |
| 0.4B encoder | 74 | 0.469 | 0.400 | 0.539 | 0.655 |
| start of page (positional) | 67 | 0.386 | 0.292 | 0.434 | 0.470 |
| BM25 + queries (lexical) | 68 | 0.446 | 0.375 | 0.479 | 0.544 |
| Provence 0.1 (QA pruner) | 58 | 0.433 | 0.317 | 0.457 | 0.638 |

Compression per claim on 50 new claims (about 12 pages, 390k characters, H100): 2B 3.8 s,
encoder 0.4 s, teacher 8.6 s.

## Intended use and limits

For selecting evidence sentences from web pages toward a claim before a verifier reads them.
Trained and evaluated on English news-style pages and 2020-era claims; the decision is a
judgement of relevance to the claim, not of truth. The students copy the teacher's mistakes
and add their own: the encoder sometimes cuts one sentence short of the deciding passage, the
2B sometimes keeps a quote without its speaker. See the paper for the full evaluation.

## Files

`config.json`, the weights, the tokenizer, and `claimprune.json` (`{"threshold": …}`).
