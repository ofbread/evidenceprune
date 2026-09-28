# evidenceprune

Lightweight models that prune web pages down to the content a fact-checker would need
for a given claim.

Given a claim and one or more pages of plain text, for each page it returns the
sentences to keep. The goal is to turn a set of long, mostly irrelevant
pages into a short evidence pool that is dense in information.

- `evidenceprune-qwen3.5-2b`: a 2B decoder, fine-tuned from Qwen3.5-2B.
- `evidenceprune-modernbert-large`: a 0.4B encoder, fine-tuned from ModernBERT-large.

## Install

```
git clone https://github.com/ofbread/evidenceprune
cd evidenceprune
pip install -e .
```

Fetch a model from Hugging Face:

```
hf download ofbread/evidenceprune-modernbert-large --local-dir checkpoints/evidenceprune-modernbert-large
hf download ofbread/evidenceprune-qwen3.5-2b --local-dir checkpoints/evidenceprune-qwen3.5-2b
```


## Example

A claim and two pages.

```python
from evidenceprune import Pruner

pruner = Pruner("checkpoints/evidenceprune-modernbert-large")

claim = "General Motors sold its Lordstown, Ohio assembly plant to Lordstown Motors in November 2019."

pages = [
    {"title": "GM completes sale of Lordstown plant to electric truck startup",
     "url": "https://www.example-news.com/business/gm-lordstown-sale",
     "text": """Skip to main content
Home  |  Business  |  Autos  |  Subscribe  |  Sign in
GM completes sale of Lordstown plant to electric truck startup
By the auto desk  ·  November 8, 2019
General Motors said on Thursday that it had closed the sale of its idled assembly plant in Lordstown, Ohio, to Lordstown Motors Corp., a startup that plans to build electric pickup trucks there.
The 6.2 million square foot plant built its last Chevrolet Cruze in March 2019, ending more than 50 years of production.
GM did not disclose the price. Lordstown Motors said the deal included a loan of about $40 million from GM.
The startup was founded by Steve Burns, who previously ran Workhorse Group, and says it wants to begin production of its Endurance pickup in late 2020.
Elsewhere in Ohio, the Browns host the Bills on Sunday with kickoff at 1 p.m.
Weather: cloudy, high of 48.
Related: Ford adds shift at Kentucky truck plant · UAW members ratify new contract
Sign up for our newsletter.  Terms of use.  Privacy policy.  © 2019 Example News
"""},
    {"title": "Five electric truck startups to watch in 2020",
     "url": "https://www.example-motoring.org/features/ev-truck-startups",
     "text": """Menu  Reviews  Features  Newsletter
Five electric truck startups to watch in 2020
Rivian has raised more than $2 billion from investors including Amazon and Ford and plans to start deliveries of its R1T pickup in 2020.
Lordstown Motors bought GM's former Lordstown, Ohio assembly plant in November 2019 and plans to build its Endurance pickup there.
Bollinger Motors is developing the boxy B2 pickup in Michigan.
Nikola has shown the Badger, an electric and hydrogen pickup.
Atlis, based in Arizona, is developing the XT pickup.
Read next: the best winter tires of 2020.
Comments are closed.
"""},
]

results = pruner.prune(claim, pages, claim_date="2019-11-20",
                       requirements=["When did GM sell the Lordstown plant, and to whom?",
                                     "What did the plant make before it closed?"])

for page, result in zip(pages, results):
    print("==", page["title"])
    print(result.text)
```

Output of the encoder:

```
== GM completes sale of Lordstown plant to electric truck startup
By the auto desk  ·  November 8, 2019
General Motors said on Thursday that it had closed the sale of its idled assembly plant in Lordstown, Ohio, to Lordstown Motors Corp., a startup that plans to build electric pickup trucks there.
The 6.2 million square foot plant built its last Chevrolet Cruze in March 2019, ending more than 50 years of production.
GM did not disclose the price. Lordstown Motors said the deal included a loan of about $40 million from GM.
== Five electric truck startups to watch in 2020
Lordstown Motors bought GM's former Lordstown, Ohio assembly plant in November 2019 and plans to build its Endurance pickup there.
```

## How it works

A page is split into sentences and read in windows of
48 sentences, at most 10,000 characters each. Each
window becomes a prompt: the claim, its date and speaker if given, the page's title and
source, what must be established, and the numbered sentences. The models are used as scorers instead of autoregressively. The 2B decoder gets the prompt followed by one marker per sentence, and a single
forward pass gives, at each marker, the logits of the tokens `keep` and `drop`. Their softmax
is that sentence's P(keep). The encoder gets the same prompt as plain text and classifies every
token. A sentence's P(keep) is the mean over its tokens. A sentence is kept when P(keep) is at
or above a threshold. A page is read for at most 16 windows, and reading stops
after two consecutive windows where nothing is kept. 

## Input

`Pruner(model, threshold=None, device=None)` loads a model.
`prune(claim, documents, requirements=None, claim_date="", speaker="", cap_windows=16, stop_after_empty=2, doc_ceiling=12000)` reads the pages.

| parameter | type | meaning |
|---|---|---|
| `model` | str | a checkpoint folder, or a Hub id such as `ofbread/evidenceprune-modernbert-large` |
| `claim` | str | the claim being checked |
| `documents` | list of `{"text": str, "title": str, "url": str}` | the pages; `title` and `url` are optional but the model was trained with them; a plain string is accepted as a page |
| `claim_date` | str, optional | when the claim was made, `YYYY-MM-DD` |
| `speaker` | str, optional | who made the claim |
| `requirements` | list of str, optional | what must be established to check the claim |
| `threshold` | float, default from the checkpoint's `evidenceprune.json` (0.3074 for the 2B, 0.3458 for the encoder) | a sentence is kept when its P(keep) is at or above this |
| `cap_windows` | int, default 16 | read at most this many windows of 48 sentences (at most 10,000 characters each) per page |
| `stop_after_empty` | int, default 2 | stop reading a page after this many consecutive windows with nothing kept |
| `doc_ceiling` | int, default 12000 | a page contributes at most this many characters |
| `device` | str, optional | `cuda` or `cpu`; picked automatically when omitted |

## Output

`prune` returns one result per document, in order:

| field | type | meaning |
|---|---|---|
| `text` | str | the kept sentences, verbatim and in page order, gaps marked with `[…]`; empty when nothing was kept |
| `kept` | list of (start, end) | offsets of each kept sentence in the document's `text` |
| `sentences` | list of str | the same sentences as strings |
| `probs` | dict, sentence index → float | P(keep) for every sentence the model read |
| `title_kept` | bool | the title is read as the page's first sentence; whether it was kept |
| `windows_read`, `n_windows` | int | windows the model read, windows in the page |
| `stopped_early` | bool | whether the stop rule ended the read |

## Evaluation

```python
import json
from evidenceprune.evaluation import averitec, claude, ev2r, judge, openai_compatible, verifier

record = json.load(open("dev.json"))[102]          # one claim from the AVeriTeC dev set
pool = [{"text": r.text, "title": p["title"], "url": p["url"]}
        for p, r in zip(pages, results) if r.text]  # pages pruned as in the example above

qwen = openai_compatible("http://localhost:8000/v1", "Qwen/Qwen3.8-27B", max_tokens=700)
v = verifier.verify(qwen, record["claim"], pool, claim_date=averitec.claim_date(record),
                    gold=averitec.gold_verdict(record))
print(v["verdict"], v["score"])                     # e.g. refuted right

grader = openai_compatible("http://localhost:8000/v1", "Qwen/Qwen3.8-27B", max_tokens=60)
print(ev2r.recall(grader, pool, averitec.qa_pairs(record))["recall"])

print(judge.judge(claude(), record["claim"], averitec.questions(record), pool))
```

## License

Apache-2.0
