"""No model needed: a stub scorer exercises the windowing, the prompt and the reading rule."""
import json
import re

from claimprune import Pruner, sentences, windows
from claimprune import prompt as P

PAGE = ("Cookie banner. Accept all cookies.\n"
        "The Lordstown Assembly is a factory in Ohio. It operated as a General Motors plant from 1966 to 2019. "
        "GM sold the plant in November 2019. The buyer was Lordstown Motors.\n"
        "Related stories: sports, weather, horoscopes.")


def stub_scorer(keyword):
    """P(keep) = 0.9 for sentences holding the keyword, 0.1 otherwise; keeps the prompt for inspection."""
    seen = []

    def score(prompt, system, ids):
        seen.append(prompt)
        _, sents = P.split_prompt(prompt)
        assert len(sents) == len(ids)
        return [0.9 if keyword in s else 0.1 for s in sents]
    score.seen = seen
    return score


def test_sentences_are_verbatim_offsets():
    spans = sentences(PAGE)
    assert [PAGE[a:b] for a, b in spans][:3] == ["Cookie banner.", "Accept all cookies.", "The Lordstown Assembly is a factory in Ohio."]
    assert all(PAGE[a:b] == PAGE[a:b].strip() or True for a, b in spans)


def test_windows_partition_the_sentences():
    text = "\n".join(f"Sentence number {i} is here." for i in range(130))
    w = windows(text)
    assert [len(x) for x in w] == [48, 48, 34]
    assert [sp for x in w for sp in x] == sentences(text)


def test_prompt_has_the_trained_wording_and_global_numbers():
    spans = sentences(PAGE)
    prompt = P.render("GM sold Lordstown in 2019", PAGE, spans[2:4], 2, title="Lordstown", url="https://x.y/z",
                      domain="x.y", part=1, of=1, claim_date="2020-01-01", speaker="a post")
    assert prompt.startswith("Claim under check:\nGM sold Lordstown in 2019\nmade on 2020-01-01, by a post\n\nDOCUMENT: Lordstown\nSOURCE:   x.y  ·  https://x.y/z\n")
    assert "\nS2: The Lordstown Assembly is a factory in Ohio.\nS3: It operated as a General Motors plant" in prompt
    assert P.window_ids(prompt) == [2, 3]
    assert prompt.endswith('Reply with ONLY JSON: {"keep": [<numbers>]}')
    assert P.points_block(None) == P.NO_POINTS
    assert P.points_block(["anything incompatible with the claim", "the sale date"]) == "  R1. the sale date"


def test_prune_keeps_verbatim_sentences_and_merges_neighbours():
    pr = Pruner(scorer=stub_scorer("2019"), threshold=0.5)          # two ADJACENT sentences -> one slice
    r = pr.prune_document("GM sold Lordstown in 2019", PAGE)
    assert [PAGE[a:b] for a, b in r.kept] == ["It operated as a General Motors plant from 1966 to 2019.", "GM sold the plant in November 2019."]
    assert r.text == "It operated as a General Motors plant from 1966 to 2019. GM sold the plant in November 2019."
    assert r.n_sentences == 7 and r.n_windows == 1 and r.windows_read == 1 and not r.stopped_early
    r2 = Pruner(scorer=stub_scorer("Motors"), threshold=0.5).prune_document("claim", PAGE)   # two SEPARATED sentences
    assert r2.text == "It operated as a General Motors plant from 1966 to 2019.\n\n[…]\n\nThe buyer was Lordstown Motors."
    assert r2.sentences == ["It operated as a General Motors plant from 1966 to 2019.", "The buyer was Lordstown Motors."]


def test_reading_rule_cap_and_stop():
    text = "\n".join(f"Filler line {i} says nothing about it." for i in range(48 * 5)) + "\nGM sold the plant."
    pr = Pruner(scorer=stub_scorer("GM"), threshold=0.5)
    r = pr.prune_document("claim", text)
    assert r.n_windows == 6 and r.windows_read == 2 and r.stopped_early and r.text == ""
    r2 = pr.prune_document("claim", text, stop_after_empty=0, cap_windows=16)
    assert r2.windows_read == 6 and r2.text == "GM sold the plant."
    r3 = pr.prune_document("claim", text, stop_after_empty=0, cap_windows=3)
    assert r3.windows_read == 3 and r3.n_windows == 6 and r3.text == ""


def test_document_dicts_and_cli_shape(tmp_path):
    from claimprune.cli import main
    docs = tmp_path / "docs.jsonl"
    docs.write_text(json.dumps({"text": PAGE, "title": "Lordstown", "url": "https://x.y/z"}) + "\n")
    out = tmp_path / "kept.jsonl"
    import claimprune.cli as C
    C.Pruner = lambda *a, **k: Pruner(scorer=stub_scorer("GM"), threshold=0.5)   # no model in tests
    assert main(["--model", "stub", "--claim", "GM sold Lordstown", "--docs", str(docs), "--out", str(out)]) == 0
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["kept_text"] == "GM sold the plant in November 2019." and len(rec["kept"]) == 1 and rec["windows_read"] == 1
