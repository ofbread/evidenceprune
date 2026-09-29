"""The 0.4B encoder"""
from __future__ import annotations

import json
from pathlib import Path

from .prompt import split_prompt


def is_encoder_dir(model_dir: str) -> bool:
    cfg = Path(model_dir) / "config.json"
    if not cfg.exists():
        return False
    try:
        return "bert" in str(json.loads(cfg.read_text()).get("model_type", "")).lower()
    except Exception:
        return False


def build_text(context: str, sentences: list[str]) -> tuple[str, list[tuple[int, int]]]:
    text = context + "\n\n"
    spans = []
    for i, s in enumerate(sentences):
        if i:
            text += "\n"
        a = len(text)
        text += s
        spans.append((a, len(text)))
    return text, spans


def encode(tok, context: str, sentences: list[str], max_len: int) -> dict:
    text, spans = build_text(context, sentences)
    owner = [-1] * (len(text) + 1)
    for i, (a, b) in enumerate(spans):
        for c in range(a, b):
            owner[c] = i
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    ids, offs = list(enc["input_ids"]), list(enc["offset_mapping"])
    sent_of = [owner[a] if b > a else -1 for a, b in offs]
    if len(ids) > max_len:                       # keep the head, close with [SEP]
        ids = ids[:max_len - 1] + [tok.sep_token_id]
        sent_of = sent_of[:max_len - 1] + [-1]
    return {"input_ids": ids, "sent_of": sent_of, "n_sentences": len(sentences)}


class EncoderScorer:
    def __init__(self, model_dir: str, device: str | None = None, max_len: int = 8192):
        import torch
        import transformers as T
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = T.AutoTokenizer.from_pretrained(model_dir)
        dt = torch.bfloat16 if self.device == "cuda" else None
        kw = {"num_labels": 2}
        try:
            self.model = T.AutoModelForTokenClassification.from_pretrained(model_dir, **({"dtype": dt} if dt else {}), **kw)
        except TypeError:
            self.model = T.AutoModelForTokenClassification.from_pretrained(model_dir, torch_dtype=dt, **kw)
        self.model.to(self.device).eval()
        self.max_len = max_len

    def score(self, prompt: str, system: str, ids: list[int]) -> list[float]:
        import torch
        context, sents = split_prompt(prompt)
        e = encode(self.tok, context, sents, self.max_len)
        inp = torch.tensor([e["input_ids"]], dtype=torch.long, device=self.device)
        att = torch.ones_like(inp)
        with torch.no_grad(), torch.autocast(device_type="cuda" if self.device == "cuda" else "cpu",
                                             dtype=torch.bfloat16, enabled=(self.device == "cuda")):
            logits = self.model(input_ids=inp, attention_mask=att).logits
        p_tok = torch.softmax(logits.float(), dim=-1)[0, :, 1].tolist()
        acc = [0.0] * e["n_sentences"]
        cnt = [0] * e["n_sentences"]
        for t, s in enumerate(e["sent_of"]):
            if s >= 0:
                acc[s] += p_tok[t]
                cnt[s] += 1
        return [acc[s] / cnt[s] if cnt[s] else 0.0 for s in range(e["n_sentences"])]
