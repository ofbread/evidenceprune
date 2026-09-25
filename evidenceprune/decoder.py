"""The 2B decoder (Qwen3.5-2B)"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

KEEP, DROP, NL = " keep", " drop", "\n"


def block_kernel_packages(names=("fla", "causal_conv1d")) -> list[str]:
    """Make transformers use its plain PyTorch implementation of the Qwen3.5 layers."""
    if os.environ.get("EVIDENCEPRUNE_KERNELS", "torch").lower() != "torch":
        return []
    blocked = []
    for n in names:
        if n in sys.modules and sys.modules[n] is not None:
            warnings.warn(f"{n} is already imported; the fast kernels stay on")  
            continue
        sys.modules[n] = None
        blocked.append(n)
    return blocked


def as_ids(x) -> list[int]:
    """Turn the tokenizer's output into a flat list of ints."""
    if hasattr(x, "keys") and "input_ids" in x:
        x = x["input_ids"]
    if hasattr(x, "tolist"):
        x = x.tolist()
    x = list(x)
    if x and isinstance(x[0], (list, tuple)):
        if len(x) != 1:
            raise ValueError(f"expected one sequence, got a batch of {len(x)}")
        x = list(x[0])
    return [int(t) for t in x]


class Pieces:

    def __init__(self, tok):
        self.tok = tok
        self.keep = self._one(KEEP)       
        self.drop = self._one(DROP)       
        self.nl = as_ids(tok(NL, add_special_tokens=False))
        self._markers: dict[int, list[int]] = {}

    def _one(self, word: str) -> int:
        ids = as_ids(self.tok(word, add_special_tokens=False))
        if len(ids) != 1:
            raise RuntimeError(f"{word!r} is {len(ids)} tokens in this vocabulary. The decision needs one")
        return ids[0]

    def marker(self, k: int) -> list[int]:
        """Token ids of "S<k>:", cached."""
        if k not in self._markers:
            self._markers[k] = as_ids(self.tok(f"S{k}:", add_special_tokens=False))
        return self._markers[k]

    def block(self, n: int, base: int) -> tuple[list[int], list[int]]:
        """The decision block for n sentences numbered from `base`."""
        ids, pos = [], []
        for k in range(n):
            m = self.marker(base + k)
            ids += m
            pos += [-100] * len(m)
            if k > 0:
                pos[len(ids) - len(m)] = self.keep       # sentence k-1 is decided at the start of marker k
        ids += self.nl
        pos += [-100] * len(self.nl)
        if n:
            pos[len(ids) - len(self.nl)] = self.keep     # the last sentence is decided at the newline
        return ids, pos


def prompt_ids(tok, prompt: str, system: str) -> list[int]:
    """Tokenize system + prompt through the model's chat template, with the assistant turn
    opened and thinking off"""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    try:
        out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, enable_thinking=False)
    except TypeError:
        out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    return as_ids(out)


def _architectures(path: str) -> list[str]:
    try:
        return list(json.loads((Path(path) / "config.json").read_text()).get("architectures") or [])
    except Exception:  
        return []


def load_model(path: str, dtype):

    import transformers as T
    archs = _architectures(path)
    names = (("AutoModelForImageTextToText",) if any(a.endswith("ForConditionalGeneration") for a in archs)
             else ("AutoModelForCausalLM", "AutoModelForImageTextToText"))
    last = None
    for name in names:
        cls = getattr(T, name, None)
        if cls is None:
            continue
        try:
            try:
                return cls.from_pretrained(path, dtype=dtype)
            except TypeError:
                return cls.from_pretrained(path, torch_dtype=dtype)
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"could not load {path}: {last}")


class DecoderScorer:
    """Scores a window"""

    def __init__(self, model_dir: str, device: str | None = None):
        block_kernel_packages()
        import torch
        import transformers as T
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = T.AutoTokenizer.from_pretrained(model_dir)
        self.pieces = Pieces(self.tok)
        self.model = load_model(model_dir, torch.bfloat16 if self.device == "cuda" else torch.float32).to(self.device)
        self.model.eval()

    def score(self, prompt: str, system: str, ids: list[int]) -> list[float]:

        import torch
        p = prompt_ids(self.tok, prompt, system)                          
        b, pos = self.pieces.block(len(ids), ids[0] if ids else 0)         
        inp = torch.tensor([p + b], dtype=torch.long, device=self.device)
        att = torch.ones_like(inp)
        marks = torch.tensor([[-100] * len(p) + pos], dtype=torch.long)    
        with torch.no_grad():
            logits = self.model(input_ids=inp, attention_mask=att).logits[:, :-1, :]   # logits[t] predicts token t+1
        where = (marks[0, 1:] != -100).nonzero(as_tuple=True)[0].to(logits.device)
        two = logits[0, where][:, [self.pieces.keep, self.pieces.drop]].float()      # only " keep" vs " drop"
        return torch.softmax(two, dim=-1)[:, 0].tolist()
