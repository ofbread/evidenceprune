"""The decoder student (Qwen3.5-2B): one forward pass per window, no generation.

The prompt is rendered through the chat template with the generation prompt added and
thinking off; a "decision block" of scaffolds `S<k>:` for every sentence in the window is
appended, and P(keep) for sentence k is read from the logits at the position right after
its scaffold, over the two tokens " keep" and " drop" — exactly where training read them.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

KEEP, DROP, NL = " keep", " drop", "\n"


def block_kernel_packages(names=("fla", "causal_conv1d")) -> list[str]:
    """Veto the optional fast-kernel packages so the Qwen3.5 layers run transformers' pure
    torch paths (the students were trained and evaluated that way; the packages' Hopper
    kernels need CUDA features many machines lack). CLAIMPRUNE_KERNELS=fast leaves them."""
    if os.environ.get("CLAIMPRUNE_KERNELS", "torch").lower() != "torch":
        return []
    blocked = []
    for n in names:
        if n in sys.modules and sys.modules[n] is not None:
            continue
        sys.modules[n] = None
        blocked.append(n)
    return blocked


def as_ids(x) -> list[int]:
    """One flat list of ints from whatever the tokenizer returned (transformers 4 or 5)."""
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
    """Token ids for the two decision words and the scaffolds."""

    def __init__(self, tok):
        self.tok = tok
        self.keep = self._one(KEEP)
        self.drop = self._one(DROP)
        self.nl = as_ids(tok(NL, add_special_tokens=False))
        self._scaf: dict[int, list[int]] = {}

    def _one(self, word: str) -> int:
        ids = as_ids(self.tok(word, add_special_tokens=False))
        if len(ids) != 1:
            raise RuntimeError(f"{word!r} is {len(ids)} tokens in this vocabulary; the decision needs one")
        return ids[0]

    def scaffold(self, k: int) -> list[int]:
        if k not in self._scaf:
            self._scaf[k] = as_ids(self.tok(f"S{k}:", add_special_tokens=False))
        return self._scaf[k]

    def block(self, n: int, base: int) -> tuple[list[int], list[int]]:
        """(ids, targets): scaffolds S<base>… S<base+n-1> and a closing newline; the target
        for sentence k sits on the token AFTER its scaffold (a dummy word id marks it)."""
        ids, tgt = [], []
        for k in range(n):
            s = self.scaffold(base + k)
            ids += s
            tgt += [-100] * len(s)
            if k > 0:
                tgt[len(ids) - len(s)] = self.keep
        ids += self.nl
        tgt += [-100] * len(self.nl)
        if n:
            tgt[len(ids) - len(self.nl)] = self.keep
        return ids, tgt


def prompt_ids(tok, prompt: str, system: str) -> list[int]:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    try:
        out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, enable_thinking=False)
    except TypeError:
        out = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    return as_ids(out)


def _architectures(path: str) -> list[str]:
    try:
        return list(json.loads((Path(path) / "config.json").read_text()).get("architectures") or [])
    except Exception:  # noqa: BLE001
        return []


def load_model(path: str, dtype):
    """Load the checkpoint in its own class: a *ForConditionalGeneration architecture
    (Qwen3.5) through AutoModelForImageTextToText, a plain causal LM through
    AutoModelForCausalLM."""
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
    """P(keep) per sentence of a window, one forward pass, no decoding."""

    def __init__(self, model_dir: str, device: str | None = None):
        block_kernel_packages()
        import torch
        import transformers as T
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = T.AutoTokenizer.from_pretrained(model_dir)
        self.pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        self.pieces = Pieces(self.tok)
        self.model = load_model(model_dir, torch.bfloat16 if self.device == "cuda" else torch.float32).to(self.device)
        self.model.eval()

    def score(self, prompt: str, system: str, ids: list[int]) -> list[float]:
        import torch
        p = prompt_ids(self.tok, prompt, system)
        b, tgt = self.pieces.block(len(ids), ids[0] if ids else 0)
        inp = torch.tensor([p + b], dtype=torch.long, device=self.device)
        att = torch.ones_like(inp)
        labels = torch.tensor([[-100] * len(p) + tgt], dtype=torch.long)
        with torch.no_grad():
            lg = self.model(input_ids=inp, attention_mask=att).logits[:, :-1, :]
        pos = (labels[0, 1:] != -100).nonzero(as_tuple=True)[0].to(lg.device)
        two = lg[0, pos][:, [self.pieces.keep, self.pieces.drop]].float()
        return torch.softmax(two, dim=-1)[:, 0].tolist()
