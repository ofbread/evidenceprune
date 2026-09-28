from __future__ import annotations

import json
import urllib.request


def openai_compatible(base_url: str, model: str, max_tokens: int = 700, temperature: float = 0.0,
                      thinking: bool = False, api_key: str = "EMPTY", timeout: float = 900.0):
    url = base_url.rstrip("/") + "/chat/completions"

    def call(system: str, prompt: str) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": thinking}}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"] or ""
    return call


def claude(model: str = "claude-sonnet-5", max_tokens: int = 16000):
    import anthropic
    client = anthropic.Anthropic()

    def call(system: str, prompt: str) -> str:
        kw = {"system": system} if system else {}
        r = client.messages.create(model=model, max_tokens=max_tokens,
                                   messages=[{"role": "user", "content": prompt}], **kw)
        if r.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in r.content if b.type == "text")
    return call
