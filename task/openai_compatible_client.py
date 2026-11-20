"""OpenAI-compatible Chat Completions HTTP API (stdlib only).

Works with Groq, OpenRouter, Mistral AI, Together, Fireworks, any /v1/chat/completions server.
See README or env: KB_LLM_BACKEND=openai_compatible
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Iterator

# Groq and other hosts sit behind Cloudflare; the default Python-urllib User-Agent is often blocked (HTTP 403, code 1010).
def _request_headers(api_key: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "KBCopilot/1.0",
    }


def _messages(system: str | None, user: str) -> list[dict]:
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    out.append({"role": "user", "content": user})
    return out


def chat_complete(
    base_url: str,
    api_key: str,
    model: str,
    user: str,
    system: str | None = None,
    *,
    timeout_s: float = 120.0,
    temperature: float = 0.5,
    max_tokens: int = 1024,
) -> str:
    """Non-streaming chat completion."""
    base = base_url.rstrip("/")
    url = f"{base}/chat/completions"
    payload = {
        "model": model,
        "messages": _messages(system, user),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=_request_headers(api_key), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"LLM HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach LLM API at {base}: {e}") from e
    try:
        return str(body["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected LLM response: {body!r}") from e


def chat_stream(
    base_url: str,
    api_key: str,
    model: str,
    user: str,
    system: str | None = None,
    *,
    timeout_s: float | None = 120.0,
    temperature: float = 0.5,
    max_tokens: int = 1024,
) -> Iterator[str]:
    """Streaming chat completion; yields text deltas."""
    base = base_url.rstrip("/")
    url = f"{base}/chat/completions"
    payload = {
        "model": model,
        "messages": _messages(system, user),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=_request_headers(api_key), method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout_s)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"LLM HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach LLM API at {base}: {e}") from e
    try:
        buf = b""
        while True:
            chunk = resp.read(1024)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line_s = line.decode("utf-8", errors="replace").strip()
                if not line_s.startswith("data:"):
                    continue
                data_str = line_s[5:].strip()
                if data_str == "[DONE]":
                    return
                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    if piece:
                        yield str(piece)
    finally:
        try:
            resp.close()
        except Exception:
            pass
