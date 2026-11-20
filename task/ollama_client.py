"""Call Ollama HTTP API (no extra deps)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Callable, Iterator


def _build_payload(
    model: str,
    prompt: str,
    system: str | None,
    *,
    stream: bool,
    keep_alive: str | None,
    num_predict: int = 512,
    num_ctx: int = 4096,
    temperature: float = 0.4,
    stop: list[str] | None = None,
) -> dict:
    payload: dict = {"model": model, "prompt": prompt, "stream": stream}
    if system:
        payload["system"] = system
    if keep_alive:
        payload["keep_alive"] = keep_alive
    options: dict = {
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "temperature": temperature,
        "top_p": 0.9,
    }
    if stop:
        options["stop"] = stop
    payload["options"] = options
    return payload


def generate(
    base_url: str,
    model: str,
    prompt: str,
    system: str | None = None,
    timeout_s: float = 120.0,
    keep_alive: str | None = "30m",
    num_predict: int = 512,
    temperature: float | None = None,
    num_ctx: int | None = None,
) -> str:
    base = base_url.rstrip("/")
    t = 0.4 if temperature is None else temperature
    ctx = 4096 if num_ctx is None else num_ctx
    payload = _build_payload(
        model,
        prompt,
        system,
        stream=False,
        keep_alive=keep_alive,
        num_predict=num_predict,
        num_ctx=ctx,
        temperature=t,
    )
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach Ollama at {base}: {e}") from e
    return str(body.get("response", "")).strip()


def generate_stream(
    base_url: str,
    model: str,
    prompt: str,
    system: str | None = None,
    timeout_s: float | None = None,
    keep_alive: str | None = "30m",
    num_predict: int = 512,
    temperature: float | None = None,
    num_ctx: int | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> Iterator[str]:
    """Yield text chunks as Ollama generates them.

    timeout_s: max seconds for low-level socket ops; None = no limit between chunks.
    temperature / num_ctx: optional overrides (defaults match _build_payload).
    cancel_check: optional callable returning True to stop reading (client disconnect).
    """
    base = base_url.rstrip("/")
    t = 0.4 if temperature is None else temperature
    ctx = 4096 if num_ctx is None else num_ctx
    payload = _build_payload(
        model,
        prompt,
        system,
        stream=True,
        keep_alive=keep_alive,
        num_predict=num_predict,
        num_ctx=ctx,
        temperature=t,
    )
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout_s)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach Ollama at {base}: {e}") from e
    try:
        for raw in resp:
            if cancel_check is not None and cancel_check():
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk = obj.get("response") or ""
            if chunk:
                yield chunk
            if obj.get("done"):
                break
    finally:
        try:
            resp.close()
        except Exception:
            pass


def warmup(
    base_url: str,
    model: str,
    keep_alive: str | None = "30m",
    timeout_s: float = 60.0,
) -> None:
    """Best-effort: load the model into memory so the first prompt is fast."""
    base = base_url.rstrip("/")
    payload = {"model": model, "prompt": " ", "stream": False, "keep_alive": keep_alive}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=timeout_s).read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return


def is_reachable(base_url: str, timeout_s: float = 1.5) -> bool:
    base = base_url.rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=timeout_s) as resp:
            return resp.status == 200
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return False


def list_loaded_models(base_url: str, timeout_s: float = 2.0) -> list[str]:
    """Models currently loaded in Ollama RAM (via /api/ps)."""
    base = base_url.rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/ps", timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return []
    out: list[str] = []
    for m in body.get("models") or []:
        name = m.get("name") or m.get("model")
        if name:
            out.append(str(name))
    return out


def list_models(base_url: str, timeout_s: float = 3.0) -> list[str]:
    base = base_url.rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError):
        return []
    out: list[str] = []
    for m in body.get("models") or []:
        name = m.get("name") or m.get("model")
        if name:
            out.append(str(name))
    return out
