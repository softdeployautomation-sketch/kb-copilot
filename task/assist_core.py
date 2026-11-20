"""
Shared assist logic: retrieve from KB, call Ollama with grounded + optional general fallback.
"""

from __future__ import annotations

import os
import re
from typing import Any

from ollama_client import generate as ollama_generate
from openai_compatible_client import chat_complete
from retrieval import BM25Index, tokenize


def _use_cloud_llm() -> bool:
    return (os.environ.get("KB_LLM_BACKEND") or "").strip().lower() in (
        "openai_compatible",
        "cloud",
        "groq",
        "openai",
    )


def normalize_llm_source(raw: str | None) -> str | None:
    """UI / API: None = follow .env default; 'ollama' | 'cloud' = force path."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s or s in ("env", "auto", "default", "."):
        return None
    if s in ("ollama", "local"):
        return "ollama"
    if s in ("cloud", "openai_compatible", "groq", "openai", "api"):
        return "cloud"
    return None


def use_cloud_for_llm_source(llm_source: str | None) -> bool:
    """True = use OpenAI-compatible HTTP API; False = Ollama."""
    n = normalize_llm_source(llm_source)
    if n == "ollama":
        return False
    if n == "cloud":
        return True
    return _use_cloud_llm()


def _cloud_chat_params() -> tuple[str, str, str]:
    key = (os.environ.get("OPENAI_COMPAT_API_KEY") or os.environ.get("GROQ_API_KEY") or "").strip()
    base = (os.environ.get("OPENAI_COMPAT_BASE_URL") or "https://api.groq.com/openai/v1").rstrip("/")
    model = (
        os.environ.get("OPENAI_COMPAT_QUICK_MODEL")
        or os.environ.get("GROQ_MODEL")
        or "llama-3.1-8b-instant"
    ).strip()
    return base, key, model


def resolve_cloud_quick_model(override: str | None) -> str:
    o = (override or "").strip()
    if o:
        return o
    return (
        os.environ.get("OPENAI_COMPAT_QUICK_MODEL")
        or os.environ.get("GROQ_MODEL")
        or "llama-3.1-8b-instant"
    ).strip()


def resolve_cloud_research_model(override: str | None) -> str:
    o = (override or "").strip()
    if o:
        return o
    return (
        os.environ.get("OPENAI_COMPAT_RESEARCH_MODEL")
        or os.environ.get("OPENAI_COMPAT_QUICK_MODEL")
        or os.environ.get("GROQ_MODEL")
        or "llama-3.1-8b-instant"
    ).strip()


def getenv_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def kb_is_weak(hits: list[dict], threshold: float | None = None) -> tuple[bool, float]:
    """Heuristic: low top BM25 score ⇒ KB probably missing the answer."""
    if not hits:
        return True, 0.0
    top = float(hits[0].get("score") or 0.0)
    thr = threshold if threshold is not None else getenv_float("KB_WEAK_SCORE", 0.12)
    return top < thr, top


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def best_kb_snippet(question: str, hits: list[dict]) -> dict | None:
    """Pick the best KB chunk paragraph and the sentence inside it that overlaps the question most.

    Returns {source, paragraph, match_sentence, sentence, score} or None.
    ``sentence`` duplicates ``match_sentence`` for backward compatibility.
    """
    if not hits:
        return None
    q_tokens = set(tokenize(question))
    if not q_tokens:
        return None

    best: dict | None = None

    def score_text(text: str) -> int:
        if not text:
            return 0
        toks = tokenize(text)
        if not toks:
            return 0
        bag = {}
        for t in toks:
            bag[t] = bag.get(t, 0) + 1
        return sum(bag.get(t, 0) for t in q_tokens)

    for h in hits[:5]:
        src = h.get("source", "")
        text = h.get("text", "") or ""
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            paragraphs = [text]
        for para in paragraphs:
            p_score = score_text(para)
            if p_score <= 0:
                continue
            sentences = [s.strip() for s in _SENT_SPLIT.split(para) if s.strip()]
            picked_sent = ""
            picked_sent_score = -1
            for sent in sentences:
                s_score = score_text(sent)
                bonus = 0.5 if 40 <= len(sent) <= 220 else 0.0
                total = s_score + bonus
                if total > picked_sent_score:
                    picked_sent_score = total
                    picked_sent = sent
            cand = {
                "source": src,
                "paragraph": para,
                "match_sentence": picked_sent or para,
                "score": p_score,
            }
            if best is None or cand["score"] > best["score"]:
                best = cand

    if best is not None:
        best["sentence"] = best["match_sentence"]
    return best


def build_system_prompt(mode: str, *, allow_general_knowledge: bool, strict_kb: bool = False) -> str:
    mode = (mode or "interview").strip().lower()
    grounded = (
        "User-specific facts (employers, dates, metrics, tool names tied to the user) must "
        "only come from CONTEXT or SESSION BRIEF when they are stated there."
    )
    strict = ""
    if strict_kb:
        strict = (
            "STRICT_GROUNDING: For projects, clients, products, and concrete outcomes, only use what "
            "appears in CONTEXT or SESSION BRIEF. Do not substitute a plausible generic or alternate "
            "project story. If the question references work not named in CONTEXT, say your notes do "
            "not contain that project and answer from CONTEXT only. "
        )
    general = (
        "When CONTEXT does not cover the question, you may use careful general knowledge for "
        "frameworks, definitions, and how-to structure an answer. Never invent the user's "
        "personal history. If unsure, say what you would verify before claiming it as your own."
    )
    general_strict_kb = (
        "When CONTEXT is thin, use general knowledge only for framing and interview technique—not "
        "to fabricate the user's work history or substitute missing project details. "
    )
    tight = (
        "If CONTEXT does not contain enough detail, say what is missing in one sentence, then "
        "offer a neutral framework without fabricating user-specific facts."
    )

    if mode == "meeting":
        base = (
            "You help during live meetings. Be concise and professional. "
            + strict
            + grounded
            + " "
        )
        geo = general_strict_kb + general if (allow_general_knowledge and strict_kb) else general
        return base + (geo if allow_general_knowledge else tight)
    if mode == "general":
        base = "You are a personal knowledge assistant. " + strict + grounded + " "
        geo = general_strict_kb + general if (allow_general_knowledge and strict_kb) else general
        return base + (geo if allow_general_knowledge else tight)
    # interview
    base = (
        "You help during live interviews. Answers must sound natural and conversational when "
        "requested. Be honest in spirit: do not claim credentials that are not supported by "
        "CONTEXT or SESSION BRIEF. "
        + strict
        + grounded
        + " "
    )
    geo = general_strict_kb + general if (allow_general_knowledge and strict_kb) else general
    return base + (geo if allow_general_knowledge else tight)


def trim_hits_for_prompt(
    hits: list[dict],
    *,
    total_budget: int = 7000,
    per_hit_max: int = 2800,
) -> list[dict]:
    """Shallow-copy hits with truncated text so Ollama sees a smaller, faster prompt."""
    if total_budget <= 0 or not hits:
        return list(hits)
    out: list[dict] = []
    budget = total_budget
    for h in hits:
        if budget <= 0:
            break
        text = (h.get("text") or "").strip()
        if not text:
            continue
        cap = min(per_hit_max, budget, len(text))
        chunk = text[:cap]
        if len(text) > cap:
            chunk = chunk.rstrip() + "…"
        nh = dict(h)
        nh["text"] = chunk
        out.append(nh)
        budget -= cap
    return out if out else list(hits[:1])


def kb_fallback_reply(
    hits: list[dict],
    best: dict | None,
    *,
    want_spoken: bool,
    question: str,
    use_cloud_llm: bool | None = None,
    emit_verbatim_kb_excerpt: bool | None = None,
) -> str:
    """When the configured LLM yields nothing: help text; optional verbatim (avoid duplicating green KB panel)."""
    parts: list[str] = []
    cloud = use_cloud_llm if use_cloud_llm is not None else _use_cloud_llm()
    if cloud:
        parts.append("— Model did not return an answer (cloud path) —")
    else:
        parts.append("— Model did not return an answer (Ollama path) —")

    if emit_verbatim_kb_excerpt is None:
        emit_verbatim_kb_excerpt = best is None

    if not emit_verbatim_kb_excerpt and best:
        parts.append(
            "\nThe **Verbatim from your KB** panel above already shows the retrieved quote (not generated by the model). "
            "This box will contain a rephrased / spoken-style draft once the LLM responds successfully."
        )
    elif best:
        para = (best.get("paragraph") or "").strip()
        line = (best.get("match_sentence") or best.get("sentence") or "").strip()
        if line:
            parts.append("\nSAY THIS (verbatim from your KB):\n" + line)
        if para and para != line and len(para) > len(line) + 40:
            extra = para[:2000] + ("…" if len(para) > 2000 else "")
            parts.append("\nFull KB excerpt:\n" + extra)
    elif hits:
        t0 = (hits[0].get("text") or "").strip()
        if t0:
            parts.append("\nTop KB excerpt:\n" + (t0[:2200] + ("…" if len(t0) > 2200 else "")))
    if cloud:
        parts.append(
            "\n\nTip: Groq blocks the default Python User-Agent — this build sends a real UA. If errors persist, "
            "check the API key, `OPENAI_COMPAT_BASE_URL`, and model id in `.env`, then restart the app."
        )
    else:
        parts.append(
            "\n\nTip: open Ollama, click Warm up until the green Model ready dot shows, then try again."
        )
    return "\n".join(parts).strip()


def build_user_prompt(
    mode: str,
    question: str,
    session_brief: str,
    hits: list[dict],
    want_spoken: bool,
    *,
    kb_weak: bool,
    top_kb_score: float,
    prev_question: str | None = None,
    prev_reply: str | None = None,
    strict_kb: bool = False,
) -> str:
    ctx_parts = []
    for h in hits:
        ctx_parts.append(f"--- SOURCE: {h['source']}\n{h['text']}\n")
    ctx = "\n".join(ctx_parts) if ctx_parts else "(no excerpts retrieved)\n"
    brief = (session_brief or "").strip()
    brief_block = f"SESSION BRIEF:\n{brief}\n\n" if brief else ""
    weak_hint = (
        "If kb_weak=yes, lean on SESSION BRIEF (if present) and safe general framing.\n\n"
        if not strict_kb
        else (
            "If kb_weak=yes, CONTEXT may be incomplete—say so briefly; still do not invent "
            "user-specific projects or employers beyond SESSION BRIEF + CONTEXT.\n\n"
        )
    )
    strict_hint = ""
    if strict_kb:
        strict_hint = (
            "GROUNDING: Name only projects, tools, and outcomes that appear in CONTEXT or SESSION BRIEF. "
            "If the QUESTION asks about something not in CONTEXT, state that explicitly before any bullets.\n\n"
        )
    coverage = (
        f"KB SIGNAL: top_retrieval_score={top_kb_score:.4f}; "
        f"kb_weak={'yes' if kb_weak else 'no'}.\n"
        + weak_hint
        + strict_hint
    )
    spoken = (
        "Output format:\n"
        "1) Key bullets (max 6 short bullets)\n"
        "2) SAY THIS: a 20–45 second spoken answer in first person (unless MODE implies otherwise)\n"
    )
    not_spoken = (
        "Output format:\n"
        "1) Key bullets (max 8)\n"
        "2) Short paragraph summary\n"
    )
    pq = (prev_question or "").strip()
    pr = (prev_reply or "").strip()
    anti_echo = ""
    if pq and pr:
        same = pq.lower() == (question or "").strip().lower()
        if same:
            anti_echo = (
                "ANTI-ECHO: This QUESTION matches the prior turn. Do not repeat the same sentences or bullet order. "
                "Keep facts aligned with CONTEXT only; restructure and rephrase.\n"
                f"Prior answer (do not parrot):\n{pr[:1800]}\n\n"
            )
        else:
            anti_echo = (
                "PRIOR TURN (context only; answer the current QUESTION):\n"
                f"Earlier Q: {pq[:600]}\nEarlier A: {pr[:900]}\n\n"
            )
    return (
        f"MODE: {mode.strip().lower()}\n\n"
        f"{brief_block}{anti_echo}{coverage}"
        f"QUESTION (transcribed or pasted):\n{question.strip()}\n\n"
        f"CONTEXT (from the user's knowledge base; may be partial):\n{ctx}\n\n"
        f"{spoken if want_spoken else not_spoken}"
    )


def build_research_system_prompt(mode: str, *, allow_general_knowledge: bool, strict_kb: bool = False) -> str:
    mode = (mode or "interview").strip().lower()
    grounded = (
        "Facts about the user (employers, dates, metrics, names) must only appear if supported by "
        "CONTEXT or SESSION BRIEF."
    )
    strict = ""
    if strict_kb:
        strict = (
            "STRICT: Do not invent projects or client stories. If CONTEXT lacks a named initiative, "
            "call that out in Gaps instead of substituting a generic example as the user's work. "
        )
    extra = (
        "You may use careful general knowledge for frameworks and interview strategy when CONTEXT "
        "is thin—never invent the user's biography."
        if allow_general_knowledge
        else "If CONTEXT is thin, say so plainly; do not invent user-specific claims."
    )
    extra_kb = (
        " When CONTEXT is thin, use general knowledge only for strategy—not to fabricate work history."
        if (allow_general_knowledge and strict_kb)
        else ""
    )
    return (
        f"You are a sharp research assistant during a live {mode}. "
        "The user already sees a quick spoken-style answer elsewhere; you add depth. "
        f"{strict}{grounded} {extra}{extra_kb}"
    )


def build_research_user_prompt(
    mode: str,
    question: str,
    session_brief: str,
    hits: list[dict],
    *,
    kb_weak: bool,
    top_kb_score: float,
    quick_reply: str | None = None,
    strict_kb: bool = False,
) -> str:
    ctx_parts = []
    for h in hits:
        ctx_parts.append(f"--- SOURCE: {h['source']}\n{h['text']}\n")
    ctx = "\n".join(ctx_parts) if ctx_parts else "(no excerpts retrieved)\n"
    brief = (session_brief or "").strip()
    brief_block = f"SESSION BRIEF:\n{brief}\n\n" if brief else ""
    qr = (quick_reply or "").strip()
    qr_block = (
        f"QUICK ANSWER (already shown to user — extend, do not repeat verbatim):\n{qr}\n\n"
        if qr
        else ""
    )
    strict_line = (
        "STRICT: Deep angles must cite CONTEXT for user-specific claims; use Gaps for missing detail.\n\n"
        if strict_kb
        else ""
    )
    return (
        f"MODE: {mode.strip().lower()}\n\n"
        f"{brief_block}{qr_block}"
        f"KB SIGNAL: top_retrieval_score={top_kb_score:.4f}; kb_weak={'yes' if kb_weak else 'no'}.\n\n"
        f"{strict_line}"
        f"QUESTION:\n{question.strip()}\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        "Write this structure (markdown headings, tight bullets):\n"
        "## Gaps\nWhat CONTEXT does not clearly cover.\n"
        "## Deep angles\n3–6 bullets: risks, follow-up questions, steel-man objections, bridges to their experience — grounded in CONTEXT.\n"
        "## Lines to say\n2–5 short sentences they can speak if faithful to CONTEXT (prefer paraphrasing KB, not inventing).\n"
        "## Verify\nWhat to double-check before claiming.\n"
    )


def run_assist(
    index: BM25Index,
    *,
    question: str,
    mode: str,
    session_brief: str,
    top_k: int,
    use_ollama: bool,
    want_spoken: bool,
    allow_general_knowledge: bool,
    ollama_base_url: str,
    ollama_model_name: str,
    ollama_timeout_s: float,
    kb_weak_threshold: float | None = None,
    prev_question: str | None = None,
    prev_reply: str | None = None,
    ollama_temperature: float | None = None,
    llm_source: str | None = None,
    cloud_quick_model: str | None = None,
    strict_kb: bool = False,
    force_openai_answers: bool = False,
) -> dict[str, Any]:
    question = (question or "").strip()
    if not question:
        return {"error": "question is required", "http_status": 400}

    hits = index.search(question, top_k=top_k)
    kb_weak, top_score = kb_is_weak(hits, threshold=kb_weak_threshold)
    allow_general = allow_general_knowledge

    if not use_ollama:
        return {
            "hits": hits,
            "suggestion": "",
            "warning": "Ollama disabled — sources only.",
            "transcript": "",
            "kb_weak": kb_weak,
            "top_kb_score": top_score,
            "allow_general_effective": allow_general,
        }

    system = build_system_prompt(mode, allow_general_knowledge=allow_general, strict_kb=strict_kb)
    hits_prompt = trim_hits_for_prompt(
        hits,
        total_budget=int(os.environ.get("QUICK_CONTEXT_CHARS", "7200")),
        per_hit_max=int(os.environ.get("QUICK_CONTEXT_PER_HIT", "2600")),
    )
    user = build_user_prompt(
        mode,
        question,
        session_brief,
        hits_prompt,
        want_spoken,
        kb_weak=kb_weak,
        top_kb_score=top_score,
        prev_question=prev_question,
        prev_reply=prev_reply,
        strict_kb=strict_kb,
    )
    try:
        try:
            mt = int(os.environ.get("OLLAMA_QUICK_NUM_PREDICT", "320"))
        except ValueError:
            mt = 320
        t = ollama_temperature if ollama_temperature is not None else 0.52

        openai_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        openai_base = (os.environ.get("OPENAI_API_BASE") or "https://api.openai.com/v1").rstrip("/")
        use_openai_direct = bool(force_openai_answers and openai_key)
        cloud_path = use_cloud_for_llm_source(llm_source) or use_openai_direct
        if cloud_path:
            if use_openai_direct:
                cbase = openai_base
                ckey = openai_key
                cmodel = (
                    (cloud_quick_model or "").strip()
                    or (os.environ.get("OPENAI_ANSWER_MODEL") or "").strip()
                    or "gpt-4o-mini"
                )
            else:
                cbase, ckey, _ = _cloud_chat_params()
                cmodel = resolve_cloud_quick_model(cloud_quick_model)
            if not ckey:
                b = best_kb_snippet(question, hits)
                return {
                    "hits": hits,
                    "suggestion": kb_fallback_reply(
                        hits,
                        b,
                        want_spoken=want_spoken,
                        question=question,
                        use_cloud_llm=True,
                        emit_verbatim_kb_excerpt=(b is None),
                    ),
                    "warning": "Cloud LLM selected but OPENAI_COMPAT_API_KEY (or GROQ_API_KEY) is not set.",
                    "transcript": "",
                    "kb_weak": kb_weak,
                    "top_kb_score": top_score,
                    "allow_general_effective": allow_general,
                }
            suggestion = chat_complete(
                cbase,
                ckey,
                cmodel,
                user,
                system=system,
                timeout_s=ollama_timeout_s,
                temperature=t,
                max_tokens=mt,
            )
        else:
            nctx = 2048
            try:
                nctx = int(os.environ.get("OLLAMA_QUICK_NUM_CTX", "2048"))
            except ValueError:
                pass
            suggestion = ollama_generate(
                ollama_base_url,
                ollama_model_name,
                user,
                system=system,
                timeout_s=ollama_timeout_s,
                temperature=ollama_temperature,
                num_ctx=nctx,
                num_predict=mt,
            )
        suggestion = (suggestion or "").strip()
        warn = ""
        if not suggestion:
            b = best_kb_snippet(question, hits)
            suggestion = kb_fallback_reply(
                hits,
                b,
                want_spoken=want_spoken,
                question=question,
                use_cloud_llm=cloud_path,
                emit_verbatim_kb_excerpt=(b is None),
            )
            warn = "LLM returned no text; read the suggestion below and the Verbatim KB panel."
        return {
            "hits": hits,
            "suggestion": suggestion,
            "warning": warn,
            "transcript": "",
            "kb_weak": kb_weak,
            "top_kb_score": top_score,
            "allow_general_effective": allow_general,
        }
    except RuntimeError as e:
        b = best_kb_snippet(question, hits)
        fb = kb_fallback_reply(
            hits,
            b,
            want_spoken=want_spoken,
            question=question,
            use_cloud_llm=use_cloud_for_llm_source(llm_source),
            emit_verbatim_kb_excerpt=(b is None),
        )
        return {
            "hits": hits,
            "suggestion": fb,
            "warning": str(e),
            "transcript": "",
            "kb_weak": kb_weak,
            "top_kb_score": top_score,
            "allow_general_effective": allow_general,
        }
