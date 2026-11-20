"""
KB-first live assistant: local Whisper → retrieve → Ollama (offline-capable on laptop).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

# Load kb-copilot/.env before any os.environ reads (optional: pip install python-dotenv).
_ROOT = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

from flask import Flask, jsonify, make_response, render_template, request, stream_with_context
from werkzeug.utils import secure_filename

import shutil
import subprocess
import time

from assist_core import (
    best_kb_snippet,
    build_research_system_prompt,
    build_research_user_prompt,
    build_system_prompt,
    build_user_prompt,
    kb_fallback_reply,
    kb_is_weak,
    normalize_llm_source,
    resolve_cloud_quick_model,
    resolve_cloud_research_model,
    run_assist,
    trim_hits_for_prompt,
    use_cloud_for_llm_source,
)
from kb_store import delete_file as kb_delete_file
from kb_store import list_files as kb_list_files
from kb_store import save_upload as kb_save_upload
from openai_compatible_client import chat_stream as openai_chat_stream
from ollama_client import generate_stream as ollama_stream
from ollama_client import is_reachable as ollama_reachable
from ollama_client import list_loaded_models as ollama_list_loaded_models
from ollama_client import list_models as ollama_list_models
from ollama_client import warmup as ollama_warmup
from retrieval import BM25Index, load_corpus
from transcription import transcribe_path
from transcription import warmup as whisper_warmup

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "32")) * 1024 * 1024

_index_lock = threading.Lock()
_index: BM25Index | None = None
_kb_root: Path | None = None


def kb_path() -> Path:
    raw = os.environ.get("KB_ROOT", "")
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path(__file__).resolve().parent / "kb").resolve()


def ollama_base() -> str:
    return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def ollama_model() -> str:
    # Match a model you have locally: `ollama list` (e.g. llama3.1:8b, llama3.2:3b).
    return os.environ.get("OLLAMA_MODEL", "llama3.1:8b")


_QUICK_PREF = ["llama3.2:3b", "llama3.2:1b", "qwen2.5:3b", "phi3.5:3.8b", "gemma2:2b"]
# Used only when OLLAMA_RESEARCH_MODEL is unset and OLLAMA_RESEARCH_USE_BIG=1
_RESEARCH_BIG_PREF = ["llama3.1:8b", "qwen2.5:7b", "llama3.1:70b", "llama3.2:3b"]


def _auto_pick(preferred: list[str]) -> str | None:
    try:
        installed = set(ollama_list_models(ollama_base(), timeout_s=2.0))
    except Exception:
        installed = set()
    for name in preferred:
        if name in installed:
            return name
    return None


def quick_model() -> str:
    explicit = os.environ.get("OLLAMA_QUICK_MODEL", "").strip()
    if explicit:
        return explicit
    pick = _auto_pick(_QUICK_PREF)
    return pick or ollama_model()


def research_model() -> str:
    explicit = os.environ.get("OLLAMA_RESEARCH_MODEL", "").strip()
    if explicit:
        return explicit
    # Default: same weights as quick. Two simultaneous pulls (3B quick + 8B research)
    # unload each other on typical laptops → long "model loading" stalls and timeouts.
    if os.environ.get("OLLAMA_RESEARCH_USE_BIG", "").strip().lower() in ("1", "true", "yes"):
        pick = _auto_pick(_RESEARCH_BIG_PREF)
        return pick or ollama_model()
    return quick_model()


def quick_num_predict() -> int:
    try:
        return int(os.environ.get("OLLAMA_QUICK_NUM_PREDICT", "320"))
    except ValueError:
        return 320


def quick_temperature() -> float:
    try:
        return float(os.environ.get("OLLAMA_QUICK_TEMPERATURE", "0.52"))
    except ValueError:
        return 0.52


def llm_backend() -> str:
    return (os.environ.get("KB_LLM_BACKEND") or "ollama").strip().lower()


def use_cloud_llm() -> bool:
    return llm_backend() in ("openai_compatible", "cloud", "groq", "openai")


def openai_compat_base() -> str:
    return (os.environ.get("OPENAI_COMPAT_BASE_URL") or "https://api.groq.com/openai/v1").rstrip("/")


def openai_compat_key() -> str:
    return (os.environ.get("OPENAI_COMPAT_API_KEY") or os.environ.get("GROQ_API_KEY") or "").strip()


def openai_native_key() -> str:
    """Official OpenAI API key (optional — used when UI requests OpenAI answers)."""
    return (os.environ.get("OPENAI_API_KEY") or "").strip()


def openai_native_base() -> str:
    return (os.environ.get("OPENAI_API_BASE") or "https://api.openai.com/v1").rstrip("/")


def openai_answer_model_quick(override: str | None) -> str:
    o = (override or "").strip()
    if o:
        return o
    return (os.environ.get("OPENAI_ANSWER_MODEL") or "gpt-4o-mini").strip()


def openai_answer_model_research(quick_override: str | None, research_override: str | None) -> str:
    r = (research_override or "").strip()
    if r:
        return r
    env_r = (os.environ.get("OPENAI_ANSWER_RESEARCH_MODEL") or "").strip()
    if env_r:
        return env_r
    return openai_answer_model_quick(quick_override)


def transcript_looks_like_question(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 12:
        return False
    if "?" in t:
        return True
    low = t.lower()
    needles = (
        "why ",
        "why?",
        "how ",
        "what ",
        "when ",
        "where ",
        "who ",
        "tell me",
        "describe ",
        "explain ",
        "walk me",
        "could you",
        "can you",
        "would you",
        "hire you",
        "choose you",
        "your experience",
    )
    if any(n in low for n in needles):
        return True
    return len(t) >= 96


def cloud_quick_model_name() -> str:
    return (
        os.environ.get("OPENAI_COMPAT_QUICK_MODEL")
        or os.environ.get("GROQ_MODEL")
        or "llama-3.1-8b-instant"
    ).strip()


def cloud_research_model_name() -> str:
    return (os.environ.get("OPENAI_COMPAT_RESEARCH_MODEL") or cloud_quick_model_name()).strip()


def research_num_predict() -> int:
    try:
        return int(os.environ.get("OLLAMA_RESEARCH_NUM_PREDICT", "384"))
    except ValueError:
        return 384


def quick_first_token_timeout_s() -> float:
    try:
        return float(os.environ.get("OLLAMA_QUICK_FIRST_TOKEN_S", "20"))
    except ValueError:
        return 20.0


def ollama_timeout() -> float:
    return float(os.environ.get("OLLAMA_TIMEOUT", "180"))


def ollama_stream_socket_timeout() -> float | None:
    """Max seconds between socket reads while streaming. None = no limit (avoid bogus timeouts during long loads)."""
    raw = (os.environ.get("OLLAMA_STREAM_SOCKET_TIMEOUT", "600") or "600").strip().lower()
    if raw in ("none", "inf", "off", "0"):
        return None
    try:
        v = float(raw)
        return None if v < 0 else v
    except ValueError:
        return 600.0


def quick_stream_socket_timeout() -> float | None:
    """Faster failure on quick path so we can stream a KB fallback instead of hanging."""
    raw = (os.environ.get("OLLAMA_QUICK_STREAM_SOCKET_TIMEOUT", "120") or "").strip()
    if not raw or raw.lower() in ("none", "inf", "off"):
        return ollama_stream_socket_timeout()
    try:
        v = float(raw)
        return None if v < 0 else v
    except ValueError:
        return 120.0


def quick_num_ctx() -> int:
    try:
        return int(os.environ.get("OLLAMA_QUICK_NUM_CTX", "2048"))
    except ValueError:
        return 2048


def ollama_keep_alive() -> str:
    return os.environ.get("OLLAMA_KEEP_ALIVE", "30m")


def ensure_index() -> BM25Index:
    global _index, _kb_root
    root = kb_path()
    with _index_lock:
        if _index is not None and _kb_root == root:
            return _index
        corpus = load_corpus(root)
        if not corpus:
            corpus = [
                (
                    "(empty knowledge base)",
                    "Add .md or .txt files under your KB folder, then click Reindex.",
                )
            ]
        _index = BM25Index.build(corpus)
        _kb_root = root
        return _index


def upload_suffix(file_storage) -> str:
    fn = (secure_filename(file_storage.filename or "") or "").lower()
    for ext in (".webm", ".mp4", ".m4a", ".wav", ".ogg", ".oga"):
        if fn.endswith(ext):
            return ext
    ct = (file_storage.mimetype or "").lower()
    if "webm" in ct:
        return ".webm"
    if "mp4" in ct or "m4a" in ct or "mp4a" in ct:
        return ".mp4"
    if "wav" in ct:
        return ".wav"
    return ".webm"


@app.route("/")
def home():
    cloud = use_cloud_llm()
    html = render_template(
        "index.html",
        kb_root=str(kb_path()),
        ollama_base=ollama_base(),
        ollama_model=ollama_model(),
        quick_model=quick_model(),
        research_model=research_model(),
        llm_backend=llm_backend(),
        llm_provider="cloud" if cloud else "ollama",
        llm_uses_cloud=cloud,
        quick_llm_label=cloud_quick_model_name() if cloud else quick_model(),
        research_llm_label=cloud_research_model_name() if cloud else research_model(),
        whisper_model=os.environ.get("WHISPER_MODEL", "base"),
    )
    resp = make_response(html)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


def _invalidate_index() -> None:
    global _index, _kb_root
    with _index_lock:
        _index = None
        _kb_root = None


@app.post("/api/reindex")
def api_reindex():
    _invalidate_index()
    idx = ensure_index()
    return jsonify({"ok": True, "chunks": len(idx.chunks), "kb_root": str(kb_path())})


@app.get("/api/kb/files")
def api_kb_files():
    files = kb_list_files(kb_path())
    return jsonify({"kb_root": str(kb_path()), "files": files})


@app.post("/api/kb/upload")
def api_kb_upload():
    if "files" not in request.files:
        return jsonify({"error": "files field required"}), 400
    items = request.files.getlist("files")
    if not items:
        return jsonify({"error": "no files in upload"}), 400

    saved: list[dict] = []
    errors: list[dict] = []
    for f in items:
        if not f or f.filename == "":
            continue
        try:
            info = kb_save_upload(kb_path(), f)
            saved.append(info)
        except RuntimeError as e:
            errors.append({"name": f.filename, "error": str(e)})
        except OSError as e:
            errors.append({"name": f.filename, "error": str(e)})

    if saved:
        _invalidate_index()
        idx = ensure_index()
        chunks = len(idx.chunks)
    else:
        chunks = None

    status = 200 if saved else 400
    return jsonify(
        {
            "ok": bool(saved),
            "saved": saved,
            "errors": errors,
            "chunks": chunks,
        }
    ), status


def _run_open(args: list[str], timeout: float = 20.0) -> bool:
    try:
        r = subprocess.run(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _try_start_ollama_app() -> tuple[bool, str]:
    """Best-effort start Ollama. Returns (ok, how)."""
    bundled = "/Applications/Ollama.app/Contents/Resources/ollama"
    app_paths = [
        "/Applications/Ollama.app",
        os.path.expanduser("~/Applications/Ollama.app"),
    ]

    if _run_open(["open", "-a", "Ollama"]):
        return True, "open -a Ollama"
    if _run_open(["open", "-b", "com.electron.ollama"]):
        return True, "open -b com.electron.ollama"

    for app_path in app_paths:
        if os.path.isdir(app_path) and _run_open(["open", app_path]):
            return True, f"open {app_path}"

    cli = shutil.which("ollama")
    if not cli and os.path.isfile(bundled):
        cli = bundled
    if cli:
        try:
            subprocess.Popen(
                [cli, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return True, f"{cli} serve"
        except OSError:
            return False, "ollama serve failed"
    return False, "ollama not found"


def _wait_for_ollama(base: str, max_seconds: float | None = None) -> bool:
    if max_seconds is None:
        max_seconds = float(os.environ.get("OLLAMA_START_WAIT", "60"))
    deadline = time.time() + max_seconds
    while time.time() < deadline:
        if ollama_reachable(base, timeout_s=1.0):
            return True
        time.sleep(0.5)
    return False


@app.get("/api/health")
def api_health():
    base = ollama_base()
    reachable = ollama_reachable(base)
    models = ollama_list_models(base) if reachable else []
    loaded = ollama_list_loaded_models(base) if reachable else []
    target = ollama_model()
    have = target in models if models else False
    qm = quick_model()
    rm = research_model()
    cloud = use_cloud_llm()
    key_ok = bool(openai_compat_key())
    oa_ok = bool(openai_native_key())
    return jsonify(
        {
            "llm": {
                "backend": llm_backend(),
                "provider": "cloud" if cloud else "ollama",
                "uses_ollama_for_text": not cloud,
                "has_cloud_key": key_ok,
                "has_openai_answer_key": oa_ok,
                "note": (
                    "Quick answer and research pass both use this OpenAI-compatible API. "
                    "Ollama is not used for LLM text while KB_LLM_BACKEND is set to a cloud mode."
                    if cloud
                    else (
                        "Quick answer and research pass both use local Ollama. "
                        "Set KB_LLM_BACKEND=openai_compatible and an API key to use cloud instead."
                    )
                ),
                "cloud_base": openai_compat_base() if key_ok else None,
                "cloud_key_set": key_ok,
                "cloud_quick_model": cloud_quick_model_name() if key_ok else None,
                "cloud_research_model": cloud_research_model_name() if key_ok else None,
                "ollama_quick_model": quick_model(),
                "ollama_research_model": research_model(),
            },
            "ollama": {
                "base": base,
                "reachable": reachable,
                "model": target,
                "model_available": have,
                "models": models,
                "loaded": loaded,
                "quick_model": qm,
                "research_model": rm,
                "quick_available": (qm in models) if models else False,
                "research_available": (rm in models) if models else False,
                "quick_loaded": qm in loaded,
                "research_loaded": rm in loaded,
                "ready": (qm in loaded) and (qm in models) and reachable,
            },
            "kb_root": str(kb_path()),
        }
    )


@app.post("/api/warmup")
def api_warmup():
    base = ollama_base()
    if not ollama_reachable(base):
        return jsonify({"ok": False, "error": "Ollama not reachable"}), 503
    qm = quick_model()
    rm = research_model()
    threading.Thread(
        target=lambda: ollama_warmup(base, qm, ollama_keep_alive(), timeout_s=120.0),
        daemon=True,
    ).start()
    if rm and rm != qm:
        threading.Thread(
            target=lambda: ollama_warmup(base, rm, ollama_keep_alive(), timeout_s=120.0),
            daemon=True,
        ).start()
    return jsonify({"ok": True, "warming": [m for m in {qm, rm} if m]})


@app.post("/api/start_ollama")
def api_start_ollama():
    base = ollama_base()
    if ollama_reachable(base):
        return jsonify({"ok": True, "already_running": True, "base": base})
    started, how = _try_start_ollama_app()
    if not started:
        return jsonify(
            {
                "ok": False,
                "error": (
                    "Could not find Ollama.app or `ollama` CLI. Install from "
                    "https://ollama.com/download or open Ollama from /Applications."
                ),
            }
        ), 503
    ok = _wait_for_ollama(base)
    if ok:
        threading.Thread(
            target=lambda: ollama_warmup(base, ollama_model(), ollama_keep_alive()),
            daemon=True,
        ).start()
        return jsonify({"ok": True, "started": True, "base": base, "via": how})
    # Avoid HTTP 504 — browsers label it “Gateway timeout” and hide the real message.
    return (
        jsonify(
            {
                "ok": False,
                "via": how,
                "error": (
                    "Ollama still not answering at "
                    f"{base} after {os.environ.get('OLLAMA_START_WAIT', '60')}s. "
                    "Open the Ollama app from Spotlight (⌘+Space → “Ollama”) and wait "
                    "until the whale icon appears in the menu bar, then click “Start Ollama” "
                    "again. Or in Terminal: `ollama serve`."
                ),
            }
        ),
        503,
    )


@app.post("/api/kb/delete")
def api_kb_delete():
    data = request.get_json(force=True, silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name is required"}), 400
    ok = kb_delete_file(kb_path(), name)
    if not ok:
        return jsonify({"error": "file not found"}), 404
    _invalidate_index()
    idx = ensure_index()
    return jsonify({"ok": True, "chunks": len(idx.chunks)})


@app.post("/api/assist")
def api_assist():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400

    idx = ensure_index()
    out = run_assist(
        idx,
        question=question,
        mode=str(data.get("mode") or "interview"),
        session_brief=str(data.get("session_brief") or ""),
        top_k=int(data.get("top_k") or 6),
        use_ollama=bool(data.get("use_ollama", True)),
        want_spoken=bool(data.get("spoken", True)),
        allow_general_knowledge=bool(data.get("allow_general", True)),
        ollama_base_url=ollama_base(),
        ollama_model_name=quick_model(),
        ollama_timeout_s=ollama_timeout(),
        prev_question=str(data.get("prev_question") or "").strip() or None,
        prev_reply=str(data.get("prev_reply") or "").strip() or None,
        ollama_temperature=quick_temperature(),
        llm_source=normalize_llm_source(data.get("llm_source")),
        cloud_quick_model=str(data.get("cloud_quick_model") or "").strip() or None,
        strict_kb=bool(data.get("strict_kb", True)),
        force_openai_answers=bool(data.get("use_openai_answers")),
    )
    if out.get("error"):
        return jsonify({"error": out["error"]}), int(out.get("http_status") or 400)
    out["best"] = best_kb_snippet(question, out.get("hits") or [])
    return jsonify(out)


@app.post("/api/transcribe")
def api_transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "audio file field required (multipart form key: audio)"}), 400
    up = request.files["audio"]
    if not up or up.filename == "":
        return jsonify({"error": "empty upload"}), 400
    lang = (request.form.get("language") or request.args.get("language") or "").strip() or None
    if lang == "auto":
        lang = None
    suf = upload_suffix(up)
    path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suf, delete=False) as tmp:
            up.save(tmp.name)
            path = tmp.name
        text, meta = transcribe_path(path, language=lang)
    except (RuntimeError, OSError, FileNotFoundError) as e:
        return jsonify({"error": str(e)}), 503
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass
    return jsonify({"text": text, "meta": meta})


@app.post("/api/listen_assist")
def api_listen_assist():
    if "audio" not in request.files:
        return jsonify({"error": "audio file required"}), 400
    up = request.files["audio"]
    if not up or up.filename == "":
        return jsonify({"error": "empty upload"}), 400

    lang = (request.form.get("language") or "").strip() or None
    if lang == "auto":
        lang = None
    mode = request.form.get("mode") or "interview"
    session_brief = request.form.get("session_brief") or ""
    top_k = int(request.form.get("top_k") or 6)
    use_ollama = request.form.get("use_ollama", "1") not in ("0", "false", "no")
    spoken = request.form.get("spoken", "1") not in ("0", "false", "no")
    allow_general = request.form.get("allow_general", "1") not in ("0", "false", "no")
    strict_kb = request.form.get("strict_kb", "1") not in ("0", "false", "no")

    suf = upload_suffix(up)
    path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suf, delete=False) as tmp:
            up.save(tmp.name)
            path = tmp.name
        try:
            transcript, meta = transcribe_path(path, language=lang)
        except (RuntimeError, OSError, FileNotFoundError) as e:
            return jsonify({"error": str(e), "transcript": ""}), 503
    finally:
        if path:
            try:
                os.unlink(path)
            except Exception:
                pass

    if not (transcript or "").strip():
        return jsonify(
            {
                "error": "Transcription was empty — speak closer to the mic or check levels.",
                "transcript": "",
                "meta": meta,
            }
        ), 400

    idx = ensure_index()
    prev_q = (request.form.get("prev_question") or "").strip() or None
    prev_a = (request.form.get("prev_reply") or "").strip() or None
    out = run_assist(
        idx,
        question=transcript,
        mode=str(mode),
        session_brief=str(session_brief),
        top_k=top_k,
        use_ollama=use_ollama,
        want_spoken=spoken,
        allow_general_knowledge=allow_general,
        ollama_base_url=ollama_base(),
        ollama_model_name=quick_model(),
        ollama_timeout_s=ollama_timeout(),
        prev_question=prev_q,
        prev_reply=prev_a,
        ollama_temperature=quick_temperature(),
        llm_source=normalize_llm_source(request.form.get("llm_source")),
        cloud_quick_model=(request.form.get("cloud_quick_model") or "").strip() or None,
        strict_kb=strict_kb,
    )
    out["transcript"] = transcript
    out["meta"] = meta
    out["best"] = best_kb_snippet(transcript, out.get("hits") or [])
    return jsonify(out)


def _sse(name: str, payload) -> bytes:
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


@app.post("/api/listen_assist_stream")
def api_listen_assist_stream():
    if "audio" not in request.files:
        return jsonify({"error": "audio file required"}), 400
    up = request.files["audio"]
    if not up or up.filename == "":
        return jsonify({"error": "empty upload"}), 400

    lang = (request.form.get("language") or "").strip() or None
    if lang == "auto":
        lang = None
    mode = request.form.get("mode") or "interview"
    session_brief = request.form.get("session_brief") or ""
    top_k = int(request.form.get("top_k") or 6)
    spoken = request.form.get("spoken", "1") not in ("0", "false", "no")
    allow_general = request.form.get("allow_general", "1") not in ("0", "false", "no")
    strict_kb = request.form.get("strict_kb", "1") not in ("0", "false", "no")
    prev_q = (request.form.get("prev_question") or "").strip() or None
    prev_a = (request.form.get("prev_reply") or "").strip() or None
    llm_src = normalize_llm_source(request.form.get("llm_source"))
    cloud_quick_override = (request.form.get("cloud_quick_model") or "").strip() or None
    question_gate_flag = request.form.get("question_gate") == "1"
    research_first_flag = request.form.get("research_first") == "1"
    force_openai_flag = request.form.get("use_openai_answers") == "1"
    cloud_research_override = (request.form.get("cloud_research_model") or "").strip() or None

    suf = upload_suffix(up)
    with tempfile.NamedTemporaryFile(suffix=suf, delete=False) as tmp:
        up.save(tmp.name)
        audio_path = tmp.name

    def stream():
        try:
            yield _sse("stage", {"name": "transcribing"})
            try:
                transcript, meta = transcribe_path(audio_path, language=lang)
            except Exception as e:
                yield _sse("warning", {"message": str(e)})
                yield _sse("transcript", {"text": "", "meta": {}})
                yield _sse("done", {})
                return

            transcript_text = (transcript or "").strip()
            idx = ensure_index()
            hits = idx.search(transcript_text, top_k=top_k) if transcript_text else []
            weak, top_score = kb_is_weak(hits)
            allow_general_eff = allow_general

            base_url = ollama_base()
            keep_alive = ollama_keep_alive()
            best = best_kb_snippet(transcript_text, hits)

            yield _sse("transcript", {"text": transcript_text, "meta": meta})
            yield _sse(
                "context",
                {
                    "hits": hits,
                    "kb_weak": weak,
                    "top_kb_score": top_score,
                    "allow_general_effective": allow_general_eff,
                },
            )
            if best:
                yield _sse("best", best)
            if not transcript_text:
                yield _sse("warning", {"message": "Transcription was empty — try again, closer to the mic."})
                yield _sse("done", {})
                return

            if question_gate_flag and not transcript_looks_like_question(transcript_text):
                yield _sse(
                    "warning",
                    {"message": "No interview question detected in this clip — skipped LLM answer (continuous mode)."},
                )
                yield _sse("done", {})
                return

            use_openai_direct = bool(force_openai_flag and openai_native_key())
            if force_openai_flag and not use_openai_direct:
                yield _sse(
                    "warning",
                    {
                        "message": "OpenAI answers are enabled in the UI but OPENAI_API_KEY is missing in .env — falling back to your normal LLM setting.",
                    },
                )

            try:
                rtemp = float(os.environ.get("OPENAI_COMPAT_RESEARCH_TEMP", "0.42"))
            except ValueError:
                rtemp = 0.42

            if research_first_flag:
                hits_prompt = trim_hits_for_prompt(
                    hits,
                    total_budget=int(os.environ.get("RESEARCH_CONTEXT_CHARS", "12000")),
                    per_hit_max=int(os.environ.get("RESEARCH_CONTEXT_PER_HIT", "4000")),
                )
                system = build_research_system_prompt(mode, allow_general_knowledge=allow_general_eff, strict_kb=strict_kb)
                user_prompt = build_research_user_prompt(
                    mode,
                    transcript_text,
                    session_brief,
                    hits_prompt,
                    kb_weak=weak,
                    top_kb_score=top_score,
                    quick_reply=None,
                    strict_kb=strict_kb,
                )
            else:
                system = build_system_prompt(mode, allow_general_knowledge=allow_general_eff, strict_kb=strict_kb)
                hits_prompt = trim_hits_for_prompt(
                    hits,
                    total_budget=int(os.environ.get("QUICK_CONTEXT_CHARS", "7200")),
                    per_hit_max=int(os.environ.get("QUICK_CONTEXT_PER_HIT", "2600")),
                )
                user_prompt = build_user_prompt(
                    mode,
                    transcript_text,
                    session_brief,
                    hits_prompt,
                    spoken,
                    kb_weak=weak,
                    top_kb_score=top_score,
                    prev_question=prev_q,
                    prev_reply=prev_a,
                    strict_kb=strict_kb,
                )

            cloud_llm = use_cloud_for_llm_source(llm_src) or use_openai_direct
            yield _sse(
                "stage",
                {"name": "llm", "provider": "cloud" if cloud_llm else "ollama"},
            )

            emitted_any = False
            if cloud_llm:
                api_key = openai_native_key() if use_openai_direct else openai_compat_key()
                api_base = openai_native_base() if use_openai_direct else openai_compat_base()
                if not api_key:
                    fb = kb_fallback_reply(
                        hits,
                        best,
                        want_spoken=spoken,
                        question=transcript_text,
                        use_cloud_llm=True,
                        emit_verbatim_kb_excerpt=(best is None),
                    )
                    yield _sse(
                        "warning",
                        {
                            "message": (
                                "Cloud / OpenAI LLM selected but no API key is set in .env "
                                "(OPENAI_API_KEY for OpenAI answers, or OPENAI_COMPAT_API_KEY / GROQ_API_KEY for Groq)."
                            ),
                        },
                    )
                    yield _sse("token", {"text": fb})
                    emitted_any = True
                else:
                    if research_first_flag:
                        mid = (
                            openai_answer_model_research(cloud_quick_override, cloud_research_override)
                            if use_openai_direct
                            else resolve_cloud_research_model(cloud_research_override)
                        )
                        max_tok = research_num_predict()
                        temp = rtemp
                        to = 180.0
                    else:
                        mid = (
                            openai_answer_model_quick(cloud_quick_override)
                            if use_openai_direct
                            else resolve_cloud_quick_model(cloud_quick_override)
                        )
                        max_tok = quick_num_predict()
                        temp = quick_temperature()
                        to = 120.0
                    try:
                        for chunk in openai_chat_stream(
                            api_base,
                            api_key,
                            mid,
                            user_prompt,
                            system=system,
                            timeout_s=to,
                            temperature=temp,
                            max_tokens=max_tok,
                        ):
                            emitted_any = True
                            yield _sse("token", {"text": chunk})
                    except Exception as e:
                        yield _sse("warning", {"message": str(e)})
            else:
                omodel = research_model() if research_first_flag else quick_model()
                onpred = research_num_predict() if research_first_flag else quick_num_predict()
                osock = ollama_stream_socket_timeout() if research_first_flag else quick_stream_socket_timeout()
                otemp = rtemp if research_first_flag else quick_temperature()
                try:
                    for chunk in ollama_stream(
                        base_url,
                        omodel,
                        user_prompt,
                        system=system,
                        timeout_s=osock,
                        keep_alive=keep_alive,
                        num_predict=onpred,
                        temperature=otemp,
                        num_ctx=quick_num_ctx(),
                    ):
                        emitted_any = True
                        yield _sse("token", {"text": chunk})
                except Exception as e:
                    yield _sse("warning", {"message": str(e)})
            if not emitted_any:
                fb = kb_fallback_reply(
                    hits,
                    best,
                    want_spoken=spoken,
                    question=transcript_text,
                    use_cloud_llm=cloud_llm,
                    emit_verbatim_kb_excerpt=(best is None),
                )
                yield _sse("token", {"text": fb})
                if cloud_llm:
                    if use_openai_direct and not openai_native_key():
                        msg = (
                            "OpenAI answers need OPENAI_API_KEY in .env — a KB-only draft is shown above."
                        )
                    elif not openai_compat_key() and not use_openai_direct:
                        msg = (
                            "Cloud LLM is configured but no API key is set — a KB-only draft is shown above. "
                            "Add OPENAI_COMPAT_API_KEY (or GROQ_API_KEY) to .env and restart."
                        )
                    else:
                        msg = (
                            "The cloud LLM returned no text (empty stream or error). "
                            "Check base URL, model name, quota, and network. "
                            "A KB-only draft is shown above."
                        )
                else:
                    msg = (
                        "Ollama did not return any text. A KB-only draft is shown above — "
                        "check that Ollama is running and the quick model is loaded (Warm up / green dot)."
                    )
                yield _sse("warning", {"message": msg})
            yield _sse("done", {})
        except Exception as e:
            yield _sse(
                "warning",
                {"message": f"{type(e).__name__}: {e}. If this repeats, check the terminal where Flask runs."},
            )
            yield _sse("done", {})
        finally:
            try:
                os.unlink(audio_path)
            except OSError:
                pass

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return app.response_class(stream_with_context(stream()), headers=headers)


@app.post("/api/research_stream")
def api_research_stream():
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    llm_src = normalize_llm_source(data.get("llm_source"))
    cq = (data.get("cloud_quick_model") or "").strip() or None
    cr = (data.get("cloud_research_model") or "").strip() or None

    mode = str(data.get("mode") or "interview")
    session_brief = str(data.get("session_brief") or "")
    allow_general = bool(data.get("allow_general", True))
    strict_kb = bool(data.get("strict_kb", True))
    quick_reply = (data.get("quick_reply") or "").strip() or None
    force_openai = bool(data.get("use_openai_answers"))
    use_openai_direct = bool(force_openai and openai_native_key())
    try:
        top_k = int(data.get("top_k") or os.environ.get("RESEARCH_TOP_K", "10"))
    except ValueError:
        top_k = 10

    idx = ensure_index()
    hits = idx.search(question, top_k=max(4, top_k))
    weak, top_score = kb_is_weak(hits)
    hits_prompt = trim_hits_for_prompt(
        hits,
        total_budget=int(os.environ.get("RESEARCH_CONTEXT_CHARS", "12000")),
        per_hit_max=int(os.environ.get("RESEARCH_CONTEXT_PER_HIT", "4000")),
    )
    system = build_research_system_prompt(mode, allow_general_knowledge=allow_general, strict_kb=strict_kb)
    user = build_research_user_prompt(
        mode,
        question,
        session_brief,
        hits_prompt,
        kb_weak=weak,
        top_kb_score=top_score,
        quick_reply=quick_reply,
        strict_kb=strict_kb,
    )

    base_url = ollama_base()
    model = research_model()
    keep_alive = ollama_keep_alive()
    sock_to = ollama_stream_socket_timeout()
    try:
        rtemp = float(os.environ.get("OPENAI_COMPAT_RESEARCH_TEMP", "0.42"))
    except ValueError:
        rtemp = 0.42

    def gen():
        try:
            if force_openai and not use_openai_direct:
                yield _sse(
                    "warning",
                    {"message": "OpenAI answers requested but OPENAI_API_KEY is missing in .env — using your normal LLM path."},
                )
            if use_cloud_for_llm_source(llm_src) or use_openai_direct:
                key = openai_native_key() if use_openai_direct else openai_compat_key()
                base_u = openai_native_base() if use_openai_direct else openai_compat_base()
                if not key:
                    yield _sse("warning", {"message": "Cloud LLM enabled but API key missing (set OPENAI_API_KEY or OPENAI_COMPAT_API_KEY)."})
                else:
                    rm = (
                        openai_answer_model_research(cq, cr)
                        if use_openai_direct
                        else resolve_cloud_research_model(cr or cq)
                    )
                    for chunk in openai_chat_stream(
                        base_u,
                        key,
                        rm,
                        user,
                        system=system,
                        timeout_s=180.0,
                        temperature=rtemp,
                        max_tokens=research_num_predict(),
                    ):
                        yield _sse("token", {"text": chunk})
            else:
                for chunk in ollama_stream(
                    base_url,
                    model,
                    user,
                    system=system,
                    timeout_s=sock_to,
                    keep_alive=keep_alive,
                    num_predict=research_num_predict(),
                ):
                    yield _sse("token", {"text": chunk})
        except Exception as e:
            yield _sse("warning", {"message": str(e)})
        yield _sse("done", {})

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return app.response_class(gen(), headers=headers)


def main():
    ensure_index()
    port = int(os.environ.get("PORT", "5055"))

    if os.environ.get("WARMUP", "1") not in ("0", "false", "no"):
        threading.Thread(target=whisper_warmup, daemon=True).start()
        for m in {quick_model(), research_model()}:
            threading.Thread(
                target=lambda m=m: ollama_warmup(ollama_base(), m, ollama_keep_alive()),
                daemon=True,
            ).start()

    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
