"""
Local speech-to-text via faster-whisper (fully offline after first model download).
Requires ffmpeg installed (e.g. brew install ffmpeg) for WebM/MP4 from the browser.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path

_lock = threading.Lock()
_model = None


def whisper_model_name() -> str:
    return os.environ.get("WHISPER_MODEL", "tiny.en")


def get_whisper_model():
    global _model
    with _lock:
        if _model is not None:
            return _model
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise RuntimeError(
                "faster-whisper is not installed. Run: pip install -r requirements.txt"
            ) from e
        name = whisper_model_name()
        device = os.environ.get("WHISPER_DEVICE", "cpu")
        compute = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
        cpu_threads = int(os.getenv("WHISPER_CPU_THREADS", "0") or 0)
        kwargs = {}
        if cpu_threads > 0:
            kwargs["cpu_threads"] = cpu_threads
        _model = WhisperModel(name, device=device, compute_type=compute, **kwargs)
        return _model


def warmup() -> None:
    """Eagerly load the Whisper model so the first transcription is fast."""
    try:
        get_whisper_model()
    except Exception:
        pass


def _ffmpeg_pcm_wav(src: Path) -> Path | None:
    """Best-effort: re-mux/decode browser WebM/MP4 to 16 kHz mono WAV (fixes many InvalidDataError cases)."""
    if not shutil.which("ffmpeg"):
        return None
    out = src.with_suffix(".fw16k.wav")
    try:
        r = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(src),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(out),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r.returncode != 0 or not out.is_file() or out.stat().st_size < 64:
            try:
                out.unlink(missing_ok=True)
            except OSError:
                pass
            return None
        return out
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        try:
            out.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def _one_transcribe_pass(
    model,
    media_path: Path,
    language: str | None,
    beam: int,
    vad: bool,
) -> tuple[str, dict]:
    segments, info = model.transcribe(
        str(media_path),
        language=language if language else None,
        task="transcribe",
        beam_size=max(1, beam),
        vad_filter=vad,
        word_timestamps=False,
        condition_on_previous_text=False,
    )
    parts: list[str] = []
    for seg in segments:
        t = seg.text.strip()
        if t:
            parts.append(t)
    text = " ".join(parts).strip()
    meta = {
        "language": getattr(info, "language", None),
        "duration": round(float(getattr(info, "duration", 0.0) or 0.0), 3),
    }
    return text, meta


def transcribe_path(
    audio_path: str | Path,
    language: str | None = None,
) -> tuple[str, dict]:
    """
    Transcribe a file on disk. Returns (text, meta).
    language: ISO code like 'en', or None to auto-detect.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    model = get_whisper_model()
    beam = int(os.environ.get("WHISPER_BEAM_SIZE", "1") or 1)
    vad = os.environ.get("WHISPER_VAD", "0") not in ("0", "false", "no")

    aux: list[Path] = []
    try:
        try:
            return _one_transcribe_pass(model, path, language, beam, vad)
        except Exception as first:
            wav = _ffmpeg_pcm_wav(path)
            if wav is not None:
                aux.append(wav)
                try:
                    return _one_transcribe_pass(model, wav, language, beam, vad)
                except Exception as second:
                    raise RuntimeError(
                        "Could not transcribe audio even after ffmpeg conversion. The recording may be "
                        "too short or corrupted — speak 2–3 seconds, then click Stop & Answer. "
                        f"Original decode error: {first}. After ffmpeg: {second}"
                    ) from second
            raise RuntimeError(
                "Could not read the audio file (often a truncated browser recording). "
                "Record at least 2 seconds before stopping, avoid double-clicking Stop. "
                "Install ffmpeg for better WebM repair: brew install ffmpeg. "
                f"Details: {first}"
            ) from first
    finally:
        for p in aux:
            try:
                p.unlink()
            except OSError:
                pass
