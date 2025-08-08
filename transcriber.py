import os
import shutil
import uuid
import tempfile
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List, Callable

from yt_dlp import YoutubeDL
from stt_service import STTService, BusyError, OOMError
from config import Settings, load_settings

SESSIONS_ROOT = os.path.join(os.path.dirname(__file__), "sessions")
os.makedirs(SESSIONS_ROOT, exist_ok=True)


def _new_session_dir() -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    sid = f"{stamp}-{uuid.uuid4().hex[:8]}"
    path = os.path.join(SESSIONS_ROOT, sid)
    os.makedirs(path, exist_ok=True)
    return path


def _download_with_progress(url: str, out_dir: str, debug_print: bool, emit: Optional[Callable[[str, dict], None]]) -> str:
    """
    Download audio using yt-dlp and emit progress via `emit('progress', {...})`.
    """
    def hook(d):
        if not emit:
            return
        st = d.get("status")
        if st == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            done = d.get("downloaded_bytes") or 0
            pct = (100.0 * done / total) if total else None
            spd = d.get("speed") or 0
            note = f"{spd/1024/1024:.2f} MB/s" if spd else ""
            payload = {"stage": "downloading"}
            if pct is not None:
                payload["pct"] = round(min(99.0, max(0.0, pct)), 1)
            if note:
                payload["note"] = note
            emit("progress", payload)
        elif st == "finished":
            emit("progress", {"stage": "downloading", "pct": 100.0})

    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "%(title).200s.%(ext)s"),
        "noprogress": not debug_print,
        "quiet": not debug_print,
        "nocheckcertificate": True,
        "progress_hooks": [hook],
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if "requested_downloads" in info and info["requested_downloads"]:
            return info["requested_downloads"][0]["filepath"]
        title = info.get("title", "audio")
        ext = info.get("ext", "m4a")
        return os.path.join(out_dir, f"{title}.{ext}")


def transcribe_url(
    url: str,
    *,
    lang: Optional[str] = None,
    chunk_size: int = 15,
    keep_files: bool = False,
    debug_print: bool = False,
    num_speakers: Optional[int] = None,
    hf_token: Optional[str] = None,
    service: Optional[STTService] = None,
    settings: Optional[Settings] = None,
    safe_mode: bool = False,
) -> Tuple[str, Optional[str], Dict[str, Any], List[str]]:
    """
    Legacy one-shot call (no streaming).
    """
    session_dir = _new_session_dir() if keep_files else tempfile.mkdtemp(prefix="stt_")
    audio_path = _download_with_progress(url, session_dir, debug_print, emit=None)

    svc = service or STTService(settings or load_settings())

    try:
        text, strategy, warnings = svc.transcribe(
            audio_path,
            lang=lang,
            chunk_size=chunk_size,
            num_speakers=num_speakers,
            hf_token=hf_token,
            safe_mode=safe_mode,
        )
        if keep_files:
            return text, session_dir, strategy, warnings
        else:
            shutil.rmtree(session_dir, ignore_errors=True)
            return text, None, strategy, warnings
    except Exception:
        if not keep_files:
            shutil.rmtree(session_dir, ignore_errors=True)
        raise


# ---- NEW: streaming job used by SSE endpoint ----
def run_stream_job(
    *,
    url: str,
    lang: Optional[str],
    chunk_size: int,
    num_speakers: Optional[int],
    hf_token: Optional[str],
    safe_mode: bool,
    service: STTService,
    settings: Settings,
    emit: Callable[[str, dict], None],
) -> None:
    """
    Orchestrates download + streaming transcription and emits SSE-friendly events via `emit`.
    """
    emit("progress", {"stage": "queue", "pct": 1.0})

    session_dir = tempfile.mkdtemp(prefix="stt_")
    try:
        # Download with true progress
        audio_path = _download_with_progress(url, session_dir, debug_print=False, emit=emit)

        # Transcribe with streaming progress from Faster-Whisper
        def on_progress(payload: dict):
            # normalize to SSE 'progress' events
            emit("progress", payload)

        text, strategy, warnings = service.stream_transcribe(
            filename=audio_path,
            lang=lang,
            chunk_size=chunk_size,
            num_speakers=num_speakers,
            hf_token=hf_token,
            safe_mode=safe_mode,
            progress_cb=on_progress,
        )

        emit("progress", {"stage": "finalizing", "pct": 100.0})
        emit("done", {"transcript": text, "strategy": strategy, "warnings": warnings})

    finally:
        shutil.rmtree(session_dir, ignore_errors=True)
