# transcriber.py

import os
import shutil
import uuid
import tempfile
import time
import math
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List, Callable
from urllib.parse import urlparse, parse_qs, urlencode

from yt_dlp import YoutubeDL

from stt_service import STTService, OOMError, BusyError
from config import Settings, load_settings

# sessions root
SESSIONS_ROOT = os.path.join(os.path.dirname(__file__), "sessions")
os.makedirs(SESSIONS_ROOT, exist_ok=True)

# Type for SSE emitter: emit(event_type, payload_dict)
Emit = Callable[[str, dict], None]


def _new_session_dir() -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    sid = f"{stamp}-{uuid.uuid4().hex[:8]}"
    path = os.path.join(SESSIONS_ROOT, sid)
    os.makedirs(path, exist_ok=True)
    return path


def _normalize_youtube_url(url: str) -> str:
    """
    Make youtu.be links and odd query strings friendly for yt-dlp.
    - youtu.be/<id>?si=...  → https://www.youtube.com/watch?v=<id>
    - youtube.com/watch?v=<id>&si=... → keep only v (and t if present)
    """
    u = urlparse(url)
    host = u.netloc.lower()

    # youtu.be/<id>
    if "youtu.be" in host:
        vid = u.path.lstrip("/")
        if not vid:
            return url
        qs = parse_qs(u.query)
        t = qs.get("t", [None])[0]
        base = f"https://www.youtube.com/watch?v={vid}"
        return f"{base}&t={t}" if t else base

    # youtube.com/watch?v=...
    if "youtube.com" in host and u.path.startswith("/watch"):
        qs = parse_qs(u.query)
        v = qs.get("v", [None])[0]
        if not v:
            return url
        keep = {"v": v}
        if "t" in qs and qs["t"]:
            keep["t"] = qs["t"][0]
        return f"https://www.youtube.com/watch?{urlencode(keep)}"

    return url


def _download_youtube_audio(
    url: str,
    out_dir: str,
    *,
    debug_print: bool = False,
    emit: Optional[Emit] = None,
) -> str:
    """
    Download best audio using yt-dlp with a progress hook so we can
    stream 'downloading' progress via SSE.
    Returns the full path of the downloaded audio file.
    """
    # progress hook
    def _hook(d):
        if emit is None:
            return
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            done = d.get("downloaded_bytes") or 0
            pct = None
            if total and total > 0:
                pct = max(0.0, min(100.0, round(done * 100.0 / total, 1)))
            note = None
            if d.get("speed"):
                # bytes/sec → human-ish MB/s
                spd = d["speed"]
                note = f"{spd/1024/1024:.2f} MB/s"
            emit("progress", {"stage": "downloading", "pct": pct, "note": note})
        elif d.get("status") == "finished":
            emit and emit("progress", {"stage": "downloading", "pct": 100.0, "note": "Merging"})

    ytdlp_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "%(title).200s.%(ext)s"),
        "noprogress": not debug_print,  # console progress (we also emit SSE)
        "quiet": not debug_print,
        "nocheckcertificate": True,
        "ignoreerrors": False,
        "retries": 10,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 1,  # reduce flaky throttling
        "noplaylist": True,
        "progress_hooks": [_hook],
        # You can optionally enable cookies if you hit age/region blocks:
        # "cookiesfrombrowser": ("chrome",),  # or ("firefox",)
        # or a cookies file:
        # "cookiefile": "/path/to/cookies.txt",
    }

    with YoutubeDL(ytdlp_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        # Result path logic
        if "requested_downloads" in info and info["requested_downloads"]:
            return info["requested_downloads"][0]["filepath"]

        # Fallback: build from title/ext
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
    Non-streaming path (JSON POST). Kept for compatibility.
    """
    session_dir = _new_session_dir() if keep_files else tempfile.mkdtemp(prefix="stt_")
    try:
        norm_url = _normalize_youtube_url(url)
        audio_path = _download_youtube_audio(norm_url, session_dir, debug_print=debug_print)

        svc = service or STTService(settings or load_settings())

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
    emit: Emit,
) -> None:
    """
    Streaming path used by /transcribe/stream. Emits:
      - progress {stage: "downloading"/"transcribing"/"finalizing", pct?, note?}
      - advise    {error_code: OOM|BUSY, ...}  (raised upstream, but listed here for clarity)
      - done      {transcript, strategy, warnings}
      - server_error {message}
    """
    # session on disk (always keep until we finish, then remove unless keep_files is configured elsewhere)
    keep_files = False  # keep ephemeral unless you want to persist
    session_dir = _new_session_dir() if keep_files else tempfile.mkdtemp(prefix="stt_")

    try:
        # 1) Normalize URL + Download with live progress
        norm_url = _normalize_youtube_url(url)
        emit("progress", {"stage": "downloading", "pct": 0.0})
        audio_path = _download_youtube_audio(norm_url, session_dir, emit=emit)

        # 2) Transcribe (we don't have a clean callback from faster-whisper; we simulate coarse steps)
        emit("progress", {"stage": "transcribing", "pct": 5.0})

        text, strategy, warnings = service.transcribe(
            audio_path,
            lang=lang,
            chunk_size=chunk_size,
            num_speakers=num_speakers,
            hf_token=hf_token,
            safe_mode=safe_mode,
        )

        emit("progress", {"stage": "finalizing", "pct": 98.0})
        # 3) Done
        emit("done", {"transcript": text, "strategy": strategy, "warnings": warnings})

    except (OOMError, BusyError):
        # These are converted to 'advise' by the caller in app.py, so re-raise
        raise
    except Exception as e:
        emit("server_error", {"message": str(e)})
        raise
    finally:
        if not keep_files:
            shutil.rmtree(session_dir, ignore_errors=True)
