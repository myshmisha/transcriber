import os
import shutil
import uuid
import tempfile
import time
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List

from yt_dlp import YoutubeDL
from stt_service import STTService
from config import Settings, load_settings

SESSIONS_ROOT = os.path.join(os.path.dirname(__file__), "sessions")
os.makedirs(SESSIONS_ROOT, exist_ok=True)

def _new_session_dir() -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    sid = f"{stamp}-{uuid.uuid4().hex[:8]}"
    path = os.path.join(SESSIONS_ROOT, sid)
    os.makedirs(path, exist_ok=True)
    return path

def _download_youtube_audio(url: str, out_dir: str, debug_print: bool = False) -> str:
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "%(title).200s.%(ext)s"),
        "noprogress": not debug_print,
        "quiet": not debug_print,
        "nocheckcertificate": True,
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
) -> Tuple[str, Optional[str], Dict[str, Any], List[str], Dict[str, float]]:
    """
    Returns: text, session_dir|None, strategy, warnings, timing
    timing: {download_sec, transcribe_sec, total_sec}
    """
    session_dir = _new_session_dir() if keep_files else tempfile.mkdtemp(prefix="stt_")
    timing = {"download_sec": 0.0, "transcribe_sec": 0.0, "total_sec": 0.0}

    t0 = time.time()
    audio_path = _download_youtube_audio(url, session_dir, debug_print=debug_print)
    t1 = time.time()

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
        t2 = time.time()

        timing["download_sec"] = round(t1 - t0, 3)
        timing["transcribe_sec"] = round(t2 - t1, 3)
        timing["total_sec"] = round(t2 - t0, 3)

        if keep_files:
            return text, session_dir, strategy, warnings, timing
        else:
            shutil.rmtree(session_dir, ignore_errors=True)
            return text, None, strategy, warnings, timing
    except Exception:
        if not keep_files:
            shutil.rmtree(session_dir, ignore_errors=True)
        raise
