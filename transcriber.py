import os
import shutil
import uuid
import tempfile
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
from urllib.parse import urlparse, parse_qs, urlencode

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


def _download_youtube_audio(url: str, out_dir: str, debug_print: bool = False) -> str:
    """
    Download best audio using yt-dlp.
    Returns the full path of the downloaded audio file.
    """
    ytdlp_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "%(title).200s.%(ext)s"),
        "noprogress": not debug_print,
        "quiet": not debug_print,
        "nocheckcertificate": True,
        "ignoreerrors": False,
        "retries": 10,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 1,
        "noplaylist": True,
    }

    with YoutubeDL(ytdlp_opts) as ydl:
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
    JSON POST path: download then transcribe.
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
