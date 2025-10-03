import os
import shutil
import uuid
import tempfile
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from stt_service import STTService
from config import Settings, load_settings

BASE_DIR = os.path.dirname(__file__)
DEFAULT_BROWSER = os.environ.get("YT_BROWSER", "chrome")
DEFAULT_BROWSER_PROFILE = os.environ.get(
    "YT_BROWSER_PROFILE", os.path.join(BASE_DIR, "youtube_profile")
)
DEFAULT_COOKIE_FILE = os.environ.get(
    "YT_COOKIE_FILE", os.path.join(BASE_DIR, "yt_cookies.txt")
)
LOGIN_REQUIRED_HINTS = (
    "Sign in to confirm you're not a bot",
    "Sign in to view this video",
    "Sign in to prove you're not a bot",
    "Please sign in to view this content",
)

SESSIONS_ROOT = os.path.join(BASE_DIR, "sessions")
os.makedirs(SESSIONS_ROOT, exist_ok=True)

def _new_session_dir() -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    sid = f"{stamp}-{uuid.uuid4().hex[:8]}"
    path = os.path.join(SESSIONS_ROOT, sid)
    os.makedirs(path, exist_ok=True)
    return path

def _looks_like_login_required(error: Exception) -> bool:
    message = str(error)
    return any(hint in message for hint in LOGIN_REQUIRED_HINTS)


def _download_youtube_audio(
    url: str,
    out_dir: str,
    debug_print: bool = False,
    browser: Optional[str] = None,
    browser_profile: Optional[str] = None,
    cookie_file: Optional[str] = None,
) -> str:
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "%(title).200s.%(ext)s"),
        "noprogress": not debug_print,
        "quiet": not debug_print,
        "nocheckcertificate": True,
    }

    def _extract(extra_opts: Dict[str, Any]) -> Dict[str, Any]:
        merged_opts = {**opts, **extra_opts}
        with YoutubeDL(merged_opts) as ydl:
            return ydl.extract_info(url, download=True)

    cookie_attempts: List[Tuple[str, Dict[str, Any]]] = []
    if cookie_file:
        cookie_attempts.append(("cookie file", {"cookiefile": cookie_file}))

    if browser:
        if browser_profile:
            cookie_attempts.append(
                (
                    "browser profile",
                    {"cookiesfrombrowser": (browser, browser_profile)},
                )
            )
        else:
            cookie_attempts.append(("browser", {"cookiesfrombrowser": (browser,)}))

    try:
        info = _extract({})
    except DownloadError as error:
        if not cookie_attempts or not _looks_like_login_required(error):
            raise

        last_error: Optional[Exception] = error
        for label, extra in cookie_attempts:
            if debug_print:
                print(f"Retrying download using {label} cookies...")
            try:
                info = _extract(extra)
                break
            except DownloadError as fallback_error:
                last_error = fallback_error
        else:
            raise last_error if last_error else error
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
    model_key: Optional[str] = None,
    browser: Optional[str] = None,
    browser_profile: Optional[str] = DEFAULT_BROWSER_PROFILE,
    cookie_file: Optional[str] = DEFAULT_COOKIE_FILE,
) -> Tuple[str, Optional[str], Dict[str, Any], List[str]]:
    resolved_cookie_file = cookie_file if cookie_file and os.path.exists(cookie_file) else None

    resolved_browser_profile = (
        browser_profile if browser_profile and os.path.isdir(browser_profile) else None
    )
    resolved_browser = browser or (DEFAULT_BROWSER if resolved_browser_profile else None)

    session_dir = _new_session_dir() if keep_files else tempfile.mkdtemp(prefix="stt_")
    audio_path = _download_youtube_audio(
        url,
        session_dir,
        debug_print=debug_print,
        browser=resolved_browser,
        browser_profile=resolved_browser_profile,
        cookie_file=resolved_cookie_file,
    )

    svc = service or STTService(settings or load_settings())

    try:
        text, strategy, warnings = svc.transcribe(
            audio_path,
            lang=lang,
            chunk_size=chunk_size,
            num_speakers=num_speakers,
            hf_token=hf_token,
            safe_mode=safe_mode,
            model_key=model_key,
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
