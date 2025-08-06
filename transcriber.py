# transcriber.py

import os, subprocess, datetime, shutil
from stt import stt

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
SESSIONS = os.path.join(BASE_DIR, "sessions")
# os.makedirs(SESSIONS, exist_ok=True)

def download_audio(youtube_url, out_path, fmt="wav"):
    cmd = [
        "yt-dlp","-x",
        f"--audio-format={fmt}",
        "--no-cache-dir",
        "-o", f"{out_path}.%(ext)s",
        youtube_url
    ]
    subprocess.run(cmd, check=True)
    return f"{out_path}.{fmt}"

def transcribe_url(youtube_url, lang=None, chunk_size=15, keep_files=False, debug_print=False):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    session_dir = os.path.join(SESSIONS, ts)
    os.makedirs(session_dir)
    try:
        if debug_print:
            print(f"[{ts}] Downloading…")
        audio = download_audio(youtube_url, os.path.join(session_dir, "audio"), fmt="wav")

        if debug_print:
            print(f"[{ts}] Transcribing…")
        _ = stt(audio, num_speakers=None, save_file=True, LANG=lang,
                chunk_size=chunk_size)

        txt_path = os.path.join(session_dir, "audio_transcript.txt")
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text, session_dir

    finally:
        if not keep_files:
            if debug_print:
                print(f"[{ts}] Cleaning up…")
            shutil.rmtree(session_dir, ignore_errors=True)
