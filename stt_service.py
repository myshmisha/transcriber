import os
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
from jinja2 import ChoiceLoader, FileSystemLoader

from config import load_settings, Settings
from stt_service import STTService, BusyError, OOMError
from transcriber import transcribe_url

AUTH_PASSWORDS = {
    "droop": {"theme": "pink",  "greeting": "Hey Shmisha 🌸"},
    "goofy": {"theme": "light", "greeting": "Hey Sunflower 🌻"},
}

ALLOWED_EXTS = {"mp3","wav","m4a","mp4","mkv","webm","mov","aac","flac","ogg","opus"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

def create_app(config_path: Optional[str] = None) -> Flask:
    settings: Settings = load_settings(config_path)

    BASE_DIR = Path(__file__).resolve().parent
    TEMPLATE_ROOT = BASE_DIR
    TEMPLATE_FALLBACK = BASE_DIR / "templates"
    STATIC_DIR = BASE_DIR / "static"

    app = Flask(
        __name__,
        static_folder=str(STATIC_DIR),
        template_folder=str(TEMPLATE_ROOT),
    )
    app.jinja_loader = ChoiceLoader([
        FileSystemLoader(str(TEMPLATE_ROOT)),
        FileSystemLoader(str(TEMPLATE_FALLBACK)),
    ])

    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-me-please")
    app.config["SESSION_COOKIE_SAMESITE"] = "None"
    app.config["SESSION_COOKIE_SECURE"] = True

    app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH_BYTES", 2 * 1024 * 1024 * 1024))

    gh_pages_origin = os.environ.get("GH_PAGES_ORIGIN", "https://myshmisha.github.io")
    ngrok_url = os.environ.get("NGROK_ORIGIN", "https://your-ngrok-url.ngrok-free.app")
    CORS(
        app,
        origins=[gh_pages_origin, ngrok_url, "http://localhost:5500", "http://127.0.0.1:5500"],
        supports_credentials=True,
        expose_headers=["Content-Type"],
        allow_headers=["Content-Type"],
    )
