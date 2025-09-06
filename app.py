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

# ---------- Passwords & themes ----------
AUTH_PASSWORDS = {
    "droop": {"theme": "pink",  "greeting": "Hey Shmisha 🌸"},
    "goofy": {"theme": "light", "greeting": "Hey Sunflower 🌻"},
}

ALLOWED_EXTS = {"mp3","wav","m4a","mp4","mkv","webm","mov","aac","flac","ogg","opus"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

def create_app(config_path: Optional[str] = None) -> Flask:
    settings: Settings = load_settings(config_path)

    # Paths (support index.html in root and login.html in templates/)
    BASE_DIR = Path(__file__).resolve().parent
    TEMPLATE_ROOT = BASE_DIR
    TEMPLATE_FALLBACK = BASE_DIR / "templates"
    STATIC_DIR = BASE_DIR / "static"

    app = Flask(
        __name__,
        static_folder=str(STATIC_DIR),
        template_folder=str(TEMPLATE_ROOT),  # root first
    )
    app.jinja_loader = ChoiceLoader([
        FileSystemLoader(str(TEMPLATE_ROOT)),
        FileSystemLoader(str(TEMPLATE_FALLBACK)),
    ])

    # ---- Sessions / cookies (for same-origin use)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-me-please")
    # Cross-site cookies are blocked in modern browsers; keep for same-origin only
    app.config["SESSION_COOKIE_SAMESITE"] = "None"
    app.config["SESSION_COOKIE_SECURE"] = True

    # Upload limit (default 2GB)
    app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH_BYTES", 2 * 1024 * 1024 * 1024))

    # ---- CORS ----
    gh_pages_origin = os.environ.get("GH_PAGES_ORIGIN", "https://myshmisha.github.io")
    CORS(
        app,
        origins=[gh_pages_origin, r"https://.*\.ngrok-free\.app", r"https://.*\.ngrok-free\.dev"],
        supports_credentials=True,
        expose_headers=["Content-Type"],
        allow_headers=["Content-Type", "X-Auth"],
    )

    # App state
    app.config["_SETTINGS"] = settings
    app.config["_STT_SERVICE"] = STTService(settings)

    # ---------- Auth helpers ----------
    def _auth_from_header():
        token = (request.headers.get("X-Auth") or "").strip()
        return AUTH_PASSWORDS.get(token)

    def _is_authed():
        # Session cookie (same-origin) OR header token (GH Pages)
        if session.get("authed"):
            return True, {"theme": session.get("theme", "light"), "greeting": session.get("greeting", "Hello")}
        info = _auth_from_header()
        if info:
            return True, info
        return False, None

    # ---------------- Views ----------------

    @app.get("/")
    def index():
        # Server-rendered only used for same-origin; GH Pages serves its own file.
        ok, info = _is_authed()
        if not ok:
            return render_template("login.html")
        theme = info["theme"]
        greeting = info["greeting"]
        show_greet = bool(session.pop("show_greet", False))
        return render_template("index.html", theme=theme, greeting=greeting, show_greet=show_greet)

    # ----- Auth API (same-origin cookie flow) -----

    @app.get("/auth/check")
    def auth_check():
        ok, info = _is_authed()
        if ok:
            return jsonify(
                authed=True,
                theme=info["theme"],
                greeting=info["greeting"],
            ), 200
        return jsonify(authed=False), 200

    @app.post("/auth/login")
    def auth_login():
        # Cookie session path (same-origin). For GH Pages, client will use X-Auth header instead.
        data = request.get_json(force=True) or {}
        pw = (data.get("password") or "").strip()
        info = AUTH_PASSWORDS.get(pw)
        if not info:
            return jsonify(ok=False, message="Invalid password."), 401
        session["authed"] = True
        session["theme"] = info["theme"]
        session["greeting"] = info["greeting"]
        session["show_greet"] = True
        return jsonify(ok=True), 200

    @app.post("/auth/logout")
    def auth_logout():
        session.clear()
        return jsonify(ok=True), 200

    # ----- Models list for dropdown -----
    @app.get("/models")
    def models():
        svc: STTService = app.config["_STT_SERVICE"]
        return jsonify(svc.list_models_meta()), 200


    # ----- Transcribe by URL -----
    @app.post("/transcribe")
    def transcribe():
        ok, info = _is_authed()
        if not ok:
            return jsonify(error="Unauthorized"), 401

        data = request.get_json(force=True) or {}
        url = data.get("youtube_url")
        if not url:
            return jsonify(error="Missing 'youtube_url'"), 400

        try:
            text, session_dir, strategy, warnings = transcribe_url(
                url,
                lang=data.get("language") or "en",
                chunk_size=int(data.get("chunk_size", 30)),
                keep_files=bool(data.get("keep_files", False)),
                debug_print=bool(data.get("debug_print", False)),
                num_speakers=(int(data["num_speakers"]) if data.get("num_speakers") is not None else None),
                hf_token=data.get("hf_token"),
                service=app.config["_STT_SERVICE"],
                settings=app.config["_SETTINGS"],
                safe_mode=bool(data.get("safe_mode", False)),
                model_key=data.get("model_key"),  # NEW: choose model
            )
            payload = {
                "transcript": text,
                "strategy": strategy,
                "warnings": warnings,
                "theme": info["theme"],
                "greeting": info["greeting"],
            }
            if session_dir:
                payload["session_folder"] = session_dir
            return jsonify(payload), 200

        except BusyError as e:
            return jsonify(error_code="BUSY", message="Transcriber is busy; please retry shortly.", retry_after=e.retry_after), 429
        except OOMError as e:
            return jsonify(error_code="OOM", message="GPU memory was insufficient for this request.", suggestion=e.suggestion), 507
        except Exception as e:
            app.logger.exception("Transcription failed")
            return jsonify(error_code="SERVER_ERROR", message=str(e)), 500

    # ----- Transcribe by File -----
    @app.post("/transcribe_file")
    def transcribe_file():
        ok, info = _is_authed()
        if not ok:
            return jsonify(error="Unauthorized"), 401

        if "file" not in request.files:
            return jsonify(error="Missing file"), 400
        f = request.files["file"]
        if f.filename == "":
            return jsonify(error="Empty filename"), 400
        if not allowed_file(f.filename):
            return jsonify(error=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTS))}"), 400

        lang = request.form.get("language") or "en"
        try:
            chunk_size = int(request.form.get("chunk_size", 30))
        except ValueError:
            chunk_size = 30
        safe_mode = (request.form.get("safe_mode", "0") in ("1", "true", "True", "yes"))
        num_speakers = request.form.get("num_speakers")
        num_speakers = int(num_speakers) if num_speakers else None
        hf_token = request.form.get("hf_token")
        model_key = request.form.get("model_key")

        tmpdir = tempfile.mkdtemp(prefix="stt_upload_")
        try:
            fname = secure_filename(f.filename)
            path = os.path.join(tmpdir, fname)
            f.save(path)

            text, strategy, warnings = app.config["_STT_SERVICE"].transcribe(
                path,
                lang=lang,
                chunk_size=chunk_size,
                num_speakers=num_speakers,
                hf_token=hf_token,
                safe_mode=safe_mode,
                model_key=model_key,  # NEW
            )
            return jsonify({
                "transcript": text,
                "strategy": strategy,
                "warnings": warnings,
                "theme": info["theme"],
                "greeting": info["greeting"],
            }), 200

        except BusyError as e:
            return jsonify(error_code="BUSY", message="Transcriber is busy; please retry shortly.", retry_after=e.retry_after), 429
        except OOMError as e:
            return jsonify(error_code="OOM", message="GPU memory was insufficient for this request.", suggestion=e.suggestion), 507
        except Exception as e:
            app.logger.exception("File transcription failed")
            return jsonify(error_code="SERVER_ERROR", message=str(e)), 500
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @app.get("/healthz")
    def healthz():
        ok, _ = _is_authed()
        if not ok:
            return jsonify(error="Unauthorized"), 401
        svc = app.config["_STT_SERVICE"]
        return {
            "status": "ok",
            "device": svc.device,
            "gpu_index": svc.gpu_index,
            "compute_type": svc.compute_type,
            "model_size": app.config["_SETTINGS"].model_size,
        }, 200

    return app

# Gunicorn target
app = create_app()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to settings.yml")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app(args.config)
    app.run(host=args.host, port=args.port, debug=args.debug)
