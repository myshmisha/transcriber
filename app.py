import argparse
from typing import Optional

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from config import load_settings, Settings
from stt_service import STTService, BusyError, OOMError
from transcriber import transcribe_url


def create_app(config_path: Optional[str] = None) -> Flask:
    settings: Settings = load_settings(config_path)
    app = Flask(__name__, static_folder="static", template_folder="templates")
    CORS(app, origins="*")

    app.config["_SETTINGS"] = settings
    app.config["_STT_SERVICE"] = STTService(settings)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/transcribe")
    def transcribe():
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
            )
            payload = {"transcript": text, "strategy": strategy, "warnings": warnings}
            if session_dir:
                payload["session_folder"] = session_dir
            return jsonify(payload), 200

        except BusyError as e:
            return jsonify(
                error_code="BUSY",
                message="Transcriber is busy; please retry shortly.",
                retry_after=e.retry_after,
            ), 429

        except OOMError as e:
            return jsonify(
                error_code="OOM",
                message="GPU memory was insufficient for this request.",
                suggestion=e.suggestion,
            ), 507

        except Exception as e:
            app.logger.exception("Transcription failed")
            return jsonify(error_code="SERVER_ERROR", message=str(e)), 500

    @app.get("/healthz")
    def healthz():
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
