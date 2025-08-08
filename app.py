import argparse
import json
import threading
import queue
import time
from typing import Optional
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from flask_cors import CORS

from config import load_settings, Settings
from stt_service import STTService, BusyError, OOMError
from transcriber import (
    transcribe_url,
    run_stream_job,  # new: SSE job runner (download + streaming transcription)
)


def create_app(config_path: Optional[str] = None) -> Flask:
    settings: Settings = load_settings(config_path)
    app = Flask(__name__, static_folder="static", template_folder="templates")
    CORS(app, origins="*")

    app.config["_SETTINGS"] = settings
    app.config["_STT_SERVICE"] = STTService(settings)

    @app.get("/")
    def index():
        return render_template("index.html")

    # ---- existing JSON POST (kept) ----
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

    # ---- NEW: SSE streaming endpoint ----
    @app.get("/transcribe/stream")
    def transcribe_stream():
        """
        Server-Sent Events stream.
        Query params: youtube_url, language, chunk_size, safe_mode (0/1), num_speakers, hf_token
        Emits:
          event: progress  data: {"stage":"downloading|transcribing|finalizing","pct":float,"note": "..."}
          event: note      data: {"message": "..."}
          event: advise    data: {"error_code":"OOM|BUSY", ...}
          event: done      data: {"transcript":"...", "strategy": {...}, "warnings":[...]}
          event: error     data: {"message":"..."}
        """
        args = request.args or {}
        url = args.get("youtube_url")
        if not url:
            return "missing youtube_url", 400

        lang = args.get("language", "en")
        chunk_size = int(args.get("chunk_size", 30))
        safe_mode = args.get("safe_mode", "0") in ("1", "true", "True", "yes")
        num_speakers = int(args["num_speakers"]) if args.get("num_speakers") else None
        hf_token = args.get("hf_token")

        svc: STTService = app.config["_STT_SERVICE"]
        cfg: Settings = app.config["_SETTINGS"]

        q: "queue.Queue[dict]" = queue.Queue()

        def emit(ev_type: str, payload: dict):
            q.put({"type": ev_type, "payload": payload})

        def job():
            try:
                run_stream_job(
                    url=url,
                    lang=lang,
                    chunk_size=chunk_size,
                    num_speakers=num_speakers,
                    hf_token=hf_token,
                    safe_mode=safe_mode,
                    service=svc,
                    settings=cfg,
                    emit=emit,
                )
            except BusyError as e:
                emit("advise", {"error_code": "BUSY", "message": "Transcriber busy.", "retry_after": e.retry_after})
            except OOMError as e:
                emit("advise", {"error_code": "OOM", "message": "GPU OOM.", "suggestion": e.suggestion})
            except Exception as e:
                app.logger.exception("stream job failed")
                emit("error", {"message": str(e)})
            finally:
                # signal end of stream to the generator loop
                q.put({"type": "__end__", "payload": {}})

        t = threading.Thread(target=job, daemon=True)
        t.start()

        def sse_format(event: str, data_obj: dict) -> str:
            return f"event: {event}\n" + "data: " + json.dumps(data_obj, ensure_ascii=False) + "\n\n"

        @stream_with_context
        def gen():
            last_heartbeat = time.time()
            while True:
                try:
                    ev = q.get(timeout=15)  # heartbeat every 15s
                    if ev["type"] == "__end__":
                        break
                    yield sse_format(ev["type"], ev["payload"])
                except queue.Empty:
                    # heartbeat (keeps proxies from closing idle connections)
                    yield ":\n\n"
                # keep a periodic heartbeat even with constant traffic
                if time.time() - last_heartbeat > 30:
                    yield ":\n\n"
                    last_heartbeat = time.time()

        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        }
        return Response(gen(), headers=headers, mimetype="text/event-stream")

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
