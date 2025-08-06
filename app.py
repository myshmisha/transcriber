from flask import Flask, request, jsonify
from flask_cors import CORS
from transcriber import transcribe_url
import os

app = Flask(__name__)
CORS(app, origins="*")  

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.get_json(force=True)
    url = data.get('youtube_url')
    if not url:
        return jsonify(error="Missing 'youtube_url'"), 400

    lang        = data.get('language', 'en')
    chunk_size  = data.get('chunk_size', 30)
    keep_files  = bool(data.get('keep_files', False))
    debug_print = bool(data.get('debug_print', False))

    try:
        transcript, session_folder = transcribe_url(
            url,
            lang=lang,
            chunk_size=chunk_size,
            keep_files=keep_files,
            debug_print=debug_print
        )
        resp = {"transcript": transcript}
        if keep_files:
            resp["session_folder"] = session_folder
        return jsonify(resp), 200

    except Exception as e:
        app.logger.exception("Transcription failed")
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    debug = os.getenv('FLASK_DEBUG', 'false').lower() in ('1', 'true')
    port  = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=debug)
