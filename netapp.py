# netapp.py

import argparse
from transcriber import transcribe_url

def handle_text(transcript: str):
    """
    Replace this with whatever you need to do with the text:
    - save to a DB
    - run NLP
    - send it somewhere
    - etc.
    """
    print("=== Got transcript: ===")
    print(transcript)  # preview first 500 chars
    # e.g. write to file:
    with open("deleteme/latest_transcript.txt", "w", encoding="utf-8") as out:
        out.write(transcript)
    print("Saved to deleteme/latest_transcript.txt")

def main():
    p = argparse.ArgumentParser(description="NetApp caller for transcriber")
    p.add_argument("youtube_url", help="YouTube video URL")
    p.add_argument("--lang", help="Language code, e.g. en or ar")
    p.add_argument("--keep-files", action="store_true", help="If set, leaves the session folder on disk")
    p.add_argument("--chunk_size", type=int, default=15, help="Chunk size for transcription (default: 15 seconds)")
    p.add_argument("--debug_print", action="store_true", help="Verbose output from transcriber")
    args = p.parse_args()

    # 1) call the transcriber
    text, session = transcribe_url(
        args.youtube_url,
        lang=args.lang,
        chunk_size=args.chunk_size,
        keep_files=args.keep_files,
        debug_print=args.debug_print
    )

    # 2) do whatever you like with the transcript
    handle_text(text)

    if args.keep_files:
        print("Session files kept in:", session)

if __name__ == "__main__":
    main()
