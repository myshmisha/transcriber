# netapp.py
import argparse
import requests
from transcriber import transcribe_url

def handle_text(transcript: str):
    print("=== Got transcript: ===")
    print(transcript[:500] + ("..." if len(transcript) > 500 else ""))
    with open("deleteme/latest_transcript.txt", "w", encoding="utf-8") as out:
        out.write(transcript)
    print("Saved to deleteme/latest_transcript.txt")

def main():
    p = argparse.ArgumentParser(description="Client for the transcriber")
    p.add_argument("youtube_url", help="YouTube video URL")
    p.add_argument("--endpoint", help="HTTP endpoint, e.g. http://localhost:8000/transcribe")
    p.add_argument("--lang", default="en")
    p.add_argument("--chunk_size", type=int, default=15)
    p.add_argument("--keep-files", action="store_true")
    p.add_argument("--debug_print", action="store_true")
    p.add_argument("--num_speakers", type=int)
    p.add_argument("--hf_token")
    args = p.parse_args()

    if args.endpoint:
        r = requests.post(args.endpoint, json={
            "youtube_url": args.youtube_url,
            "language": args.lang,
            "chunk_size": args.chunk_size,
            "keep_files": args.keep_files,
            "debug_print": args.debug_print,
            "num_speakers": args.num_speakers,
            "hf_token": args.hf_token,
        })
        r.raise_for_status()
        payload = r.json()
        text = payload["transcript"]
    else:
        text, _ = transcribe_url(
            args.youtube_url,
            lang=args.lang,
            chunk_size=args.chunk_size,
            keep_files=args.keep_files,
            debug_print=args.debug_print,
            num_speakers=args.num_speakers,
            hf_token=args.hf_token,
        )

    handle_text(text)

if __name__ == "__main__":
    main()
