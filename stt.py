import os
import torch
import whisperx
import GPUtil
import time

base_directory = os.path.dirname(os.path.realpath(__file__))

def get_best_gpu():
    gpus = GPUtil.getGPUs()
    print(f"{len(gpus)} GPUs found.")
    gpu_infos = []
    for gpu in gpus:
        total_mem = gpu.memoryTotal
        used_mem = gpu.memoryUsed
        free_mem = gpu.memoryFree
        info = (
            f"GPU {gpu.id}: {gpu.name} "
            f"— used: {used_mem:.1f}MB / {total_mem:.1f}MB "
            f"(free: {free_mem:.1f}MB)"
        )
        gpu_infos.append((gpu.id, free_mem))
        print(info)
    best_gpu_id = max(gpu_infos, key=lambda x: x[1])[0]
    print(f"\nSelected GPU: {best_gpu_id} (most free memory)")
    return best_gpu_id


def stt(filename, chunk_size=30, num_speakers=None, LANG=None, cache_dir=None, HF_TOKEN=None, save_file=False):
    # ─── CONFIG ────────────────────────────────────────────────────────────────
    cache_dir = cache_dir or os.path.join(base_directory, "hf_cache")
    os.makedirs(cache_dir, exist_ok=True)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_index = get_best_gpu() if torch.cuda.is_available() else None

    # ─── 1) Load Whisper (Faster‑Whisper backend) ────────────────────────────────
    model = whisperx.load_model(
        "large",
        device=device_str,
        device_index=gpu_index,
    )

    # ─── 2) Transcribe audio ────────────────────────────────────────────────────
    result = model.transcribe(
        filename,
        language=LANG,
        print_progress=True,
        chunk_size=chunk_size  # Adjust chunk size for better performance
    )

    # 6) Optionally diarize & assign speakers
    if num_speakers:
        # ─── 3) Load forced‑alignment model ─────────────────────────────────────────
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=device_str,
        )

        # ─── 4) Diarization ────────────────────────────────────────────
        diar_pipeline = whisperx.DiarizationPipeline(
            model_name="pyannote/speaker-diarization",
            use_auth_token=HF_TOKEN,
            device=torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else "cpu"),
            cache_dir=cache_dir
        )
        diarization = diar_pipeline(
            filename,
            min_speakers=num_speakers,
            max_speakers=num_speakers
        )

        # ─── 5) Align Whisper segments to audio ─────────────────────────────────────
        aligned_result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            filename,
            device=device_str,
            return_char_alignments=False
        )

        # ─── 6) Assign speaker labels ───────────────────────────────────────────────
        final_result = whisperx.assign_word_speakers(
            diarization,
            aligned_result
        )
    else:
        # no diarization → treat each aligned segment as “speakerless”
        final_result = " ".join(seg["text"].strip() for seg in result["segments"])

    if save_file:
        out_file = os.path.splitext(filename)[0] + "_transcript.txt" #(f"_{num_speakers}_speakers.txt" if num_speakers else "_transcript.txt")
        with open(out_file, "w") as f:
            if num_speakers:
                for seg in final_result["segments"]:
                    f.write(f"{seg['speaker']} ({seg['start']:05.2f}s)  {seg['text'].strip()}\n")
            else:
                f.write(final_result)

        print(f"Wrote transcript to {out_file}")
        
    return final_result




if __name__ == "__main__":
    
    start_time = time.time()
    
    # ─── CONFIG ────────────────────────────────────────────────────────────────
    input_name    = "2025-07-28"
    ext           = "opus"
    num_speakers  = None

    LANG          = "en"
    HF_TOKEN      = os.getenv("HF_HUB_TOKEN")  # set this in your shell
    filename      = os.path.join(base_directory, f"{input_name}.{ext}")


    final_result = stt(filename, num_speakers=num_speakers, LANG=LANG, HF_TOKEN=HF_TOKEN)  



    print(f"Time taken: {time.time() - start_time:.2f} seconds")
