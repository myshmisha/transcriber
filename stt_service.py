import os
import GPUtil
import torch
import whisperx
from threading import Semaphore
from typing import Optional, List, Tuple, Dict, Any
from config import Settings

class BusyError(Exception):
    def __init__(self, retry_after: int = 2):
        self.retry_after = retry_after
        super().__init__("Service is busy; please retry.")

class OOMError(Exception):
    def __init__(self, suggestion: dict):
        self.suggestion = suggestion
        super().__init__("GPU out of memory")

def _pick_gpu(exclude: List[int]) -> Optional[int]:
    gpus = GPUtil.getGPUs()
    if not gpus:
        return None
    gpus = [g for g in gpus if g.id not in set(exclude)]
    if not gpus:
        return None
    g = max(gpus, key=lambda x: x.memoryFree)
    return g.id

def _gpu_free_mb(gpu_id: int) -> int:
    for g in GPUtil.getGPUs():
        if g.id == gpu_id:
            return int(g.memoryFree)
    return 0

def _choose_compute_type(free_mb: int) -> str:
    if free_mb >= 10000:  # >=10GB
        return "float16"
    if free_mb >= 4000:
        return "int8_float16"
    return "int8"

class STTService:
    """
    Resident WhisperX + guards for concurrency / OOM.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        os.makedirs(self.settings.cache_dir, exist_ok=True)

        # Device selection
        if not torch.cuda.is_available() or settings.gpu.policy == "cpu":
            self.device = "cpu"
            self.gpu_index = None
            free_mb = 0
        elif settings.gpu.policy == "fixed":
            self.gpu_index = int(settings.gpu.id)
            self.device = "cuda"
            free_mb = _gpu_free_mb(self.gpu_index)
        else:  # best
            self.gpu_index = _pick_gpu(settings.gpu.exclude)
            if self.gpu_index is None:
                self.device = "cpu"
                free_mb = 0
            else:
                self.device = "cuda"
                free_mb = _gpu_free_mb(self.gpu_index)

        self.compute_type = _choose_compute_type(free_mb) if self.device == "cuda" else "int8"

        # Concurrency semaphore
        self._sem = Semaphore(max(1, int(settings.max_concurrency)))

        # Load GPU/CPU model (primary)
        self.model = whisperx.load_model(
            settings.model_size,
            device=self.device,
            device_index=self.gpu_index if self.device == "cuda" else None,
            compute_type=self.compute_type,
        )
        # if hasattr(self.model, "model"):
        #     self.model.model.eval()

        # Lazy helpers
        self._align = None
        self._align_meta = None
        self._diar = None
        self._cpu_model = None  # lazy CPU fallback

    def _ensure_align(self, lang_code: str):
        if self._align is None:
            self._align, self._align_meta = whisperx.load_align_model(
                language_code=lang_code, device=self.device
            )

    def _ensure_diar(self, diar_device: torch.device, hf_token: Optional[str]):
        if self._diar is None or str(self._diar.device) != str(diar_device):
            self._diar = whisperx.DiarizationPipeline(
                model_name="pyannote/speaker-diarization",
                use_auth_token=hf_token,
                device=diar_device,
                cache_dir=self.settings.cache_dir,
            )

    def _ensure_cpu_model(self):
        if self._cpu_model is None:
            self._cpu_model = whisperx.load_model(
                self.settings.model_size, device="cpu", compute_type="int8"
            )

    def _strategy(self, *, device_used: str, chunk_size: int, diar_on: str, compute_type: str, gpu_index: Optional[int]) -> Dict[str, Any]:
        return {
            "device": device_used,
            "gpu_index": gpu_index if device_used == "cuda" else None,
            "chunk_size": chunk_size,
            "compute_type": compute_type,
            "diar_device": diar_on,
        }

    def transcribe(
        self,
        filename: str,
        *,
        lang: Optional[str] = None,
        chunk_size: int = 30,
        num_speakers: Optional[int] = None,
        hf_token: Optional[str] = None,
        safe_mode: bool = False,
        acquire_timeout: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any], List[str]]:
        """
        Returns (text, strategy, warnings).
        - On normal mode OOM → raises OOMError with a suggestion dict
        - On Safe Mode: backoff chunk size and optionally fall back to CPU
        """
        warnings: List[str] = []
        acquire_timeout = (self.settings.acquire_timeout_seconds if acquire_timeout is None else acquire_timeout)

        # Try to acquire a slot
        acquired = self._sem.acquire(timeout=acquire_timeout)
        if not acquired:
            raise BusyError(retry_after=acquire_timeout)

        try:
            # Safe mode adjustments
            chosen_chunk = min(chunk_size, self.settings.safe_chunk_size) if safe_mode else chunk_size
            diar_device = torch.device("cpu") if (safe_mode or self.device == "cpu") else torch.device(f"cuda:{self.gpu_index}")
            model_to_use = self.model
            device_used = self.device
            compute_type = self.compute_type

            try:
                with torch.inference_mode():
                    result = model_to_use.transcribe(
                        filename, language=lang, print_progress=False, chunk_size=chosen_chunk
                    )
            except RuntimeError as e:
                msg = str(e)
                is_oom = ("CUDA out of memory" in msg) or ("cublas" in msg and "alloc" in msg)
                if not is_oom:
                    raise
                if not safe_mode:
                    # Return a structured OOM so UI can offer Safe Mode
                    suggestion = {
                        "safe_mode": True,
                        "suggested_chunk_size": max(10, chosen_chunk // 2),
                        "diar_on": "cpu",
                        "message": "GPU memory was insufficient. Try Safe Mode (smaller chunks, diarization on CPU).",
                    }
                    raise OOMError(suggestion)
                # Safe mode path: backoff & fallback to CPU if allowed
                if chosen_chunk > 10:
                    return self.transcribe(
                        filename,
                        lang=lang,
                        chunk_size=max(10, chosen_chunk // 2),
                        num_speakers=num_speakers,
                        hf_token=hf_token,
                        safe_mode=True,
                        acquire_timeout=acquire_timeout,
                    )
                if self.settings.allow_cpu_fallback:
                    warnings.append("GPU OOM → falling back to CPU.")
                    self._ensure_cpu_model()
                    model_to_use = self._cpu_model
                    device_used = "cpu"
                    compute_type = "int8"
                    with torch.inference_mode():
                        result = model_to_use.transcribe(
                            filename, language=lang, print_progress=False, chunk_size=chosen_chunk
                        )
                else:
                    suggestion = {
                        "safe_mode": True,
                        "suggested_chunk_size": 10,
                        "diar_on": "cpu",
                        "message": "GPU OOM. CPU fallback disabled.",
                    }
                    raise OOMError(suggestion)

            # Optional diarization
            if num_speakers:
                self._ensure_align(result["language"])
                self._ensure_diar(diar_device, hf_token)
                diar = self._diar(
                    filename, min_speakers=num_speakers, max_speakers=num_speakers
                )
                aligned = whisperx.align(
                    result["segments"],
                    self._align,
                    self._align_meta,
                    filename,
                    device=device_used if device_used == "cuda" else "cpu",
                    return_char_alignments=False,
                )
                final = whisperx.assign_word_speakers(diar, aligned)
                lines = [
                    f"{s.get('speaker','SPK')} ({s.get('start',0):05.2f}s)  {s.get('text','').strip()}"
                    for s in final["segments"]
                ]
                text = "\n".join(lines)
            else:
                text = " ".join(seg["text"].strip() for seg in result["segments"])

            strategy = self._strategy(
                device_used=device_used,
                chunk_size=chosen_chunk,
                diar_on=("cpu" if num_speakers else "disabled") if device_used == "cpu" or safe_mode else ("gpu" if num_speakers else "disabled"),
                compute_type=compute_type,
                gpu_index=self.gpu_index,
            )
            return text, strategy, warnings

        finally:
            try:
                self._sem.release()
            except Exception:
                pass
