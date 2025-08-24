import os
import GPUtil
import torch
import whisperx
from threading import Semaphore
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any

# ---------------- helpers ----------------

class BusyError(Exception):
    def __init__(self, retry_after: int = 2):
        self.retry_after = retry_after
        super().__init__("Service is busy; please retry.")

class OOMError(Exception):
    def __init__(self, suggestion: dict):
        self.suggestion = suggestion
        super().__init__("GPU out of memory")

def _gpu_free_mb(gpu_id: int) -> int:
    for g in GPUtil.getGPUs():
        if g.id == gpu_id:
            return int(g.memoryFree)
    return 0

def _pick_gpu(exclude: List[int]) -> Optional[int]:
    gpus = [g for g in GPUtil.getGPUs() if g.id not in set(exclude)]
    if not gpus:
        return None
    g = max(gpus, key=lambda x: x.memoryFree)
    return g.id

def _choose_compute_type(free_mb: int) -> str:
    if free_mb >= 10000:  # >=10GB
        return "float16"
    if free_mb >= 4000:
        return "int8_float16"
    return "int8"

def _norm_model_name(name: str) -> str:
    if not name:
        return "large-v3"
    n = name.lower().strip()
    alias = {
        "large": "large-v3",
        "largev3": "large-v3",
        "large-v2": "large-v2",
        "medium.en": "medium.en",
        "medium": "medium",
        "small.en": "small.en",
        "small": "small",
        "base.en": "base.en",
        "base": "base",
        "tiny.en": "tiny.en",
        "tiny": "tiny",
    }
    return alias.get(n, n)

CATALOG = [
    "large-v3", "large-v2",
    "medium", "medium.en",
    "small", "small.en",
    "base", "base.en",
    "tiny", "tiny.en",
]

@dataclass
class ModelSlot:
    key: str
    device: str
    gpu_index: Optional[int]
    compute_type: str
    model: Any
    cpu_fallback: Any = None


# ---------------- service ----------------

class STTService:
    """
    Multi-model resident service.

    - Loads the default model (GPU if available) at startup.
    - Other models are lazily loaded on first use on CPU int8 (to avoid VRAM blow-ups).
    - Safe Mode can fall back to CPU automatically.
    """

    def __init__(self, settings):
        self.settings = settings
        os.makedirs(self.settings.cache_dir, exist_ok=True)

        # ---- pick a default device/GPU ----
        if not torch.cuda.is_available() or getattr(settings.gpu, "policy", "best") == "cpu":
            default_device = "cpu"
            default_gpu = None
            free_mb = 0
        elif getattr(settings.gpu, "policy", "best") == "fixed":
            default_gpu = int(getattr(settings.gpu, "id", 0) or 0)
            default_device = "cuda"
            free_mb = _gpu_free_mb(default_gpu)
        else:
            default_gpu = _pick_gpu(getattr(settings.gpu, "exclude", []))
            if default_gpu is None:
                default_device = "cpu"
                free_mb = 0
            else:
                default_device = "cuda"
                free_mb = _gpu_free_mb(default_gpu)

        default_compute = _choose_compute_type(free_mb) if default_device == "cuda" else "int8"

        # expose "default" for /healthz etc. (set BEFORE using in preload)
        self.device = default_device
        self.gpu_index = default_gpu
        self.compute_type = default_compute

        # concurrency
        self._sem = Semaphore(max(1, int(getattr(settings, "max_concurrency", 1))))

        # registry
        self._models: Dict[str, ModelSlot] = {}

        # ---- default model from settings.yml ----
        self.default_key = _norm_model_name(getattr(settings, "model_size", "large-v3"))
        self._models[self.default_key] = ModelSlot(
            key=self.default_key,
            device=default_device,
            gpu_index=default_gpu,
            compute_type=default_compute,
            model=whisperx.load_model(
                self.default_key,
                device=default_device,
                device_index=default_gpu if default_device == "cuda" else None,
                compute_type=default_compute,
            ),
        )

        # ---- optional GPU preloads (same GPU as default) ----
        preload = [_norm_model_name(m) for m in getattr(self.settings, "preload_gpu_models", [])]
        pre_ct = getattr(self.settings, "preload_gpu_compute_type", default_compute)

        for key in preload:
            if key == self.default_key:
                continue
            try:
                dev = "cuda" if default_gpu is not None else "cpu"
                self._models[key] = ModelSlot(
                    key=key,
                    device=dev,
                    gpu_index=default_gpu if dev == "cuda" else None,
                    compute_type=pre_ct if dev == "cuda" else "int8",
                    model=whisperx.load_model(
                        key,
                        device=dev,
                        device_index=default_gpu if dev == "cuda" else None,
                        compute_type=pre_ct if dev == "cuda" else "int8",
                    ),
                )
                print(f"[STT] Preloaded {key} on {dev}{':' + str(default_gpu) if dev=='cuda' else ''}.")
            except Exception as e:
                print(f"[STT] Preload '{key}' on GPU failed: {e}. Will lazy-load on CPU.")

        # cached helpers (per device)
        self._align: Dict[str, Tuple[Any, Any]] = {}   # "cpu" or "cuda" -> (model_a, meta)
        self._diar: Dict[str, Any] = {}                # device string -> pipeline

    # ---------- model info for UI ----------

    def list_models_meta(self) -> List[Dict[str, Any]]:
        items = []
        seen = set()
        ordering = [self.default_key] + [m for m in CATALOG if m != self.default_key]
        for key in ordering:
            key = _norm_model_name(key)
            if key in seen:
                continue
            seen.add(key)
            slot = self._models.get(key)
            if slot:
                items.append({
                    "key": key,
                    "device": slot.device,
                    "gpu_index": slot.gpu_index,
                    "compute_type": slot.compute_type,
                    "loaded": True,
                    "default": (key == self.default_key),
                })
            else:
                items.append({
                    "key": key,
                    "device": "cpu",
                    "gpu_index": None,
                    "compute_type": "int8",
                    "loaded": False,
                    "default": (key == self.default_key),
                })
        return items

    # ---------- helpers ----------

    def _get_or_load(self, key: Optional[str]) -> ModelSlot:
        k = _norm_model_name(key or self.default_key)
        slot = self._models.get(k)
        if slot:
            return slot
        # lazy CPU load
        model = whisperx.load_model(k, device="cpu", compute_type="int8")
        slot = ModelSlot(key=k, device="cpu", gpu_index=None, compute_type="int8", model=model)
        self._models[k] = slot
        return slot

    def _cpu_fallback_for(self, slot: ModelSlot) -> Any:
        if slot.device == "cpu":
            return slot.model
        if slot.cpu_fallback is None:
            slot.cpu_fallback = whisperx.load_model(slot.key, device="cpu", compute_type="int8")
        return slot.cpu_fallback

    def _ensure_align(self, lang_code: str, device_type: str):
        if device_type not in self._align:
            model_a, meta = whisperx.load_align_model(language_code=lang_code, device=device_type)
            self._align[device_type] = (model_a, meta)
        return self._align[device_type]

    def _ensure_diar(self, diar_device: torch.device, hf_token: Optional[str]):
        key = str(diar_device)
        if key not in self._diar:
            self._diar[key] = whisperx.DiarizationPipeline(
                model_name="pyannote/speaker-diarization",
                use_auth_token=hf_token,
                device=diar_device,
                cache_dir=self.settings.cache_dir,
            )
        return self._diar[key]

    def _strategy(self, *, slot: ModelSlot, chunk_size: int, diar_on: str) -> Dict[str, Any]:
        return {
            "model": slot.key,
            "device": slot.device,
            "gpu_index": slot.gpu_index if slot.device == "cuda" else None,
            "chunk_size": chunk_size,
            "compute_type": slot.compute_type,
            "diar_device": diar_on,
        }

    # ---------- main API ----------

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
        model_key: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any], List[str]]:
        """
        Returns (text, strategy, warnings).
        """
        warnings: List[str] = []
        acquire_timeout = getattr(self.settings, "acquire_timeout_seconds", 2) if acquire_timeout is None else acquire_timeout

        if not self._sem.acquire(timeout=acquire_timeout):
            raise BusyError(retry_after=acquire_timeout)

        try:
            slot = self._get_or_load(model_key)

            chosen_chunk = min(chunk_size, getattr(self.settings, "safe_chunk_size", 10)) if safe_mode else chunk_size
            diar_device = torch.device("cpu" if (safe_mode or slot.device == "cpu") else f"cuda:{slot.gpu_index}")

            model_to_use = slot.model
            device_type_for_align = "cuda" if slot.device == "cuda" else "cpu"

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
                    suggestion = {
                        "safe_mode": True,
                        "suggested_chunk_size": max(10, chosen_chunk // 2),
                        "diar_on": "cpu",
                        "message": "GPU memory was insufficient. Try Safe Mode (smaller chunks, diarization on CPU).",
                    }
                    raise OOMError(suggestion)
                if chosen_chunk > 10:
                    return self.transcribe(
                        filename,
                        lang=lang,
                        chunk_size=max(10, chosen_chunk // 2),
                        num_speakers=num_speakers,
                        hf_token=hf_token,
                        safe_mode=True,
                        acquire_timeout=acquire_timeout,
                        model_key=slot.key,
                    )
                if getattr(self.settings, "allow_cpu_fallback", True):
                    warnings.append("GPU OOM → falling back to CPU.")
                    model_to_use = self._cpu_fallback_for(slot)
                    device_type_for_align = "cpu"
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

            if num_speakers:
                align_model, align_meta = self._ensure_align(result["language"], device_type_for_align)
                diar = self._ensure_diar(diar_device, hf_token)
                diar_out = diar(filename, min_speakers=num_speakers, max_speakers=num_speakers)

                aligned = whisperx.align(
                    result["segments"],
                    align_model,
                    align_meta,
                    filename,
                    device=("cuda" if device_type_for_align == "cuda" else "cpu"),
                    return_char_alignments=False,
                )
                final = whisperx.assign_word_speakers(diar_out, aligned)
                lines = [
                    f"{s.get('speaker','SPK')} ({s.get('start',0):05.2f}s)  {s.get('text','').strip()}"
                    for s in final["segments"]
                ]
                text = "\n".join(lines)
                diar_on = "cpu" if str(diar_device) == "cpu" else "gpu"
            else:
                text = " ".join(seg["text"].strip() for seg in result["segments"])
                diar_on = "disabled"

            strategy = self._strategy(slot=slot, chunk_size=chosen_chunk, diar_on=diar_on)
            return text, strategy, warnings

        finally:
            try:
                self._sem.release()
            except Exception:
                pass
