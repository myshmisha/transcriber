import os
from dataclasses import dataclass
from typing import List, Optional

try:
    import yaml
except ImportError:
    yaml = None

DEFAULTS = {
    "model_size": "large",
    "max_concurrency": 1,
    "cache_dir": "./hf_cache",
    "allow_cpu_fallback": True,
    "acquire_timeout_seconds": 2,
    "safe_chunk_size": 10,
    "gpu": {
        "policy": "best",   # best | fixed | cpu
        "id": None,
        "exclude": [],
    },
}

@dataclass
class GPUConfig:
    policy: str = "best"
    id: Optional[int] = None
    exclude: List[int] = None

@dataclass
class Settings:
    model_size: str = "large"
    max_concurrency: int = 1
    cache_dir: str = "./hf_cache"
    allow_cpu_fallback: bool = True
    acquire_timeout_seconds: int = 2
    safe_chunk_size: int = 10
    gpu: GPUConfig = GPUConfig()

def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_settings(config_path: Optional[str] = None) -> Settings:
    data = DEFAULTS
    if config_path is None:
        guess = os.path.join(os.getcwd(), "settings.yml")
        if os.path.isfile(guess):
            config_path = guess
    if config_path and yaml:
        with open(config_path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        data = _deep_merge(DEFAULTS, user)

    gpu = data.get("gpu", {})
    return Settings(
        model_size=data.get("model_size", "large"),
        max_concurrency=int(data.get("max_concurrency", 1)),
        cache_dir=data.get("cache_dir", "./hf_cache"),
        allow_cpu_fallback=bool(data.get("allow_cpu_fallback", True)),
        acquire_timeout_seconds=int(data.get("acquire_timeout_seconds", 2)),
        safe_chunk_size=int(data.get("safe_chunk_size", 10)),
        gpu=GPUConfig(
            policy=gpu.get("policy", "best"),
            id=gpu.get("id", None),
            exclude=gpu.get("exclude", []) or [],
        ),
    )
