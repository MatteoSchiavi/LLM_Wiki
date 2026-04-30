"""Load config.yaml with sensible defaults."""
from pathlib import Path

try:
    import yaml
    def _load_yaml(p):
        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
except ImportError:
    def _load_yaml(p):
        return {}


def _deep_merge(base: dict, override: dict) -> dict:
    r = base.copy()
    for k, v in override.items():
        if k in r and isinstance(r[k], dict) and isinstance(v, dict):
            r[k] = _deep_merge(r[k], v)
        else:
            r[k] = v
    return r


_DEFAULTS = {
    "vault_dir":    "vault",
    "uploads_tmp":  ".tmp_uploads",
    "db_path":      ".state/jobs.db",
    "ollama_url":   "http://localhost:11434",
    "ocr_model":    "glm-ocr:latest",
    "llm_model":    "qwen2.5:7b-instruct",
    "ocr": {
        "max_concurrency":     2,
        "dpi":                 200,
        "max_image_dimension": 1800,
    },
    "wiki": {
        "max_source_chars":  12000,   # truncate raw text before sending to LLM
        "context_pages":     6,       # pages to include in query context
    },
}


def load_config(path: str = "config.yaml") -> dict:
    p = Path(path)
    override = _load_yaml(p) if p.exists() else {}
    return _deep_merge(_DEFAULTS, override)


CFG = load_config()
