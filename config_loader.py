"""Load config.yaml with sensible defaults and deep-merge user overrides."""

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
    """Recursively merge override into base, preserving nested dicts."""
    r = base.copy()
    for k, v in override.items():
        if k in r and isinstance(r[k], dict) and isinstance(v, dict):
            r[k] = _deep_merge(r[k], v)
        else:
            r[k] = v
    return r


_DEFAULTS = {
    "vault_dir":    r"D:\obsidian\LLM_Wiki",
    "uploads_tmp":  ".tmp_uploads",
    "db_path":      ".state/jobs.db",
    "ollama_url":   "http://localhost:11434",
    "ocr_model":    "glm-ocr:latest",
    "llm_model":    "qwen2.5:7b-instruct",
    "ocr": {
        "max_concurrency":     2,
        "dpi":                 200,
        "max_image_dimension": 1800,
        "keep_alive":          0,
    },
    "wiki": {
        "max_source_chars":     16000,
        "context_pages":        6,
        "auto_create_entities": True,
        "auto_create_concepts": True,
        "max_entity_pages":     3,
        "max_concept_pages":    3,
        "keep_alive":           300,
    },
    "categories": [
        {"name": "AI & Machine Learning", "slug": "ai-ml",
         "keywords": ["machine learning", "deep learning", "neural network",
                      "LLM", "AI", "transformer", "NLP", "computer vision"]},
        {"name": "Science & Research", "slug": "science",
         "keywords": ["physics", "chemistry", "biology", "mathematics",
                      "research", "experiment"]},
        {"name": "Technology", "slug": "technology",
         "keywords": ["software", "hardware", "programming", "API", "database"]},
        {"name": "Health & Medicine", "slug": "health",
         "keywords": ["medical", "nutrition", "fitness", "disease", "treatment"]},
        {"name": "Business & Finance", "slug": "business",
         "keywords": ["economics", "management", "investing", "market", "finance"]},
        {"name": "Philosophy & Psychology", "slug": "philosophy",
         "keywords": ["ethics", "consciousness", "cognition", "philosophy", "psychology"]},
        {"name": "Arts & Humanities", "slug": "arts",
         "keywords": ["history", "literature", "art", "music", "culture"]},
        {"name": "Personal", "slug": "personal",
         "keywords": ["personal", "journal", "notes", "goals", "habits"]},
    ],
    "search": {
        "k1": 1.5,
        "b": 0.75,
        "min_term_length": 2,
        "max_snippets": 3,
    },
    "server": {
        "host": "0.0.0.0",
        "port": 7337,
        "health_check_interval": 60,
    },
}


def load_config(path: str = "config.yaml") -> dict:
    p = Path(path)
    override = _load_yaml(p) if p.exists() else {}
    return _deep_merge(_DEFAULTS, override)


CFG = load_config()
