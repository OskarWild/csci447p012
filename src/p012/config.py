"""Load configs/experiment.yaml and resolve project paths."""
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def load_config(path: str = "configs/experiment.yaml") -> dict:
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Absolute path for a key under `paths:` in the config."""
    return ROOT / load_config()["paths"][key]
