"""
utils/config_loader.py
──────────────────────
Loads and validates settings.yaml.
All modules call get_config() — never open YAML directly.
"""

from pathlib import Path
from typing import Any, Dict
import yaml
from loguru import logger


_config: Dict[str, Any] = {}


def load_config(path: str = "config/settings.yaml") -> Dict[str, Any]:
    global _config
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        _config = yaml.safe_load(f)
    logger.info(f"Config loaded from {config_path}")
    return _config


def get_config() -> Dict[str, Any]:
    if not _config:
        load_config()
    return _config


def get(key_path: str, default=None) -> Any:
    """Dot-notation access: get('perception.detector.confidence')"""
    cfg = get_config()
    keys = key_path.split(".")
    val = cfg
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val
