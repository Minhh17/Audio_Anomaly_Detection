import json
from pathlib import Path
from typing import Any, Dict

class ConfigManager:
    _instance = None
    _config = None

    def __init__(self):
        if self._config is None:
            self._load_default_config()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance


    def _load_default_config(self):
        """Load default configuration from JSON file"""
        default_path = "/home/haiminh/config.json"
        with open(default_path, 'r') as f:
            self._config = json.load(f)

    def update_from_file(self, config_path: str):
        """Update configuration from another JSON file"""
        with open(config_path, 'r') as f:
            update_config = json.load(f)
            self._deep_update(self._config, update_config)

    def _deep_update(self, d: Dict, u: Dict) -> Dict:
        """Recursively update nested dictionary"""
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from config using dot notation"""
        try:
            value = self._config
            for k in key.split('.'):
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    @property
    def config(self) -> Dict:
        """Get full configuration"""
        return self._config.copy()
