import json
import os
import logging

logger = logging.getLogger(__name__)

class AutoTuner:
    """
    Persistent Auto-Tuning Cache.
    Saves optimal kernel configs to disk.
    """
    def __init__(self, cache_file="~/.inference_pio/tuning_cache.json"):
        self.cache_file = os.path.expanduser(cache_file)
        self.cache = {}
        self.load()

    def load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load tuning cache: {e}")

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Failed to save tuning cache: {e}")

    def get_best_config(self, kernel_name, shapes):
        key = f"{kernel_name}_{shapes}"
        return self.cache.get(key)

    def set_best_config(self, kernel_name, shapes, config):
        key = f"{kernel_name}_{shapes}"
        self.cache[key] = config
        self.save()
