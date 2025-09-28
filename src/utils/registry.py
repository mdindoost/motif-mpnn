# src/utils/registry.py
from typing import Callable, Dict

class Registry:
    def __init__(self, name: str):
        self.name = name
        self._store: Dict[str, Callable] = {}

    def register(self, key: str):
        def decorator(fn_or_cls: Callable):
            if key in self._store:
                raise KeyError(f"{self.name} already has key '{key}'")
            self._store[key] = fn_or_cls
            return fn_or_cls
        return decorator

    def get(self, key: str) -> Callable:
        if key not in self._store:
            keys = ', '.join(sorted(self._store.keys())) or '<empty>'
            raise KeyError(f"{self.name} missing key '{key}'. Available: {keys}")
        return self._store[key]

    def __contains__(self, key: str):
        return key in self._store

    def keys(self):
        return self._store.keys()


MODEL_REGISTRY = Registry("MODEL_REGISTRY")
DATASET_REGISTRY = Registry("DATASET_REGISTRY")
