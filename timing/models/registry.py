"""Maps model names used in YAML configs to model classes."""

from __future__ import annotations

from typing import Any, Callable

from timing.models.base import BaseModel

_REGISTRY: dict[str, type[BaseModel]] = {}


def register(name: str) -> Callable[[type[BaseModel]], type[BaseModel]]:
    def decorator(cls: type[BaseModel]) -> type[BaseModel]:
        if name in _REGISTRY:
            raise ValueError(f"Model {name!r} is already registered")
        _REGISTRY[name] = cls
        return cls

    return decorator


def registered_names() -> list[str]:
    return sorted(_REGISTRY)


def build_model(name: str, params: dict[str, Any], seed: int) -> BaseModel:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Registered: {registered_names()}")
    return _REGISTRY[name](seed=seed, **params)
