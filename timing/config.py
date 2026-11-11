"""Configuration loading and project paths."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    """All filesystem locations, relative to one root so tests can redirect them."""

    root: Path

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def dataset(self) -> Path:
        return self.root / "data" / "processed" / "dataset.csv"

    @property
    def manifest(self) -> Path:
        return self.root / "data" / "processed" / "manifest.json"

    @property
    def qa_report(self) -> Path:
        return self.root / "data" / "processed" / "qa_report.json"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def experiment_log(self) -> Path:
        return self.results / "experiment_log.csv"


@dataclass(frozen=True)
class ModelSpec:
    """One model family as declared in configs/models/<name>.yaml."""

    name: str
    role: str
    feature_set: str
    preprocess: str
    grid: dict[str, list[Any]]

    def configs(self) -> list[dict[str, Any]]:
        """Expand the grid into one parameter dict per model."""
        keys = list(self.grid)
        return [dict(zip(keys, values)) for values in itertools.product(*self.grid.values())]


def load_base_config(paths: Paths) -> dict[str, Any]:
    with open(paths.configs / "base.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_spec(paths: Paths, name: str) -> ModelSpec:
    path = paths.configs / "models" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"No model config at {path}")
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw["name"] != name:
        raise ValueError(f"{path} declares name={raw['name']!r}; expected {name!r}")
    if raw["role"] not in {"candidate", "baseline"}:
        raise ValueError(f"{path}: role must be 'candidate' or 'baseline'")
    if "window" not in raw["grid"]:
        raise ValueError(f"{path}: grid must include 'window'")
    return ModelSpec(raw["name"], raw["role"], raw["feature_set"], raw["preprocess"], raw["grid"])


def list_model_names(paths: Paths) -> list[str]:
    return sorted(p.stem for p in (paths.configs / "models").glob("*.yaml"))


def config_id(params: dict[str, Any]) -> str:
    """Filesystem-safe identifier for one grid point."""
    return "__".join(f"{key}={params[key]}" for key in sorted(params))


def stable_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
