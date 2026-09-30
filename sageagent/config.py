"""Experiment configuration: one YAML file, overridable from the command line."""

from __future__ import annotations

import argparse
import os
from typing import Any, Iterable, Sequence

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config(dict):
    """Nested dictionary with attribute access (``cfg.agent.temperature``)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as err:
            raise AttributeError(f"config has no key '{key}'") from err

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @classmethod
    def wrap(cls, obj: Any) -> Any:
        if isinstance(obj, dict):
            return cls({k: cls.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls.wrap(v) for v in obj]
        return obj

    def to_dict(self) -> dict:
        def unwrap(obj):
            if isinstance(obj, dict):
                return {k: unwrap(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [unwrap(v) for v in obj]
            return obj
        return unwrap(self)


def _apply_override(tree: dict, item: str) -> None:
    """Apply one ``dotted.key=value`` override; the value is parsed as YAML."""
    if "=" not in item:
        raise ValueError(f"override must look like key.path=value, got '{item}'")
    key, raw = item.split("=", 1)
    parts = key.strip().split(".")
    node = tree
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise KeyError(f"unknown config section '{part}' in override '{item}'")
        node = node[part]
    if parts[-1] not in node:
        raise KeyError(f"unknown config key '{key}' in override '{item}'")
    node[parts[-1]] = yaml.safe_load(raw)


def load_config(path: str, overrides: Iterable[str] = ()) -> Config:
    """Load a YAML config and apply command-line overrides."""
    path = resolve_path(path)
    with open(path) as f:
        tree = yaml.safe_load(f)
    for item in overrides:
        _apply_override(tree, item)
    return Config.wrap(tree)


def resolve_path(path: str) -> str:
    """Absolute paths are kept; relative paths are taken from the repository root."""
    path = os.path.expanduser(path)
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def save_config(cfg: Config, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False, allow_unicode=True)


def base_parser(description: str) -> argparse.ArgumentParser:
    """Command-line options shared by all pipeline scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/glioma.yaml", help="experiment config (YAML)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. --set agent.temperature=0.3")
    parser.add_argument("--outer", type=int, nargs="*", default=None,
                        help="outer folds to run (default: all)")
    parser.add_argument("--inner", type=int, nargs="*", default=None,
                        help="inner folds to run (default: all)")
    parser.add_argument("--device", default=None,
                        help="device for the predictor, e.g. cuda:0 or cpu (default: cuda if available)")
    return parser


def select_folds(requested: Sequence[int] | None, n: int) -> list[int]:
    folds = list(range(1, n + 1)) if not requested else sorted(set(requested))
    for fold in folds:
        if not 1 <= fold <= n:
            raise ValueError(f"fold {fold} out of range 1..{n}")
    return folds
