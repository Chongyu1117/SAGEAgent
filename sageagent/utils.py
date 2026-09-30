"""Small shared helpers: seeding, devices, logging and JSON I/O."""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any

import numpy as np
import torch

try:
    import fcntl
except ImportError:  # Windows: no advisory locks; do not run folds in parallel there
    fcntl = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fold_seed(base: int, outer: int, inner: int) -> int:
    """Distinct, reproducible seed for each (outer, inner) pipeline."""
    return base + 100 * outer + inner


def resolve_device(spec: str | None) -> str:
    if spec is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"device '{spec}' requested but CUDA is not available")
    return spec


def get_logger(name: str = "sageagent") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def save_json(obj: Any, path: str, indent: int | None = 2) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=_to_builtin, ensure_ascii=False)


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def update_json(path: str, entries: dict) -> None:
    """Merge entries into a JSON object on disk.

    Runs over different folds (e.g. one outer fold per GPU) share one file, so the
    read-modify-write holds a lock on the folder and the file is replaced atomically.
    """
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)
    fd = os.open(folder, os.O_RDONLY)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        data = load_json(path) if os.path.exists(path) else {}
        data.update(entries)
        tmp = f"{path}.{os.getpid()}.tmp"
        save_json(dict(sorted(data.items())), tmp)
        os.replace(tmp, path)
    finally:
        os.close(fd)
