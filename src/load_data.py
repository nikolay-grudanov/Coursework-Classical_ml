"""Deprecated compatibility wrapper for the canonical data loader.

This module delegates load/save operations to the canonical implementation
located at ``src/data/load_data.py``. Keep this thin wrapper to preserve
backwards compatibility for scripts that import ``src.load_data``.
"""

import importlib
import warnings
import logging

logger = logging.getLogger(__name__)

_real = None


def _ensure_real():
    """Import and cache the canonical loader module.

    Tries both package-style and relative import paths to be resilient
    to different invocation contexts (scripts vs package).
    """
    global _real
    if _real is not None:
        return _real
    candidates = ["src.data.load_data", "data.load_data"]
    for c in candidates:
        try:
            _real = importlib.import_module(c)
            logger.debug(f"Using data loader: {c}")
            return _real
        except Exception:
            continue
    warnings.warn("Could not import canonical data loader (src/data/load_data.py).")
    raise ImportError("Canonical data loader not found")


def load_data(filepath: str):
    mod = _ensure_real()
    return mod.load_data(filepath)


def save_data(data, path: str):
    mod = _ensure_real()
    return mod.save_data(data, path)


__all__ = ["load_data", "save_data"]
