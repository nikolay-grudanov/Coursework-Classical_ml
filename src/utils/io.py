"""Simple IO utilities used by the coursework pipeline."""
import json
from pathlib import Path
from typing import Any


def append_jsonl(obj: Any, path: str) -> None:
    """Append a JSON-serializable object as one line in JSONL file.

    Creates parent directory if necessary.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as fh:
        fh.write(json.dumps(obj) + "\n")


def read_jsonl(path: str):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with p.open() as fh:
        for line in fh:
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out
