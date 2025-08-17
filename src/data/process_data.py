"""Wrapper script expected by the Makefile:

Makefile -> src/data/process_data.py

This script calls existing data loading / preprocessing utilities in the repo
and saves processed data to the configured path.
"""

from pathlib import Path
import logging
import yaml
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parents[1]))

from data.load_data import load_data, save_data
from data.preprocess import clean_data, create_features

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_config(config_path="configs/model_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()

    # Resolve environment-like values in config (support ${VAR:-default} and $VAR)
    import os

    def resolve_path(val):
        if not isinstance(val, str):
            return val
        # Handle ${VAR:-default} pattern
        if val.startswith("${") and val.endswith("}") and ":-" in val:
            inner = val[2:-1]
            var, default = inner.split(":-", 1)
            return os.environ.get(var, default)
        # Fallback to expanding $VAR and ~
        return os.path.expanduser(os.path.expandvars(val))

    data_path = resolve_path(cfg.get("data_path"))
    processed_path = resolve_path(cfg.get("processed_data_path"))

    data = load_data(data_path)
    data = clean_data(data)
    data = create_features(data)
    save_data(data, processed_path)
    logger.info("Processed data saved")


if __name__ == "__main__":
    main()
