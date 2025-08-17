"""Wrapper script expected by Makefile to run feature engineering step."""

from pathlib import Path
import logging
import sys

# Add src to path
sys.path.append(str(Path(__file__).parents[1]))

from data.load_data import load_data, save_data
from data.preprocess import clean_data, create_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path="configs/model_config.yaml"):
    with open(config_path, "r") as f:
        import yaml as _yaml

        return _yaml.safe_load(f)


def main():
    cfg = load_config()
    # Resolve environment-like values in config (support ${VAR:-default} and $VAR)
    import os

    def resolve_path(val):
        if not isinstance(val, str):
            return val
        if val.startswith("${") and val.endswith("}") and ":-" in val:
            inner = val[2:-1]
            var, default = inner.split(":-", 1)
            return os.environ.get(var, default)
        return os.path.expanduser(os.path.expandvars(val))

    data = load_data(resolve_path(cfg.get("data_path")))
    data = clean_data(data)
    data = create_features(data)
    save_data(data, resolve_path(cfg.get("processed_data_path")))
    logger.info("Feature engineering completed")


if __name__ == "__main__":
    main()
