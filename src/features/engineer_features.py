"""Wrapper script expected by Makefile to run feature engineering step."""
from pathlib import Path
import logging
import yaml
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
    data = load_data(cfg.get("data_path"))
    data = clean_data(data)
    data = create_features(data)
    save_data(data, cfg.get("processed_data_path"))
    logger.info("Feature engineering completed")


if __name__ == "__main__":
    main()
