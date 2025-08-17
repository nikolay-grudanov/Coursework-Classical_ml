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
    data = load_data(cfg.get("data_path"))
    data = clean_data(data)
    data = create_features(data)
    save_data(data, cfg.get("processed_data_path"))
    logger.info("Processed data saved")


if __name__ == "__main__":
    main()
