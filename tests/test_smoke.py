"""A minimal smoke test to ensure pipeline pieces import and run on tiny data."""
import pandas as pd
import os
from pathlib import Path

from src.utils.io import append_jsonl  # type: ignore


def test_append_jsonl(tmp_path):
    p = tmp_path / "reg.jsonl"
    obj = {"a": 1}
    append_jsonl(obj, str(p))
    assert p.exists()
    text = p.read_text()
    assert "\"a\": 1}" in text


def test_process_data_creates_processed(tmp_path, monkeypatch):
    # Create tiny dataframe and fake read_excel behaviour
    df = pd.DataFrame({
        "IC50, mM": [1.0, 2.0],
        "CC50, mM": [10.0, 20.0],
        "feature1": [0.1, 0.2]
    })

    # Monkeypatch pandas.read_excel used in src/data/load_data.py
    import pandas as _pd

    def fake_read_excel(path):
        return df

    monkeypatch.setattr(_pd, "read_excel", fake_read_excel)

    # Ensure config points to our tmp_path files
    cfg = {
        "data_path": "dummy.xlsx",
        "processed_data_path": str(tmp_path / "processed.csv")
    }

    # Write temporary config file and call wrapper
    cfg_file = tmp_path / "cfg.yaml"
    import yaml
    cfg_file.write_text(yaml.dump(cfg))

    # Run the process_data wrapper with monkeypatched config loader
    from src.data.process_data import main as process_main  # type: ignore

    # Monkeypatch load_config to use our cfg
    import src.data.process_data as pdmod  # type: ignore
    pdmod.load_config = lambda config_path="configs/model_config.yaml": cfg

    process_main()
    assert Path(cfg["processed_data_path"]).exists()

