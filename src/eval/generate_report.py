"""Wrapper to generate reports using evaluate_models utilities."""
from pathlib import Path
import logging
import yaml
import sys

# Add src to path
sys.path.append(str(Path(__file__).parents[1]))

from eval.evaluate_models import generate_regression_comparison_report, generate_classification_comparison_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path="configs/model_config.yaml"):
    with open(config_path, "r") as f:
        import yaml as _yaml
        return _yaml.safe_load(f)


def main():
    cfg = load_config()
    # The evaluate functions expect results dict, but main pipeline will already save results via models.save_model_results
    # This wrapper is a simple helper that will try to read results from models/model_results.json and generate reports.
    import json
    p = Path("models/model_results.json")
    if not p.exists():
        logger.warning("models/model_results.json not found. Run training first or provide results file.")
        return
    results = json.loads(p.read_text())
    generate_regression_comparison_report(results, "reports/regression_comparison.md")
    generate_classification_comparison_report(results, "reports/classification_comparison.md")
    logger.info("Reports generated")


if __name__ == "__main__":
    main()
