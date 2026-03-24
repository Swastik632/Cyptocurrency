from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "dataset.csv"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "reports" / "figures"

ARTIFACTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS / "volatility_model.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"
FEATURE_NAMES_PATH = ARTIFACTS / "feature_names.json"
