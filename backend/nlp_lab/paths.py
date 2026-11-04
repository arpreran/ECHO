from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = ROOT / "datasets"
PROCESSED_DIR = DATASETS_DIR / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "experiments" / "results"

SEED = 42
