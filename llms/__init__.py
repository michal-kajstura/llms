from pathlib import Path

PROJECT_PATH = Path(__file__).parent.parent
STORAGE_DIR = PROJECT_PATH / "storage"
DATASETS_PATH = STORAGE_DIR / "datasets"
METRICS_PATH = PROJECT_PATH / "llms" / "metrics"
