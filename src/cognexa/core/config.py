from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

SAMPLE_DOCUMENTS_DIR = DATA_DIR / "samples" / "documents"
OUTPUT_DIR = DATA_DIR / "samples" / "outputs"

SUPPORTED_EXTENSIONS = {".txt", ".md"}
DEFAULT_ENCODING = "utf-8"
APP_NAME = "Cognexa"
APP_VERSION = "0.2.0"