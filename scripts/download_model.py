"""Download and verify the nano pose weights; existing weights are reused."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_bike_fitting_app.model import MODEL_PATH, load_model

if __name__ == "__main__":
    load_model()
    print(f"Pose model ready: {MODEL_PATH}")
