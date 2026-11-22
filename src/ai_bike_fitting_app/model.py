"""One model loader shared by the app and the download script."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "yolo26n-pose.pt"


class ModelError(RuntimeError):
    """The pose model could not be downloaded or loaded."""


def load_model():
    # Keep Ultralytics settings in the checkout instead of the user's home.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache" / "ultralytics"))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    try:
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(2)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(MODEL_PATH))
        if model.task != "pose":
            raise ValueError("Expected a pose model")
        model.to("cpu")
        return model
    except Exception as exc:
        raise ModelError(
            "Could not download or load the pose model. Check the server's internet "
            "connection and retry, or run scripts/download_model.py first."
        ) from exc
