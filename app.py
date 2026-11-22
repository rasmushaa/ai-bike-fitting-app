"""Streamlit entrypoint, also usable with an exported requirements.txt."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ai_bike_fitting_app.ui import main

main()
