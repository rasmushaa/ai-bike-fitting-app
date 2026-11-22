# AI Bike Fit

## Introduction

A small Streamlit demo that uses YOLO pose estimation to show a cyclist's knee and elbow angles. Upload a side-on stationary-cycling video, choose the rider's visible side and processing FPS, then inspect the analysis or download it as an MP4.

Accepts MP4 files up to 100 MB and analyzes the first 30 seconds. Processing runs on CPU at 15 FPS by default. Angles are 2D estimates, not bike-fitting recommendations.

## How to run

Install **Python 3.14** and [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from the repository root:

```bash
uv sync --locked
uv run streamlit run app.py
```

Open <http://localhost:8501>. The model downloads on first analysis; no API key is needed.

## How to develop

Edit the code and save; Streamlit reloads the app.

- `app.py`: entrypoint.
- `src/ai_bike_fitting_app/`: UI, video processing, model loading, and playback.
- `tests/`: automated tests.

Run the tests:

```bash
uv run python -m unittest discover -s tests -v
```

Use `uv add <package>` to add dependencies and commit both `pyproject.toml` and `uv.lock`. Keep videos, model weights, and `.venv` out of Git.

## How to deploy

**Streamlit Community Cloud**

1. Push the project to GitHub.
2. Create an app at <https://share.streamlit.io> using your repository and `app.py` as the entrypoint.
3. Select **Python 3.14** in advanced settings and deploy. No secrets are required.

**Docker**

```bash
docker build -t ai-bike-fit .
docker run --rm -p 8501:8501 ai-bike-fit
```

Open <http://localhost:8501>. The image includes the model weights. After code changes, rebuild the image and recreate the container.
