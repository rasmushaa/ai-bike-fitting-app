FROM python:3.14-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project

COPY app.py ./
COPY src ./src
COPY scripts ./scripts
COPY .streamlit ./.streamlit
RUN python scripts/download_model.py

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
