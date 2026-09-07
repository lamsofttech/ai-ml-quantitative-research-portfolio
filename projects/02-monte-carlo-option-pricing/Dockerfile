# Builds the interactive Streamlit app for this project only. The
# repository-root Dockerfile builds the numerical-methods demo; each project
# owns its own container image rather than sharing one environment.
FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN useradd --create-home --uid 10001 researcher
WORKDIR /app

COPY pyproject.toml ./pyproject.toml
COPY src ./src
RUN pip install --no-cache-dir ".[app]"

COPY --chown=researcher:researcher app ./app
USER researcher

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')"

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
