FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN useradd --create-home --uid 10001 researcher
WORKDIR /portfolio

COPY projects/01-numerical-methods/pyproject.toml /tmp/project/pyproject.toml
COPY projects/01-numerical-methods/src /tmp/project/src
RUN pip install --no-cache-dir "/tmp/project[viz]"

COPY --chown=researcher:researcher projects/01-numerical-methods/examples.py ./examples.py
USER researcher
CMD ["python", "examples.py"]

