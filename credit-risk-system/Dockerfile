# NOT TESTED: Docker was not available on the machine where this project was built.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 CREDITRISK_CONFIG=/app/config.yaml
WORKDIR /app

COPY pyproject.toml README.md config.yaml ./
COPY src ./src
RUN pip install .

RUN useradd --create-home appuser && mkdir -p data models reports && chown -R appuser /app
USER appuser

EXPOSE 8000
# The model is not baked into the image. Train inside a volume-backed container first:
#   docker run --rm -v creditrisk-state:/app/models -v creditrisk-state-data:/app/data creditrisk python -m creditrisk.train
CMD ["python", "-m", "uvicorn", "creditrisk.api:app", "--host", "0.0.0.0", "--port", "8000"]
