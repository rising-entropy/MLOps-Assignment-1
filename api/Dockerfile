# Build from the repo root:  docker build -f api/Dockerfile -t heart-api .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY api/requirements.txt api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# src/ is needed to unpickle the pipeline (custom FeatureEngineer); model baked into the image
COPY src/ src/
COPY api/ api/
COPY models/heart_model.joblib models/heart_model.joblib

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "api.app:app"]
