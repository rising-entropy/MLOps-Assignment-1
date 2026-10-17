# Build from the repo root:  docker build -f ui/Dockerfile -t heart-ui .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 API_URL=http://api:8000
WORKDIR /app

COPY ui/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY ui/app.py app.py

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
