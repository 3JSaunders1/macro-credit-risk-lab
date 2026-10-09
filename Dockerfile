# Reproducible environment for the Macro-Driven Credit Risk Lab
FROM python:3.11-slim-bookworm

RUN apt-get update \
 && apt-get install -y --no-install-recommends make \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

# Default: run the test suite. docker-compose overrides this to serve the API and dashboard.
CMD ["python", "-m", "pytest", "-q"]
