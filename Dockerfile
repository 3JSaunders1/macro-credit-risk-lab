FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Default port exposure (API)
EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]