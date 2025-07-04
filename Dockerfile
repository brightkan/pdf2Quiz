# Base stage for both development and production
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    debugpy \
    django-debug-toolbar \
    ipython

# Create a non-root user for development
RUN adduser --disabled-password --gecos "" --uid 1000 appuser
RUN mkdir -p /app/media /app/logs && find /app -path "/app/venv" -prune -o -exec chown appuser:appuser {} \;

# Copy project files
COPY . .
RUN find /app -path "/app/venv" -prune -o -exec chown appuser:appuser {} \;

# Command to run the development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

# Production stage
FROM base as production

# Copy project files
COPY . .

# Collect static files
RUN python manage.py collectstatic --noinput

# Create a non-root user to run the application
RUN adduser --disabled-password --gecos "" appuser
RUN mkdir -p /app/logs && find /app -path "/app/venv" -prune -o -exec chown appuser:appuser {} \;
USER appuser

# Command to run the production server with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "3", "--worker-class", "uvicorn.workers.UvicornWorker", "project.asgi:application"]
