
This document provides instructions for setting up and running the PDF Quiz Generator application using Docker, both for development and production environments.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Development Environment

The development environment is configured to support hot-reloading, persistent volumes, and easy debugging.

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd quizbuilder
   ```

2. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

3. Build and start the development containers:
   ```bash
   docker-compose up --build
   ```

4. The application will be available at http://localhost:8000

5. Apply migrations (first time only):
   ```bash
   docker-compose exec web python manage.py migrate
   ```

6. Create a superuser (optional):
   ```bash
   docker-compose exec web python manage.py createsuperuser
   ```

### Development Services

The development environment includes the following services:

- **web**: Django application with hot-reloading
- **celery**: Celery worker for background tasks
- **db**: PostgreSQL database
- **redis**: Redis for caching and Celery

### Useful Commands

- Start the development environment:
  ```bash
  docker-compose up
  ```

- Run in detached mode:
  ```bash
  docker-compose up -d
  ```

- View logs:
  ```bash
  docker-compose logs -f
  ```

- Stop the development environment:
  ```bash
  docker-compose down
  ```

- Run Django management commands:
  ```bash
  docker-compose exec web python manage.py <command>
  ```

## Production Environment

The production environment is optimized for efficiency, security, and scalability.

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd quizbuilder
   ```

2. Create a `.env.prod` file in the project root with your production settings:
   ```
   # Django settings
   SECRET_KEY=your-secure-secret-key
   DEBUG=False
   ALLOWED_HOSTS=your-domain.com,www.your-domain.com

   # Database settings
   POSTGRES_PASSWORD=your-secure-database-password
   POSTGRES_USER=postgres
   POSTGRES_DB=postgres

   # Redis settings
   REDIS_PASSWORD=your-secure-redis-password

   # OpenAI API key
   OPENAI_API_KEY=your-openai-api-key
   ```

3. Create the nginx configuration directory:
   ```bash
   mkdir -p nginx
   ```

4. Build and start the production containers:
   ```bash
   docker-compose -f docker-compose.prod.yml up --build -d
   ```

5. Apply migrations:
   ```bash
   docker-compose -f docker-compose.prod.yml exec web python manage.py migrate
   ```

6. Create a superuser (optional):
   ```bash
   docker-compose -f docker-compose.prod.yml exec web python manage.py createsuperuser
   ```

### Production Services

The production environment includes the following services:

- **web**: Django application with Gunicorn and uvicorn workers
- **nginx**: Web server for serving static files and proxying requests
- **db**: PostgreSQL database
- **redis**: Redis for caching and Celery
- **celery**: Celery worker for background tasks

### Useful Commands

- Start the production environment:
  ```bash
  docker-compose -f docker-compose.prod.yml up -d
  ```

- View logs:
  ```bash
  docker-compose -f docker-compose.prod.yml logs -f
  ```

- Stop the production environment:
  ```bash
  docker-compose -f docker-compose.prod.yml down
  ```

- Run Django management commands:
  ```bash
  docker-compose -f docker-compose.prod.yml exec web python manage.py <command>
  ```

## Using Celery for Background Tasks

The application is configured to use Celery for background tasks, such as processing PDFs asynchronously.

### Development

In development, the Celery worker is automatically started as a separate service in the docker-compose.yml file. You don't need to manually start it.

If you need to see the Celery worker logs, you can run:

```bash
docker-compose logs -f celery
```

If you need to restart the Celery worker, you can run:

```bash
docker-compose restart celery
```

### Production

In production, the Celery worker is automatically started as a separate service.

### Example Usage

```python
from core.tasks import process_pdf_async

# Start an asynchronous task
result = process_pdf_async.delay(pdf_id, num_questions, difficulty)

# Get the task ID
task_id = result.id

# Check the status later
from celery.result import AsyncResult
task_result = AsyncResult(task_id)
if task_result.ready():
    print(task_result.result)
```

## Customization

### Scaling

To scale the number of Celery workers in production:

```bash
docker-compose -f docker-compose.prod.yml up -d --scale celery=3
```

### Memory Limits

You can set memory limits for containers in the docker-compose files:

```yaml
services:
  web:
    deploy:
      resources:
        limits:
          memory: 1G
```

### Persistent Data

All data is stored in Docker volumes for persistence:

- **postgres_data**: PostgreSQL data
- **redis_data**: Redis data
- **static_files**: Static files
- **media_files**: Media files (uploaded PDFs)
