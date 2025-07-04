"""
Production settings for the project.
Contains settings specific to the production environment.
"""

import os
import dj_database_url
from .base import *

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

# Allow only specific hosts in production
ALLOWED_HOSTS = os.environ.get('ALLOWED_HOSTS', '').split(',')

# Database - require DATABASE_URL in production
DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    raise Exception("DATABASE_URL environment variable is required in production")

DATABASES = {
    'default': dj_database_url.parse(DATABASE_URL)
}

# Redis configuration
REDIS_URL = os.environ.get('REDIS_URL')
if not REDIS_URL:
    raise Exception("REDIS_URL environment variable is required in production")

# Cache configuration
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': REDIS_URL,
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
            'PASSWORD': os.environ.get('REDIS_PASSWORD'),
        }
    }
}

# Celery configuration
CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL

# ChromaDB configuration
CHROMA_HOST = os.environ.get('CHROMA_HOST')
CHROMA_PORT = os.environ.get('CHROMA_PORT')

# Security settings
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_HSTS_SECONDS = 31536000  # 1 year
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_BROWSER_XSS_FILTER = True
X_FRAME_OPTIONS = 'DENY'

# Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'level': 'ERROR',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': BASE_DIR / 'logs/quizbuilder.log',
            'formatter': 'verbose',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': 'ERROR',
            'propagate': True,
        },
        'core': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}