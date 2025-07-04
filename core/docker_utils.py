"""
Utility functions for Docker compatibility.

This module provides utility functions to ensure the application works correctly
both inside and outside Docker environments.
"""

import os
from django.conf import settings
import logging

# Get a logger for this module
logger = logging.getLogger('core.docker_utils')

def get_chunks_directory(pdf_id):
    """
    Get the directory path for text chunks, compatible with both Docker and non-Docker environments.

    Args:
        pdf_id: The ID of the PDF

    Returns:
        str: The absolute path to the text chunks directory
    """
    # Use settings.MEDIA_ROOT instead of hardcoded path for Docker compatibility
    chunks_dir = os.path.join(settings.MEDIA_ROOT, f"text_chunks/{pdf_id}")
    os.makedirs(chunks_dir, exist_ok=True)
    logger.debug(f"Created chunks directory: {chunks_dir}")
    return chunks_dir

# This function is a direct replacement for ensure_text_chunks_directory in quiz_generator.py
def ensure_text_chunks_directory(pdf_id):
    """Fast directory creation with minimal error handling."""
    # For Docker compatibility, use settings.MEDIA_ROOT instead of hardcoded /app/media
    return get_chunks_directory(pdf_id)
