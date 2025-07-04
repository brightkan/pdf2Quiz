# Import logging configuration to apply monkey patching
from .logging_config import quiz_generator_logger, patched_print, log_exceptions

# This will ensure the logging configuration is loaded when the core app is imported
