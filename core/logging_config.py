import logging
import functools
import sys
import traceback

# Get loggers for different modules
def get_logger(name):
    return logging.getLogger(name)

# Quiz generator logger
quiz_generator_logger = get_logger('core.quiz_generator')

# Monkey patch print function to use logger
original_print = print

def patched_print(*args, **kwargs):
    """
    A replacement for the built-in print function that logs to the quiz_generator_logger
    while still printing to stdout.
    """
    # Call the original print function
    original_print(*args, **kwargs)

    # Log the message
    message = " ".join(str(arg) for arg in args)
    if "Error" in message or "error" in message or "failed" in message or "Failed" in message:
        quiz_generator_logger.error(message)
    else:
        quiz_generator_logger.info(message)

# Apply the monkey patch
sys.modules['builtins'].print = patched_print

def log_exceptions(func):
    """
    A decorator to log exceptions from a function.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            quiz_generator_logger.error(f"Exception in {func.__name__}: {e}", exc_info=True)
            raise
    return wrapper
