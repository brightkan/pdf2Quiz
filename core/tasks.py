from celery import shared_task
from django.conf import settings
from django.core.cache import cache
import logging
from .models import UploadedPDF, Quiz, TokenUsage
from .quiz_generator import extract_text_from_pdf, generate_quiz_from_pdf, generate_quiz_from_exam_style

# Get a logger for this module
logger = logging.getLogger('core.tasks')

def update_progress(task_id, step, total_steps, message):
    """
    Update the progress of a task in Redis.

    Args:
        task_id: The ID of the Celery task
        step: Current step number
        total_steps: Total number of steps
        message: Progress message
    """
    try:
        logger.debug(f"update_progress called: task_id={task_id}, step={step}, total_steps={total_steps}, message={message}")

        # Ensure step is a number and not less than 1
        if not isinstance(step, (int, float)):
            logger.warning(f"Step is not a number: {step}, converting to float")
            step = float(step)

        # Ensure step is not less than 1
        if step < 1:
            logger.warning(f"Step is less than 1: {step}, setting to 1")
            step = 1

        # Ensure total_steps is a number and not less than 1
        if not isinstance(total_steps, (int, float)):
            logger.warning(f"total_steps is not a number: {total_steps}, converting to float")
            total_steps = float(total_steps)

        # Calculate percent, ensuring it's between 0 and 100
        percent = int((step / total_steps) * 100)
        percent = max(0, min(100, percent))

        progress = {
            'step': step,
            'total_steps': total_steps,
            'percent': percent,
            'message': message,
            'status': 'PROGRESS'
        }

        logger.debug(f"Setting progress in cache: {progress}")
        cache_key = f"task_progress_{task_id}"
        cache.set(cache_key, progress, timeout=3600)

        # Verify the progress was set correctly
        cached_progress = cache.get(cache_key)
        if cached_progress:
            logger.debug(f"Progress successfully cached: {cached_progress}")
        else:
            logger.warning(f"Progress not found in cache after setting")
    except Exception as e:
        logger.error(f"Error in update_progress: {e}", exc_info=True)
        # Set a minimal progress object to avoid breaking the UI
        cache.set(f"task_progress_{task_id}", {
            'step': 1,
            'total_steps': 5,
            'percent': 20,
            'message': f"Error updating progress: {str(e)}",
            'status': 'PROGRESS'
        }, timeout=3600)

@shared_task(bind=True)
def process_pdf_async(self, pdf_id=None, num_questions=10, difficulty='medium', include_long_answer=False, exam_style=None):
    """
    Process a request asynchronously to generate a quiz, either from a PDF or based on an exam style.

    Args:
        pdf_id: The ID of the UploadedPDF object (optional, can be None if using exam_style)
        num_questions: Number of questions to generate
        difficulty: Difficulty level of the questions
        include_long_answer: Whether to include long-answer questions
        exam_style: Optional style of a famous exam, test, or book to mimic

    Returns:
        dict: A dictionary containing the quiz ID and token usage
    """
    task_id = self.request.id
    total_steps = 5

    logger.info(f"Starting process_pdf_async task: task_id={task_id}, pdf_id={pdf_id}, exam_style={exam_style}, num_questions={num_questions}, difficulty={difficulty}")

    try:
        # Check if we're using a PDF or exam style
        if pdf_id is not None:
            # Step 1: Retrieve the PDF object
            logger.info(f"Step 1: Retrieving PDF object with ID {pdf_id}")
            update_progress(task_id, 1, total_steps, "Retrieving PDF file...")
            pdf_obj = UploadedPDF.objects.get(id=pdf_id)
            logger.info(f"PDF object retrieved: {pdf_obj}")

            # Step 2: Extract text from PDF
            logger.info(f"Step 2: Extracting text from PDF")
            update_progress(task_id, 2, total_steps, "Extracting text from PDF...")

            try:
                result = generate_quiz_from_pdf(pdf_obj, num_questions, difficulty, task_id=task_id, include_long_answer=include_long_answer, exam_style=None)
            except Exception as e:
                # If we get an error about no text being extracted or NoneType is not iterable, create a default quiz
                if "No text could be extracted from the PDF" in str(e) or "'NoneType' object is not iterable" in str(e):
                    logger.warning(f"Error in quiz generation: {e}, creating a default quiz")
                    update_progress(task_id, 3, total_steps, "Creating default quiz...")

                    # Create a default quiz
                    quiz = Quiz.objects.create(
                        pdf=pdf_obj,
                        num_questions=1,
                        difficulty=difficulty,
                        exam_style=exam_style
                    )

                    # Create a default question
                    from .models import Question
                    Question.objects.create(
                        quiz=quiz,
                        text='No questions could be generated from the PDF content. Please try again with a different PDF.',
                        question_type='multiple_choice',
                        option_a='Try again',
                        option_b='Use a different PDF',
                        option_c='Contact support',
                        option_d='Read the documentation',
                        correct_option='b',
                        topic='Error'
                    )

                    # Create a default token usage
                    token_usage = TokenUsage.objects.create(
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0
                    )

                    result = {
                        'quiz': quiz,
                        'content': 'No text could be extracted from the PDF',
                        'token_usage': token_usage
                    }
                else:
                    # Re-raise other exceptions
                    raise
        else:
            # Using exam style only
            logger.info(f"Using exam style: {exam_style}")
            update_progress(task_id, 1, total_steps, f"Preparing to generate {exam_style} style questions...")

            result = generate_quiz_from_exam_style(exam_style, num_questions, difficulty, task_id=task_id, include_long_answer=include_long_answer)

        # Mark task as complete
        progress = {
            'step': total_steps,
            'total_steps': total_steps,
            'percent': 100,
            'message': "Quiz generation complete!",
            'status': 'SUCCESS',
            'quiz_id': result['quiz'].id
        }
        logger.info(f"Setting final progress: {progress}")
        cache.set(f"task_progress_{task_id}", progress, timeout=3600)

        # Return the quiz ID and token usage
        return {
            'quiz_id': result['quiz'].id,
            'token_usage': {
                'prompt_tokens': result['token_usage'].prompt_tokens,
                'completion_tokens': result['token_usage'].completion_tokens,
                'total_tokens': result['token_usage'].total_tokens
            }
        }

    except UploadedPDF.DoesNotExist:
        error_msg = f"PDF with ID {pdf_id} not found"
        logger.error(error_msg)
        progress = {
            'status': 'FAILURE',
            'message': error_msg
        }
        cache.set(f"task_progress_{task_id}", progress, timeout=3600)
        return {'error': error_msg}
    except Exception as e:
        # Log the error and update progress
        logger.error(f"Error processing PDF: {e}", exc_info=True)

        progress = {
            'status': 'FAILURE',
            'message': f"Error: {str(e)}"
        }
        cache.set(f"task_progress_{task_id}", progress, timeout=3600)
        return {'error': str(e)}
