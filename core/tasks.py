from celery import shared_task
from django.conf import settings
from django.core.cache import cache
from .models import UploadedPDF, Quiz, TokenUsage
from .quiz_generator import extract_text_from_pdf, generate_quiz_from_pdf

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
        print(f"update_progress called: task_id={task_id}, step={step}, total_steps={total_steps}, message={message}")

        # Ensure step is a number and not less than 1
        if not isinstance(step, (int, float)):
            print(f"Warning: step is not a number: {step}, converting to float")
            step = float(step)

        # Ensure step is not less than 1
        if step < 1:
            print(f"Warning: step is less than 1: {step}, setting to 1")
            step = 1

        # Ensure total_steps is a number and not less than 1
        if not isinstance(total_steps, (int, float)):
            print(f"Warning: total_steps is not a number: {total_steps}, converting to float")
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

        print(f"Setting progress in cache: {progress}")
        cache_key = f"task_progress_{task_id}"
        cache.set(cache_key, progress, timeout=3600)

        # Verify the progress was set correctly
        cached_progress = cache.get(cache_key)
        if cached_progress:
            print(f"Progress successfully cached: {cached_progress}")
        else:
            print(f"Warning: Progress not found in cache after setting")
    except Exception as e:
        print(f"Error in update_progress: {e}")
        # Set a minimal progress object to avoid breaking the UI
        cache.set(f"task_progress_{task_id}", {
            'step': 1,
            'total_steps': 5,
            'percent': 20,
            'message': f"Error updating progress: {str(e)}",
            'status': 'PROGRESS'
        }, timeout=3600)

@shared_task(bind=True)
def process_pdf_async(self, pdf_id, num_questions, difficulty):
    """
    Process a PDF file asynchronously to generate a quiz.

    Args:
        pdf_id: The ID of the UploadedPDF object
        num_questions: Number of questions to generate
        difficulty: Difficulty level of the questions

    Returns:
        dict: A dictionary containing the quiz ID and token usage
    """
    task_id = self.request.id
    total_steps = 5

    print(f"Starting process_pdf_async task: task_id={task_id}, pdf_id={pdf_id}, num_questions={num_questions}, difficulty={difficulty}")

    try:
        # Step 1: Retrieve the PDF object
        print(f"Step 1: Retrieving PDF object with ID {pdf_id}")
        update_progress(task_id, 1, total_steps, "Retrieving PDF file...")
        pdf_obj = UploadedPDF.objects.get(id=pdf_id)
        print(f"PDF object retrieved: {pdf_obj}")

        # Step 2: Extract text from PDF
        print(f"Step 2: Extracting text from PDF")
        update_progress(task_id, 2, total_steps, "Extracting text from PDF...")

        result = generate_quiz_from_pdf(pdf_obj, num_questions, difficulty, task_id=task_id)

        # Mark task as complete
        progress = {
            'step': total_steps,
            'total_steps': total_steps,
            'percent': 100,
            'message': "Quiz generation complete!",
            'status': 'SUCCESS',
            'quiz_id': result['quiz'].id
        }
        print(f"Setting final progress: {progress}")
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
        print(error_msg)
        progress = {
            'status': 'FAILURE',
            'message': error_msg
        }
        cache.set(f"task_progress_{task_id}", progress, timeout=3600)
        return {'error': error_msg}
    except Exception as e:
        # Log the error and update progress
        print(f"Error processing PDF: {e}")
        import traceback
        traceback.print_exc()

        progress = {
            'status': 'FAILURE',
            'message': f"Error: {str(e)}"
        }
        cache.set(f"task_progress_{task_id}", progress, timeout=3600)
        return {'error': str(e)}
