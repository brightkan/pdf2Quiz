from django.shortcuts import render, get_object_or_404, redirect
from django.urls import reverse
from django.http import JsonResponse
from django.core.cache import cache
import json
import logging

# Create your views here.

# core/views.py
from .forms import QuizForm
from .models import Quiz, Question, UploadedPDF
from .quiz_generator import generate_quiz_from_pdf
from .tasks import process_pdf_async

# Get a logger for this module
logger = logging.getLogger('core.views')

def quiz_view(request):
    if request.method == 'POST':
        logger.info("Quiz form submitted")
        form = QuizForm(request.POST, request.FILES)
        if form.is_valid():
            pdf = form.cleaned_data['pdf_file']
            num_questions = form.cleaned_data['num_questions']
            difficulty = form.cleaned_data['difficulty']
            exam_style = form.cleaned_data['exam_style']
            include_long_answer = form.cleaned_data['include_long_answer']

            logger.info(f"Form is valid. num_questions={num_questions}, difficulty={difficulty}, exam_style={exam_style}, include_long_answer={include_long_answer}")

            # Check if we're using a PDF file or exam style
            if pdf:
                logger.info(f"PDF file provided: {pdf.name}, size: {pdf.size} bytes")
                # Save the PDF file first
                try:
                    pdf_obj = UploadedPDF.objects.create(file=pdf)
                    logger.info(f"PDF saved with ID: {pdf_obj.id}")

                    # Start the asynchronous task with PDF
                    task = process_pdf_async.delay(
                        pdf_id=pdf_obj.id, 
                        num_questions=num_questions, 
                        difficulty=difficulty, 
                        include_long_answer=include_long_answer,
                        exam_style=None  # Ensure exam_style is None when using PDF
                    )
                    logger.info(f"Started PDF processing task with ID: {task.id}")
                except Exception as e:
                    logger.error(f"Error saving PDF or starting task: {e}", exc_info=True)
                    form.add_error(None, f"Error processing PDF: {str(e)}")
                    return render(request, 'form.html', {'form': form})
            else:
                logger.info(f"Using exam style: {exam_style}")
                # Start the asynchronous task with exam style only
                try:
                    task = process_pdf_async.delay(
                        pdf_id=None,
                        num_questions=num_questions, 
                        difficulty=difficulty, 
                        include_long_answer=include_long_answer,
                        exam_style=exam_style
                    )
                    logger.info(f"Started exam style processing task with ID: {task.id}")
                except Exception as e:
                    logger.error(f"Error starting exam style task: {e}", exc_info=True)
                    form.add_error(None, f"Error processing request: {str(e)}")
                    return render(request, 'form.html', {'form': form})

            # Redirect to the progress page
            return redirect('quiz_progress', task_id=task.id)
        else:
            logger.warning(f"Form validation failed. Errors: {form.errors}")
    else:
        form = QuizForm()
        logger.debug("Displaying empty quiz form")
    return render(request, 'form.html', {'form': form})

def quiz_progress(request, task_id):
    """
    Display a progress page for the quiz generation task.
    """
    logger.info(f"Displaying progress page for task: {task_id}")
    return render(request, 'progress.html', {'task_id': task_id})

def check_task_status(request, task_id):
    """
    Check the status of a Celery task and return it as JSON.
    """
    logger.debug(f"Checking status for task: {task_id}")

    # Get the task progress from Redis
    progress = cache.get(f"task_progress_{task_id}")

    if not progress:
        # Task not found or not started
        logger.warning(f"Task progress not found in cache for task: {task_id}")
        return JsonResponse({
            'status': 'PENDING',
            'message': 'Task is pending or not found',
            'percent': 0
        })

    # If the task is complete and has a quiz_id, include it in the response
    if progress.get('status') == 'SUCCESS' and 'quiz_id' in progress:
        logger.info(f"Task {task_id} completed successfully with quiz_id: {progress['quiz_id']}")
        progress['redirect_url'] = reverse('interactive_quiz', kwargs={'quiz_id': progress['quiz_id']})
    else:
        logger.debug(f"Task {task_id} status: {progress.get('status', 'unknown')}, message: {progress.get('message', 'no message')}")

    return JsonResponse(progress)

def interactive_quiz(request, quiz_id):
    logger.info(f"Displaying interactive quiz with ID: {quiz_id}")
    try:
        quiz = get_object_or_404(Quiz, id=quiz_id)
        questions = quiz.questions.all()

        logger.debug(f"Quiz {quiz_id} has {questions.count()} questions")

        # Prepare questions data to include explanation and marks
        questions_data = []
        for q in questions:
            questions_data.append({
                'id': q.id,
                'text': q.text,
                'question_type': q.question_type,
                'option_a': q.option_a,
                'option_b': q.option_b,
                'option_c': q.option_c,
                'option_d': q.option_d,
                'correct_option': q.correct_option,
                'topic': q.topic,
                'explanation': q.explanation,
                'marks': q.marks
            })

        return render(request, 'quiz.html', {
            'quiz': quiz,
            'questions': questions_data,
            'token_usage': None  # We'll handle token usage differently in this view
        })
    except Exception as e:
        logger.error(f"Error displaying quiz {quiz_id}: {e}", exc_info=True)
        # Redirect to home page with error message
        from django.contrib import messages
        messages.error(request, f"Error loading quiz: {str(e)}")
        return redirect('quiz_view')

def submit_quiz(request, quiz_id):
    logger.info(f"Processing quiz submission for quiz ID: {quiz_id}")
    try:
        quiz = get_object_or_404(Quiz, id=quiz_id)

        if request.method == 'POST':
            # Collect user answers from the form
            user_answers = {}
            for key, value in request.POST.items():
                if key.startswith('question_'):
                    question_id = key.split('_')[1]
                    user_answers[question_id] = value

            logger.debug(f"Collected {len(user_answers)} answers for quiz {quiz_id}")

            # Calculate the score
            try:
                score = quiz.calculate_score(user_answers)
                logger.info(f"Quiz {quiz_id} score: {score['percentage']}% ({score['correct']}/{score['total']})")
            except Exception as e:
                logger.error(f"Error calculating score for quiz {quiz_id}: {e}", exc_info=True)
                score = {'percentage': 0, 'correct': 0, 'total': len(user_answers)}

            # Get weak topics
            try:
                weak_topics = quiz.get_weak_topics(user_answers)
                if weak_topics:
                    logger.info(f"Weak topics identified for quiz {quiz_id}: {', '.join(weak_topics)}")
                else:
                    logger.info(f"No weak topics identified for quiz {quiz_id}")
            except Exception as e:
                logger.error(f"Error identifying weak topics for quiz {quiz_id}: {e}", exc_info=True)
                weak_topics = []

            # Determine feedback based on score
            feedback = generate_feedback(score, weak_topics)

            return render(request, 'results.html', {
                'quiz': quiz,
                'score': score,
                'feedback': feedback,
                'weak_topics': weak_topics,
                'user_answers': user_answers
            })

        # If not a POST request, redirect to the quiz page
        logger.debug(f"Non-POST request to submit_quiz for quiz {quiz_id}, redirecting to interactive_quiz")
        return redirect('interactive_quiz', quiz_id=quiz.id)
    except Exception as e:
        logger.error(f"Error processing quiz submission for quiz {quiz_id}: {e}", exc_info=True)
        from django.contrib import messages
        messages.error(request, f"Error processing quiz submission: {str(e)}")
        return redirect('quiz_view')

def generate_feedback(score, weak_topics):
    """Generate personalized feedback based on quiz performance."""
    percentage = score['percentage']
    logger.debug(f"Generating feedback for score: {percentage}%, weak topics: {weak_topics}")

    if percentage >= 80:
        # Good performance
        logger.info(f"Generating feedback for high performance: {percentage}%")
        feedback = {
            'message': "Great job! You've demonstrated a strong understanding of the material.",
            'strengths': "You performed well across most topics in this quiz.",
            'areas_to_improve': "To deepen your understanding, consider reviewing these topics: " + 
                               ", ".join(weak_topics) if weak_topics else "All topics look good!"
        }
    elif percentage >= 60:
        # Average performance
        logger.info(f"Generating feedback for average performance: {percentage}%")
        feedback = {
            'message': "Good effort! You have a decent grasp of the material, but there's room for improvement.",
            'strengths': "You showed understanding in some key areas.",
            'areas_to_improve': "Focus on reviewing these topics: " + 
                               ", ".join(weak_topics) if weak_topics else "Try to review all topics again."
        }
    else:
        # Poor performance
        logger.info(f"Generating feedback for low performance: {percentage}%")
        feedback = {
            'message': "You might need more time with this material. Don't worry, practice makes perfect!",
            'strengths': "You're taking steps to learn, which is the most important part.",
            'areas_to_improve': "We recommend rereading the sections covering: " + 
                               ", ".join(weak_topics) if weak_topics else "All the topics in this quiz."
        }

    return feedback
