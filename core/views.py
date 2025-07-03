from django.shortcuts import render, get_object_or_404, redirect
from django.urls import reverse

# Create your views here.

# core/views.py
from .forms import QuizForm
from .models import Quiz, Question
from .quiz_generator import generate_quiz_from_pdf

def quiz_view(request):
    if request.method == 'POST':
        form = QuizForm(request.POST, request.FILES)
        if form.is_valid():
            pdf = form.cleaned_data['pdf_file']
            num_questions = form.cleaned_data['num_questions']
            difficulty = form.cleaned_data['difficulty']
            result = generate_quiz_from_pdf(pdf, num_questions, difficulty)

            # Extract quiz object and token usage from the result
            quiz = result['quiz']
            token_usage = result['token_usage']

            # Redirect to the interactive quiz page
            return redirect('interactive_quiz', quiz_id=quiz.id)
    else:
        form = QuizForm()
    return render(request, 'form.html', {'form': form})

def interactive_quiz(request, quiz_id):
    quiz = get_object_or_404(Quiz, id=quiz_id)
    questions = quiz.questions.all()

    return render(request, 'quiz.html', {
        'quiz': quiz,
        'questions': questions,
        'token_usage': None  # We'll handle token usage differently in this view
    })

def submit_quiz(request, quiz_id):
    quiz = get_object_or_404(Quiz, id=quiz_id)

    if request.method == 'POST':
        # Collect user answers from the form
        user_answers = {}
        for key, value in request.POST.items():
            if key.startswith('question_'):
                question_id = key.split('_')[1]
                user_answers[question_id] = value

        # Calculate the score
        score = quiz.calculate_score(user_answers)

        # Get weak topics
        weak_topics = quiz.get_weak_topics(user_answers)

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
    return redirect('interactive_quiz', quiz_id=quiz.id)

def generate_feedback(score, weak_topics):
    """Generate personalized feedback based on quiz performance."""
    percentage = score['percentage']

    if percentage >= 80:
        # Good performance
        feedback = {
            'message': "Great job! You've demonstrated a strong understanding of the material.",
            'strengths': "You performed well across most topics in this quiz.",
            'areas_to_improve': "To deepen your understanding, consider reviewing these topics: " + 
                               ", ".join(weak_topics) if weak_topics else "All topics look good!"
        }
    elif percentage >= 60:
        # Average performance
        feedback = {
            'message': "Good effort! You have a decent grasp of the material, but there's room for improvement.",
            'strengths': "You showed understanding in some key areas.",
            'areas_to_improve': "Focus on reviewing these topics: " + 
                               ", ".join(weak_topics) if weak_topics else "Try to review all topics again."
        }
    else:
        # Poor performance
        feedback = {
            'message': "You might need more time with this material. Don't worry, practice makes perfect!",
            'strengths': "You're taking steps to learn, which is the most important part.",
            'areas_to_improve': "We recommend rereading the sections covering: " + 
                               ", ".join(weak_topics) if weak_topics else "All the topics in this quiz."
        }

    return feedback
