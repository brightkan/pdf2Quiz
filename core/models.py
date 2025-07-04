from django.db import models

# Create your models here.

# core/models.py
from django.db import models

class UploadedPDF(models.Model):
    file = models.FileField(upload_to='pdfs/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    extracted_text = models.TextField(blank=True, null=True)

    def save_extracted_text(self, text):
        try:
            print(f"Saving extracted text for PDF {self.id}")
            self.extracted_text = text
            self.save()
            print(f"Successfully saved extracted text for PDF {self.id}")
        except Exception as e:
            print(f"Error saving extracted text for PDF {self.id}: {e}")
            import traceback
            traceback.print_exc()
            # Re-raise the exception to be handled by the caller
            raise

class TokenUsage(models.Model):
    prompt_tokens = models.IntegerField(default=0)
    completion_tokens = models.IntegerField(default=0)
    total_tokens = models.IntegerField(default=0)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Usage at {self.timestamp}: {self.total_tokens} tokens"

class Quiz(models.Model):
    pdf = models.ForeignKey(UploadedPDF, on_delete=models.CASCADE, related_name='quizzes', blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    num_questions = models.IntegerField()
    difficulty = models.CharField(max_length=10)
    exam_style = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return f"Quiz {self.id} - {self.created_at}"

    def calculate_score(self, user_answers):
        correct_count = 0
        total_multiple_choice = 0
        long_answer_count = 0

        for question_id, answer in user_answers.items():
            try:
                question = self.questions.get(id=question_id)

                # Handle multiple-choice questions
                if question.question_type == 'multiple_choice':
                    total_multiple_choice += 1
                    if question.correct_option == answer:
                        correct_count += 1
                # Count long-answer questions separately
                elif question.question_type == 'long_answer':
                    long_answer_count += 1
            except Question.DoesNotExist:
                continue

        if total_multiple_choice > 0:
            score_percentage = (correct_count / total_multiple_choice) * 100
            return {
                'correct': correct_count,
                'total': total_multiple_choice,
                'percentage': score_percentage,
                'long_answer_count': long_answer_count
            }
        return {'correct': 0, 'total': 0, 'percentage': 0, 'long_answer_count': long_answer_count}

    def get_weak_topics(self, user_answers):
        weak_topics = []

        for question_id, answer in user_answers.items():
            try:
                question = self.questions.get(id=question_id)
                # Only consider multiple-choice questions for weak topics
                if question.question_type == 'multiple_choice' and question.correct_option != answer:
                    if question.topic and question.topic not in weak_topics:
                        weak_topics.append(question.topic)
            except Question.DoesNotExist:
                continue

        return weak_topics

class Question(models.Model):
    QUESTION_TYPES = (
        ('multiple_choice', 'Multiple Choice'),
        ('long_answer', 'Long Answer'),
    )

    quiz = models.ForeignKey(Quiz, on_delete=models.CASCADE, related_name='questions')
    text = models.TextField()
    question_type = models.CharField(max_length=20, choices=QUESTION_TYPES, default='multiple_choice')
    option_a = models.TextField(blank=True, null=True)
    option_b = models.TextField(blank=True, null=True)
    option_c = models.TextField(blank=True, null=True)
    option_d = models.TextField(blank=True, null=True)
    correct_option = models.CharField(max_length=1, blank=True, null=True)
    topic = models.CharField(max_length=100, blank=True, null=True)
    explanation = models.TextField(blank=True, null=True)
    marks = models.IntegerField(blank=True, null=True)

    def __str__(self):
        return f"Question {self.id} for Quiz {self.quiz_id}"
