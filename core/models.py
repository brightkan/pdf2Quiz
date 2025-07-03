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
    pdf = models.ForeignKey(UploadedPDF, on_delete=models.CASCADE, related_name='quizzes')
    created_at = models.DateTimeField(auto_now_add=True)
    num_questions = models.IntegerField()
    difficulty = models.CharField(max_length=10)

    def __str__(self):
        return f"Quiz {self.id} - {self.created_at}"

    def calculate_score(self, user_answers):
        correct_count = 0
        total_questions = self.questions.count()

        for question_id, selected_option in user_answers.items():
            try:
                question = self.questions.get(id=question_id)
                if question.correct_option == selected_option:
                    correct_count += 1
            except Question.DoesNotExist:
                continue

        if total_questions > 0:
            score_percentage = (correct_count / total_questions) * 100
            return {
                'correct': correct_count,
                'total': total_questions,
                'percentage': score_percentage
            }
        return {'correct': 0, 'total': 0, 'percentage': 0}

    def get_weak_topics(self, user_answers):
        weak_topics = []

        for question_id, selected_option in user_answers.items():
            try:
                question = self.questions.get(id=question_id)
                if question.correct_option != selected_option:
                    if question.topic and question.topic not in weak_topics:
                        weak_topics.append(question.topic)
            except Question.DoesNotExist:
                continue

        return weak_topics

class Question(models.Model):
    quiz = models.ForeignKey(Quiz, on_delete=models.CASCADE, related_name='questions')
    text = models.TextField()
    option_a = models.TextField()
    option_b = models.TextField()
    option_c = models.TextField()
    option_d = models.TextField()
    correct_option = models.CharField(max_length=1)
    topic = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return f"Question {self.id} for Quiz {self.quiz_id}"
