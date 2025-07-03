from django.contrib import admin
from django.urls import path
from core.views import quiz_view, interactive_quiz, submit_quiz

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', quiz_view, name='quiz'),
    path('quiz/<int:quiz_id>/', interactive_quiz, name='interactive_quiz'),
    path('quiz/<int:quiz_id>/submit/', submit_quiz, name='submit_quiz'),
]
