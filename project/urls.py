from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from core.views import quiz_view, interactive_quiz, submit_quiz, quiz_progress, check_task_status

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', quiz_view, name='quiz'),
    path('quiz/<int:quiz_id>/', interactive_quiz, name='interactive_quiz'),
    path('quiz/<int:quiz_id>/submit/', submit_quiz, name='submit_quiz'),
    path('progress/<str:task_id>/', quiz_progress, name='quiz_progress'),
    path('task-status/<str:task_id>/', check_task_status, name='check_task_status'),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
